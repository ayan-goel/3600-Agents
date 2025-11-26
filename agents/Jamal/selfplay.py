from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENGINE_ROOT = os.path.join(PROJECT_ROOT, "engine")
if ENGINE_ROOT not in sys.path:
    sys.path.append(ENGINE_ROOT)

from game.board import Board
from game.enums import Direction, MoveType, Result, WinReason, loc_after_direction
from game.game_map import GameMap
from game.trapdoor_manager import TrapdoorManager
from agents.Fluffy.trapdoor_belief import TrapdoorBelief

from .featurizer import make_state_tensors
from .net import ChickenNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _action_index(mv: Tuple[Direction, MoveType]) -> int:
    d, mt = mv
    return int(d) * 3 + int(mt)  # 4*3 = 12 sized head


def _softmax_sample(logits: np.ndarray, legal_indices: List[int], temperature: float, rng: np.random.Generator) -> int:
    x = logits.copy()
    mask = np.full_like(x, fill_value=-1e9)
    for idx in legal_indices:
        mask[idx] = x[idx]
    x = mask
    if temperature <= 0.0:
        return int(np.argmax(x))
    x = x - x.max()
    probs = np.exp(x / max(temperature, 1e-6))
    probs = probs / (probs.sum() + 1e-9)
    return int(rng.choice(len(probs), p=probs))


@dataclass
class PolicyConfig:
    temperature_opening: float = 1.0
    temperature_midgame: float = 0.3
    temperature_endgame: float = 0.0
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25


def _temperature_for_turn(turn: int) -> float:
    if turn < 6:
        return 1.0
    if turn < 24:
        return 0.3
    return 0.0


def jamal_pick_action(
    net: ChickenNet,
    board: Board,
    belief: TrapdoorBelief,
    rng: np.random.Generator,
    cfg: PolicyConfig,
    turn_idx: int,
) -> Tuple[Direction, MoveType]:
    board_tensor, scalar_tensor = make_state_tensors(board, belief)
    with torch.no_grad():
        logits_t, _ = net(board_tensor.to(DEVICE), scalar_tensor.to(DEVICE))
    logits = logits_t[0].detach().cpu().numpy()

    legal_moves = board.get_valid_moves()
    if not legal_moves:
        return Direction.UP, MoveType.PLAIN
    legal_indices = [_action_index(mv) for mv in legal_moves]

    # Root Dirichlet for exploration in early game
    if turn_idx < 12:
        keys = legal_indices
        noise = rng.dirichlet([cfg.dirichlet_alpha] * len(keys))
        for k, n in zip(keys, noise):
            logits[k] = (1.0 - cfg.dirichlet_frac) * logits[k] + cfg.dirichlet_frac * float(n)

    temp = _temperature_for_turn(turn_idx)
    chosen_idx = _softmax_sample(logits, legal_indices, temp, rng)
    # Map back to move
    best = None
    best_score = -1e9
    for mv, idx in zip(legal_moves, legal_indices):
        score = 1.0 if idx == chosen_idx else 0.0
        if score > best_score:
            best = mv
            best_score = score
    assert best is not None
    return best


def play_game_jamal_vs_agent(
    net: ChickenNet,
    opponent_module: str,
    rng: np.random.Generator,
    max_turns: int = 80,
) -> Result:
    """Jamal (PLAYER) vs imported opponent (ENEMY)."""
    game_map = GameMap()
    trap_manager = TrapdoorManager(game_map)
    board = Board(game_map, 360.0, build_history=False)
    spawns = trap_manager.choose_spawns()
    trap_manager.choose_trapdoors()
    board.chicken_player.start(spawns[0], 0)
    board.chicken_enemy.start(spawns[1], 1)

    belief = TrapdoorBelief(game_map)
    opp_mod = importlib.import_module(f"agents.{opponent_module}")
    enemy = opp_mod.PlayerAgent(board, time_left=lambda: 360.0)
    cfg = PolicyConfig()

    turn = 0
    while board.get_winner() is None and turn < max_turns:
        sensory = trap_manager.sample_trapdoors(board.chicken_player.get_location())
        belief.update(board.chicken_player, sensory)
        # Jamal turn
        mv = jamal_pick_action(net, board, belief, rng, cfg, turn)
        if not board.apply_move(mv[0], mv[1], check_ok=True):
            board.set_winner(Result.ENEMY, WinReason.INVALID_TURN)
            break
        if board.get_winner() is not None:
            break

        # Enemy turn
        board.reverse_perspective()
        sensory_e = trap_manager.sample_trapdoors(board.chicken_player.get_location())
        try:
            mv_e = enemy.play(board, sensory_e, time_left=lambda: 360.0)
        except Exception:
            mv_e = (Direction.UP, MoveType.PLAIN)
        if not board.apply_move(mv_e[0], mv_e[1], check_ok=True):
            # Enemy invalid -> our win
            board.set_winner(Result.PLAYER, WinReason.INVALID_TURN)
            board.reverse_perspective()
            break
        board.reverse_perspective()
        turn += 1

    return board.get_winner() or Result.TIE


def evaluate_vs_pool(
    net: ChickenNet,
    opponents: Sequence[str],
    games_per_opp: int,
    rng: np.random.Generator,
) -> float:
    """Return average win rate vs a pool of opponents (Jamal plays PLAYER)."""
    wins = 0
    total = 0
    with torch.no_grad():
        for opp in opponents:
            for _ in range(games_per_opp):
                res = play_game_jamal_vs_agent(net, opp, rng)
                if res == Result.PLAYER:
                    wins += 1
                total += 1
    return wins / max(total, 1)


def finetune_selfplay(
    net: ChickenNet,
    iterations: int = 8,
    games_per_eval_opp: int = 8,
    lr: float = 3e-4,
    batch_size: int = 256,
    epochs: int = 2,
    save_path: str = "jamal_weights.pt",
    seed: Optional[int] = None,
) -> None:
    """Simple policy-only fine-tuning loop: evaluate vs pool; if improved, train on self-play data."""
    rng = np.random.default_rng(seed)
    opponents = ["Fluffy", "Messi"]
    best_rate = evaluate_vs_pool(net, opponents, games_per_eval_opp, rng)
    print(f"Initial win rate vs pool: {best_rate:.1%}")

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for it in range(1, iterations + 1):
        print(f"Iteration {it}/{iterations}")
        # Collect fresh self-play data (Jamal vs Fluffy/Messi evenly)
        boards_list: List[np.ndarray] = []
        scalars_list: List[np.ndarray] = []
        labels: List[int] = []

        for opp in opponents:
            for _ in range(games_per_eval_opp):
                # Play one Jamal turn then record its chosen action as label
                game_map = GameMap()
                trap_manager = TrapdoorManager(game_map)
                board = Board(game_map, 360.0, build_history=False)
                spawns = trap_manager.choose_spawns()
                trap_manager.choose_trapdoors()
                board.chicken_player.start(spawns[0], 0)
                board.chicken_enemy.start(spawns[1], 1)
                belief = TrapdoorBelief(game_map)
                sensory = trap_manager.sample_trapdoors(board.chicken_player.get_location())
                belief.update(board.chicken_player, sensory)
                cfg = PolicyConfig()
                board_tensor, scalar_tensor = make_state_tensors(board, belief)
                with torch.no_grad():
                    logits_t, _ = net(board_tensor.to(DEVICE), scalar_tensor.to(DEVICE))
                logits = logits_t[0].detach().cpu().numpy()
                legal_moves = board.get_valid_moves()
                legal_indices = [_action_index(mv) for mv in legal_moves]
                idx = _softmax_sample(logits, legal_indices, temperature=1.0, rng=rng)
                # Record supervised pair
                boards_list.append(board_tensor.numpy()[0])
                scalars_list.append(scalar_tensor.numpy()[0])
                labels.append(idx)

        if not boards_list:
            print("No training samples collected; skipping.")
            continue

        boards_np = np.stack(boards_list, axis=0)
        scalars_np = np.stack(scalars_list, axis=0)
        labels_np = np.array(labels, dtype=np.int64)

        net.train()
        for ep in range(epochs):
            perm = rng.permutation(len(labels_np))
            for start in range(0, len(labels_np), batch_size):
                sel = perm[start : start + batch_size]
                boards = torch.from_numpy(boards_np[sel]).to(DEVICE)
                scalars = torch.from_numpy(scalars_np[sel]).to(DEVICE)
                y = torch.from_numpy(labels_np[sel]).to(DEVICE)
                optimizer.zero_grad()
                logits, _ = net(boards, scalars)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()
        net.eval()

        rate = evaluate_vs_pool(net, opponents, games_per_eval_opp, rng)
        print(f"  Win rate vs pool: {rate:.1%} (prev best {best_rate:.1%})")
        if rate >= best_rate:
            best_rate = rate
            out_path = os.path.join(os.path.dirname(__file__), save_path)
            torch.save(net.state_dict(), out_path)
            print(f"  Accepted new weights -> {out_path}")
        else:
            print("  Rejected; keeping previous best.")


def main():
    parser = argparse.ArgumentParser(description="Jamal self-play/opponent-pool fine-tuning with eval gating.")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--games-per-opp", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", type=str, default="jamal_weights.pt")
    args = parser.parse_args()

    net = ChickenNet(board_size=8).to(DEVICE)
    weights_path = os.path.join(os.path.dirname(__file__), args.weights)
    if os.path.exists(weights_path):
        net.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        print(f"Loaded weights: {weights_path}")
    else:
        print("No existing weights found; starting from random initialization.")

    finetune_selfplay(
        net,
        iterations=args.iterations,
        games_per_eval_opp=args.games_per_opp,
        lr=args.lr,
        batch_size=args.batch,
        epochs=args.epochs,
        save_path=args.weights,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()



