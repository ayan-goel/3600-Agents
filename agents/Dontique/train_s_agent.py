from __future__ import annotations

import argparse
import math
import os
import random
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = PROJECT_ROOT / "engine"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.append(str(ENGINE_ROOT))

from game.board import Board
from game.enums import Direction, MoveType, Result, WinReason
from game.game_map import GameMap
from game.trapdoor_manager import TrapdoorManager

from .agent import (
    ChickenNet,
    MCTS,
    decode_action,
    encode_action,
    make_state_tensors,
)
from .trapdoor_belief import TrapdoorBelief
from .policy_utils import (
    ActionBiasParams,
    jitter_simulations,
    sample_action_with_bias,
    temperature_for_turn,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_DIRECTIONS = [d for d in Direction]
ALL_MOVETYPES = [m for m in MoveType]
ACTION_SPACE = [(d, m) for d in ALL_DIRECTIONS for m in ALL_MOVETYPES]

def _remap_action_for_flip(idx: int, horizontal: bool, vertical: bool) -> int:
    d, m = ACTION_SPACE[idx]
    new_d = d
    if horizontal:
        if d == Direction.LEFT:
            new_d = Direction.RIGHT
        elif d == Direction.RIGHT:
            new_d = Direction.LEFT
    if vertical:
        if d == Direction.UP:
            new_d = Direction.DOWN
        elif d == Direction.DOWN:
            new_d = Direction.UP
    return encode_action((new_d, m))

def _apply_random_flip(
    boards_np: np.ndarray, pis_np: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    # Randomly apply horizontal and/or vertical flips to the whole batch
    h = rng.random() < 0.5
    v = rng.random() < 0.5
    if not h and not v:
        return boards_np, pis_np
    boards = boards_np.copy()
    if h:
        boards = np.flip(boards, axis=-1).copy()
    if v:
        boards = np.flip(boards, axis=-2).copy()
    boards = np.ascontiguousarray(boards)
    # Remap policy distribution
    remapped = np.zeros_like(pis_np)
    for old in range(pis_np.shape[1]):
        new = _remap_action_for_flip(old, h, v)
        remapped[:, new] = remapped[:, new] + pis_np[:, old]
    remapped = np.ascontiguousarray(remapped)
    return boards, remapped


def play_one_game(
    net: ChickenNet,
    rng: np.random.Generator,
    bias_params: ActionBiasParams,
    max_sims_per_move: int = 64,
    temperature_fn=temperature_for_turn,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Self-play a single game, returning training triples."""
    game_map = GameMap()
    play_time = 360.0
    trap_manager = TrapdoorManager(game_map)
    board = Board(game_map, play_time, build_history=False)
    spawns = trap_manager.choose_spawns()
    trap_manager.choose_trapdoors()
    board.chicken_player.start(spawns[0], 0)
    board.chicken_enemy.start(spawns[1], 1)

    beliefs = [TrapdoorBelief(game_map), TrapdoorBelief(game_map)]
    player_idx = 0
    last_mover = 0
    last_dirs: List[Direction | None] = [None, None]
    second_last_dirs: List[Direction | None] = [None, None]
    since_egg: List[int] = [0, 0]
    turn_idx = 0

    mcts = MCTS(net)
    net.eval()

    trajectory: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []

    while board.get_winner() is None:
        sensory_data = trap_manager.sample_trapdoors(board.chicken_player.get_location())
        current_belief = beliefs[player_idx]
        current_belief.update(board.chicken_player, sensory_data)

        sims_budget = jitter_simulations(max_sims_per_move, rng)
        visit_dist = mcts.run(
            root_board=board,
            root_belief=current_belief,
            max_sims=sims_budget,
            root_dirichlet=(0.30, 0.25),
        )

        board_tensor, scalar_tensor = make_state_tensors(board, current_belief)
        state_np = board_tensor.numpy()[0]
        scalar_np = scalar_tensor.numpy()[0]
        trajectory.append(
            (
                state_np,
                scalar_np,
                visit_dist,
                1.0 if player_idx == 0 else -1.0,
            )
        )

        temp = temperature_fn(turn_idx)
        legal_moves = board.get_valid_moves()
        if not legal_moves:
            break
        legal_indices = [encode_action(move) for move in legal_moves]
        action_idx = sample_action_with_bias(
            board,
            legal_indices,
            visit_dist,
            last_dirs[player_idx],
            second_last_dirs[player_idx],
            since_egg[player_idx],
            current_belief,
            rng,
            ACTION_SPACE,
            bias_params,
            temperature=temp,
        )
        action = decode_action(action_idx)
        valid = board.apply_move(action[0], action[1], check_ok=True)
        if not valid:
            # Penalize illegal move by awarding win to opponent.
            board.set_winner(Result.ENEMY, WinReason.INVALID_TURN)
            last_mover = player_idx
            break

        if action[1] == MoveType.PLAIN:
            second_last_dirs[player_idx] = last_dirs[player_idx]
            last_dirs[player_idx] = action[0]
            since_egg[player_idx] += 1
        elif action[1] == MoveType.EGG:
            second_last_dirs[player_idx] = last_dirs[player_idx]
            last_dirs[player_idx] = None
            since_egg[player_idx] = 0
        else:
            second_last_dirs[player_idx] = last_dirs[player_idx]
            last_dirs[player_idx] = None
            since_egg[player_idx] += 1

        new_loc = board.chicken_player.get_location()
        if trap_manager.is_trapdoor(new_loc):
            board.chicken_player.reset_location()
            board.chicken_enemy.increment_eggs_laid(-1 * board.game_map.TRAPDOOR_PENALTY)
            board.found_trapdoors.add(new_loc)
            for belief in beliefs:
                belief.register_known_trapdoor(new_loc)

        last_mover = player_idx
        if board.get_winner() is not None:
            break

        board.reverse_perspective()
        player_idx = 1 - player_idx
        turn_idx += 1

    winner = board.get_winner()
    if winner is None or winner == Result.TIE:
        final_val = 0.0
    else:
        if winner == Result.PLAYER:
            winning_idx = last_mover
        else:
            winning_idx = 1 - last_mover
        final_val = 1.0 if winning_idx == 0 else -1.0

    training_data = [
        (state_np, scalar_np, pi, final_val * sign)
        for state_np, scalar_np, pi, sign in trajectory
    ]
    return training_data


def train(
    total_iterations: int = 36,
    games_per_iter: int = 64,
    max_sims_per_move: int = 96,
    batch_size: int = 192,
    lr: float = 6e-4,
    weight_decay: float = 1e-4,
    save_path: str = "s_agent_weights.pt",
    eval_games: int = 24,
    eval_threshold: float = 0.5,
    train_epochs: int = 4,
    warmup_iters: int = 8,
    load_existing: bool = True,
) -> None:
    """Main training loop."""
    game_map = GameMap()
    size = game_map.MAP_SIZE
    net = ChickenNet(board_size=size).to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    replay: Deque[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = deque(maxlen=50000)
    bias_params = ActionBiasParams()
    rng = np.random.default_rng()

    if not os.path.isabs(save_path):
        save_path = os.path.join(os.path.dirname(__file__), save_path)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_state: Dict[str, torch.Tensor] | None = None
    if load_existing and os.path.exists(save_path):
        print(f"Loading existing weights from {save_path}")
        state = torch.load(save_path, map_location=DEVICE)
        net.load_state_dict(state)
        best_state = clone_state_dict(state)
    elif not load_existing and os.path.exists(save_path):
        print(f"Ignoring existing weights at {save_path}; starting fresh.")

    for iteration in range(1, total_iterations + 1):
        print(f"Iteration {iteration}/{total_iterations}")

        for game_idx in range(1, games_per_iter + 1):
            print(f"  Self-play game {game_idx}/{games_per_iter}")
            game_data = play_one_game(
                net,
                rng,
                bias_params,
                max_sims_per_move=max_sims_per_move,
            )
            replay.extend(game_data)

        if len(replay) < batch_size:
            print("  Not enough samples to train yet.")
            continue

        net.train()
        value_mix = 0.7
        ent_beta = 1e-3
        for epoch in range(train_epochs):
            # Recency-weighted sampling: half from most recent quarter
            recent_len = max(len(replay) // 4, 1)
            recent_part = min(batch_size // 2, recent_len)
            if recent_len > 0 and recent_part > 0:
                recent = list(replay)[-recent_len:]
                batch_recent = random.sample(recent, recent_part)
                batch_other = random.sample(list(replay), batch_size - recent_part)
                batch = batch_recent + batch_other
            else:
                batch = random.sample(list(replay), batch_size)

            boards_np = np.stack([b[0] for b in batch], axis=0)
            scalars_np = np.stack([b[1] for b in batch], axis=0)
            pis_np = np.stack([b[2] for b in batch], axis=0)
            zs_np = np.array([b[3] for b in batch], dtype=np.float32)

            # Symmetry augmentation (random H/V flips)
            boards_np, pis_np = _apply_random_flip(boards_np, pis_np, rng)

            # Value shaping toward egg differential
            egg_diff = scalars_np[:, 2] - scalars_np[:, 3]
            zs_np = value_mix * zs_np + (1.0 - value_mix) * egg_diff

            boards = torch.from_numpy(boards_np).to(DEVICE)
            scalars = torch.from_numpy(scalars_np).to(DEVICE)
            target_pis = torch.from_numpy(pis_np).to(DEVICE)
            target_zs = torch.from_numpy(zs_np).to(DEVICE)

            optimizer.zero_grad()
            logits, values = net(boards, scalars)
            log_probs = torch.log_softmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            policy_loss = -(target_pis * log_probs).sum(dim=1).mean()
            value_loss = nn.functional.mse_loss(values, target_zs)
            entropy = -(probs * log_probs).sum(dim=1).mean()
            loss = policy_loss + value_loss - ent_beta * entropy
            loss.backward()
            optimizer.step()

        candidate_state = clone_state_dict(net.state_dict())

        if best_state is None or iteration <= warmup_iters:
            best_state = candidate_state
            torch.save(best_state, save_path)
            if iteration <= warmup_iters:
                print(
                    f"  Warmup iteration {iteration}: "
                    f"accepted weights without evaluation (saved to {save_path})"
                )
            else:
                print(f"  Saved initial weights to {save_path}")
            continue

        print("  Evaluating against current best model...")
        win_rate = evaluate_against_best(
            net,
            best_state,
            bias_params,
            rng,
            eval_games=eval_games,
            max_sims_per_move=max(48, max_sims_per_move // 2),
        )
        print(f"  Win rate versus best: {win_rate:.1%}")
        if win_rate >= eval_threshold:
            best_state = candidate_state
            torch.save(best_state, save_path)
            print(f"  Accepted new weights (stored at {save_path})")
        else:
            print("  Rejected new weights, reverting to previous best.")
            net.load_state_dict(best_state)


def clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def evaluate_against_best(
    candidate: ChickenNet,
    best_state: Dict[str, torch.Tensor],
    bias_params: ActionBiasParams,
    rng: np.random.Generator,
    eval_games: int,
    max_sims_per_move: int,
) -> float:
    reference = ChickenNet(board_size=candidate.board_size).to(DEVICE)
    reference.load_state_dict(best_state)
    reference.eval()
    candidate.eval()

    wins = 0
    with torch.no_grad():
        for game_idx in range(eval_games):
            first, second = (candidate, reference) if game_idx % 2 == 0 else (reference, candidate)
            result = play_head_to_head(
                first,
                second,
                rng,
                bias_params,
                max_sims_per_move=max_sims_per_move,
            )
            if (game_idx % 2 == 0 and result == Result.PLAYER) or (
                game_idx % 2 == 1 and result == Result.ENEMY
            ):
                wins += 1

    candidate.train()
    return wins / eval_games


def play_head_to_head(
    net_a: ChickenNet,
    net_b: ChickenNet,
    rng: np.random.Generator,
    bias_params: ActionBiasParams,
    max_sims_per_move: int,
) -> Result:
    game_map = GameMap()
    trap_manager = TrapdoorManager(game_map)
    board = Board(game_map, 360.0, build_history=False)
    spawns = trap_manager.choose_spawns()
    trap_manager.choose_trapdoors()
    board.chicken_player.start(spawns[0], 0)
    board.chicken_enemy.start(spawns[1], 1)

    beliefs = [TrapdoorBelief(game_map), TrapdoorBelief(game_map)]
    mcts_players = [MCTS(net_a), MCTS(net_b)]
    last_dirs: List[Direction | None] = [None, None]
    second_last_dirs: List[Direction | None] = [None, None]
    since_egg: List[int] = [0, 0]
    player_idx = 0
    last_mover = 0

    while board.get_winner() is None:
        sensory_data = trap_manager.sample_trapdoors(board.chicken_player.get_location())
        current_belief = beliefs[player_idx]
        current_belief.update(board.chicken_player, sensory_data)

        visit_dist = mcts_players[player_idx].run(
            root_board=board,
            root_belief=current_belief,
            max_sims=max_sims_per_move,
        )

        legal_moves = board.get_valid_moves()
        if not legal_moves:
            board.set_winner(Result.ENEMY, WinReason.INVALID_TURN)
            break

        legal_indices = [encode_action(move) for move in legal_moves]
        action_idx = sample_action_with_bias(
            board,
            legal_indices,
            visit_dist,
            last_dirs[player_idx],
            second_last_dirs[player_idx],
            since_egg[player_idx],
            current_belief,
            rng,
            ACTION_SPACE,
            bias_params,
            temperature=0.0,
        )
        action = decode_action(action_idx)
        valid = board.apply_move(action[0], action[1], check_ok=True)
        if not valid:
            board.set_winner(Result.ENEMY, WinReason.INVALID_TURN)
            break

        if action[1] == MoveType.PLAIN:
            second_last_dirs[player_idx] = last_dirs[player_idx]
            last_dirs[player_idx] = action[0]
            since_egg[player_idx] += 1
        elif action[1] == MoveType.EGG:
            second_last_dirs[player_idx] = last_dirs[player_idx]
            last_dirs[player_idx] = None
            since_egg[player_idx] = 0
        else:
            second_last_dirs[player_idx] = last_dirs[player_idx]
            last_dirs[player_idx] = None
            since_egg[player_idx] += 1

        new_loc = board.chicken_player.get_location()
        if trap_manager.is_trapdoor(new_loc):
            board.chicken_player.reset_location()
            board.chicken_enemy.increment_eggs_laid(-1 * board.game_map.TRAPDOOR_PENALTY)
            board.found_trapdoors.add(new_loc)
            for belief in beliefs:
                belief.register_known_trapdoor(new_loc)

        last_mover = player_idx
        if board.get_winner() is not None:
            break

        board.reverse_perspective()
        player_idx = 1 - player_idx

    winner = board.get_winner()
    if winner is None or winner == Result.TIE:
        return Result.TIE
    return winner


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Dontique self-play agent.")
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Ignore existing weights and begin training from scratch.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=36,
        help="Number of outer training iterations to run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        total_iterations=args.iterations,
        games_per_iter=64,
        max_sims_per_move=96,
        batch_size=192,
        lr=6e-4,
        weight_decay=1e-4,
        save_path="s_agent_weights.pt",
        eval_games=24,
        eval_threshold=0.5,
        train_epochs=4,
        warmup_iters=8,
        load_existing=not args.fresh_start,
    )


