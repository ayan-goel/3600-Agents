from __future__ import annotations

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = PROJECT_ROOT / "engine"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.append(str(ENGINE_ROOT))

from game.board import Board
from game.enums import Direction, MoveType, Result, WinReason
from game.game_map import GameMap
from game.trapdoor_manager import TrapdoorManager

from .agent import ChickenNet, MCTS, decode_action, make_state_tensors
from .trapdoor_belief import TrapdoorBelief

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def play_one_game(
    net: ChickenNet,
    max_sims_per_move: int = 64,
    temperature: float = 1.0,
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

    mcts = MCTS(net)
    net.eval()

    trajectory: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []

    while board.get_winner() is None:
        sensory_data = trap_manager.sample_trapdoors(board.chicken_player.get_location())
        current_belief = beliefs[player_idx]
        current_belief.update(board.chicken_player, sensory_data)

        visit_dist = mcts.run(
            root_board=board,
            root_belief=current_belief,
            max_sims=max_sims_per_move,
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

        action_idx = select_action(visit_dist, temperature)
        action = decode_action(action_idx)
        valid = board.apply_move(action[0], action[1], check_ok=True)
        if not valid:
            # Penalize illegal move by awarding win to opponent.
            board.set_winner(Result.ENEMY, WinReason.INVALID_TURN)
            last_mover = player_idx
            break

        new_loc = board.chicken_player.get_location()
        if trap_manager.is_trapdoor(new_loc):
            board.chicken_player.reset_location()
            board.chicken_enemy.increment_eggs_laid(-1 * board.game_map.TRAPDOOR_PENALTY)
            board.found_trapdoors.add(new_loc)

        last_mover = player_idx
        if board.get_winner() is not None:
            break

        board.reverse_perspective()
        player_idx = 1 - player_idx

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


def select_action(visit_dist: np.ndarray, temperature: float) -> int:
    """Sample an action index from visit counts using temperature."""
    if temperature <= 0:
        return int(np.argmax(visit_dist))

    adjusted = np.power(visit_dist, 1.0 / max(temperature, 1e-6))
    if adjusted.sum() <= 0:
        adjusted = np.ones_like(visit_dist) / len(visit_dist)
    else:
        adjusted /= adjusted.sum()
    choices = np.arange(len(visit_dist))
    return int(np.random.choice(choices, p=adjusted))


def train(
    total_iterations: int = 12,
    games_per_iter: int = 32,
    max_sims_per_move: int = 64,
    batch_size: int = 128,
    lr: float = 7.5e-4,
    weight_decay: float = 1e-4,
    save_path: str = "s_agent_weights.pt",
) -> None:
    """Main training loop."""
    game_map = GameMap()
    size = game_map.MAP_SIZE
    net = ChickenNet(board_size=size).to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    replay: Deque[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = deque(maxlen=50000)

    if not os.path.isabs(save_path):
        save_path = os.path.join(os.path.dirname(__file__), save_path)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for iteration in range(1, total_iterations + 1):
        print(f"Iteration {iteration}/{total_iterations}")

        for game_idx in range(1, games_per_iter + 1):
            print(f"  Self-play game {game_idx}/{games_per_iter}")
            game_data = play_one_game(
                net,
                max_sims_per_move=max_sims_per_move,
                temperature=1.0,
            )
            replay.extend(game_data)

        if len(replay) < batch_size:
            print("  Not enough samples to train yet.")
            continue

        net.train()
        for epoch in range(2):
            batch = random.sample(replay, batch_size)
            boards_np = np.stack([b[0] for b in batch], axis=0)
            scalars_np = np.stack([b[1] for b in batch], axis=0)
            pis_np = np.stack([b[2] for b in batch], axis=0)
            zs_np = np.array([b[3] for b in batch], dtype=np.float32)

            boards = torch.from_numpy(boards_np).to(DEVICE)
            scalars = torch.from_numpy(scalars_np).to(DEVICE)
            target_pis = torch.from_numpy(pis_np).to(DEVICE)
            target_zs = torch.from_numpy(zs_np).to(DEVICE)

            optimizer.zero_grad()
            logits, values = net(boards, scalars)
            log_probs = torch.log_softmax(logits, dim=1)
            policy_loss = -(target_pis * log_probs).sum(dim=1).mean()
            value_loss = nn.functional.mse_loss(values, target_zs)
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()

        torch.save(net.state_dict(), save_path)
        print(f"  Saved weights to {save_path}")


if __name__ == "__main__":
    train(
        total_iterations=12,
        games_per_iter=32,
        max_sims_per_move=64,
        batch_size=128,
        lr=7.5e-4,
        weight_decay=1e-4,
        save_path="s_agent_weights.pt",
    )


