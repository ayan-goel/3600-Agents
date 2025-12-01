from __future__ import annotations

import numpy as np
import torch
from typing import Tuple

from game.board import Board
from game.game_map import GameMap
from agents.Fluffy.trapdoor_belief import TrapdoorBelief


def make_state_tensors(board: Board, belief: TrapdoorBelief) -> Tuple[torch.Tensor, torch.Tensor]:
    game_map: GameMap = board.game_map
    size = game_map.MAP_SIZE
    channels = 9
    state = np.zeros((channels, size, size), dtype=np.float32)

    # channel 0 - board mask ones
    state[0, :, :] = 1.0

    px, py = board.chicken_player.get_location()
    ex, ey = board.chicken_enemy.get_location()
    state[1, py, px] = 1.0
    state[2, ey, ex] = 1.0

    for (x, y) in board.eggs_player:
        state[3, y, x] = 1.0
    for (x, y) in board.eggs_enemy:
        state[4, y, x] = 1.0
    for (x, y) in board.turds_player:
        state[5, y, x] = 1.0
    for (x, y) in board.turds_enemy:
        state[6, y, x] = 1.0

    state[7:9, :, :] = belief.as_tensor().astype(np.float32)

    scalar = np.array(
        [
            board.turns_left_player / 40.0,
            board.turns_left_enemy / 40.0,
            len(board.eggs_player) / 40.0,
            len(board.eggs_enemy) / 40.0,
            board.chicken_player.turds_left / 5.0,
            board.chicken_enemy.turds_left / 5.0,
        ],
        dtype=np.float32,
    )
    board_tensor = torch.from_numpy(state).unsqueeze(0)
    scalar_tensor = torch.from_numpy(scalar).unsqueeze(0)
    return board_tensor, scalar_tensor









