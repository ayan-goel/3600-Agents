from __future__ import annotations

import glob
import json
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = PROJECT_ROOT / "engine"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.append(str(ENGINE_ROOT))

from game.enums import Direction, MoveType

# Match JSON schema (observed):
# {
#   "pos": [[ax1, ay1], [bx1, by1], [ax2, ay2], [bx2, by2], ...],
#   "left_behind": ["plain"|"egg"|... per half-turn],
#   "trapdoors": [[x,y], [x,y]],
#   "spawn_a": [x,y],
#   "spawn_b": [x,y],
#   "a_eggs_laid": [...], "b_eggs_laid": [...],
#   "a_moves_left": [...], "b_moves_left": [...],
#   "turn_count": 80, "result": 1 | -1 | 0
# }
#
# We reconstruct approximate state tensors and one-hot actions.

ACTION_SPACE: List[Tuple[Direction, MoveType]] = [
    (d, m) for d in Direction for m in MoveType
]
ACTION_INDEX: Dict[Tuple[int, int], int] = {
    (int(d), int(m)): i for i, (d, m) in enumerate(ACTION_SPACE)
}


def encode_action(action: Tuple[Direction, MoveType]) -> int:
    d, m = action
    return ACTION_INDEX[(int(d), int(m))]


def _diff_to_dir(prev: Tuple[int, int], cur: Tuple[int, int]) -> Direction:
    dx = cur[0] - prev[0]
    dy = cur[1] - prev[1]
    if abs(dx) + abs(dy) != 1:
        # fallback to a direction; treat invalid as UP
        return Direction.UP
    if dx == 1:
        return Direction.RIGHT
    if dx == -1:
        return Direction.LEFT
    if dy == 1:
        return Direction.DOWN
    return Direction.UP


def _empty_state(size: int) -> np.ndarray:
    # channels like Dontique: 0:ones, 1:self, 2:enemy, 3:self eggs, 4:enemy eggs,
    # 5:self turds, 6:enemy turds, 7-8: trap belief (zeros here)
    state = np.zeros((9, size, size), dtype=np.float32)
    state[0, :, :] = 1.0
    return state


def _apply_egg(state: np.ndarray, loc: Tuple[int, int], self_view: bool) -> None:
    x, y = loc
    if self_view:
        state[3, y, x] = 1.0
    else:
        state[4, y, x] = 1.0


def parse_match_json(path: str, board_size: int = 8) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    with open(path, "r") as f:
        data = json.load(f)
    positions: List[List[int]] = data["pos"]
    left_behind: List[str] = data.get("left_behind", [])
    spawn_a = tuple(data["spawn_a"])
    spawn_b = tuple(data["spawn_b"])
    a_eggs = data.get("a_eggs_laid", [])
    b_eggs = data.get("b_eggs_laid", [])
    a_moves_left = data.get("a_moves_left", [])
    b_moves_left = data.get("b_moves_left", [])
    result = data.get("result", 0)

    # Reconstruct approximate board state across half-turns
    jam: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []

    cur_a = spawn_a
    cur_b = spawn_b
    eggs_a: set[Tuple[int, int]] = set()
    eggs_b: set[Tuple[int, int]] = set()

    for half_idx, pos in enumerate(positions):
        mover_is_a = (half_idx % 2 == 0)
        new_pos = tuple(pos)
        prev = cur_a if mover_is_a else cur_b
        d = _diff_to_dir(prev, new_pos)
        lb = left_behind[half_idx] if half_idx < len(left_behind) else "plain"
        move_type = MoveType.EGG if lb == "egg" else MoveType.PLAIN

        # Build state tensor (player-centric)
        state = _empty_state(board_size)
        if mover_is_a:
            # self perspective is A
            state[1, cur_a[1], cur_a[0]] = 1.0
            state[2, cur_b[1], cur_b[0]] = 1.0
            for (x, y) in eggs_a:
                state[3, y, x] = 1.0
            for (x, y) in eggs_b:
                state[4, y, x] = 1.0
            scalars = np.array(
                [
                    (a_moves_left[half_idx // 2] if half_idx // 2 < len(a_moves_left) else 40) / 40.0,
                    (b_moves_left[half_idx // 2] if half_idx // 2 < len(b_moves_left) else 40) / 40.0,
                    (a_eggs[half_idx // 2] if half_idx // 2 < len(a_eggs) else len(eggs_a)) / 40.0,
                    (b_eggs[half_idx // 2] if half_idx // 2 < len(b_eggs) else len(eggs_b)) / 40.0,
                    1.0,  # turds_left unknown; placeholder
                    1.0,
                ],
                dtype=np.float32,
            )
        else:
            # self perspective is B
            state[1, cur_b[1], cur_b[0]] = 1.0
            state[2, cur_a[1], cur_a[0]] = 1.0
            for (x, y) in eggs_b:
                state[3, y, x] = 1.0
            for (x, y) in eggs_a:
                state[4, y, x] = 1.0
            scalars = np.array(
                [
                    (b_moves_left[half_idx // 2] if half_idx // 2 < len(b_moves_left) else 40) / 40.0,
                    (a_moves_left[half_idx // 2] if half_idx // 2 < len(a_moves_left) else 40) / 40.0,
                    (b_eggs[half_idx // 2] if half_idx // 2 < len(b_eggs) else len(eggs_b)) / 40.0,
                    (a_eggs[half_idx // 2] if half_idx // 2 < len(a_eggs) else len(eggs_a)) / 40.0,
                    1.0,
                    1.0,
                ],
                dtype=np.float32,
            )

        # One-hot policy label
        action_idx = encode_action((d, move_type))
        pi = np.zeros((len(ACTION_SPACE),), dtype=np.float32)
        pi[action_idx] = 1.0

        # Value label: map final result to player-centric sign
        z = 0.0
        if result == 1:
            z = 1.0 if mover_is_a else -1.0
        elif result == -1:
            z = -1.0 if mover_is_a else 1.0

        jam.append((state, scalars, pi, z))

        # Update board approximation
        if mover_is_a:
            if move_type == MoveType.EGG:
                eggs_a.add(cur_a)
            cur_a = new_pos
        else:
            if move_type == MoveType.EGG:
                eggs_b.add(cur_b)
            cur_b = new_pos

    return jam


def load_offline_dataset(match_dir: str, board_size: int = 8, limit: int | None = None) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    paths = sorted(glob.glob(os.path.join(match_dir, "*.json")))
    if limit is not None:
        paths = paths[:limit]
    all_rows: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    for p in paths:
        try:
            rows = parse_match_json(p, board_size)
            all_rows.extend(rows)
        except Exception:
            continue
    return all_rows


