from __future__ import annotations

import os
from typing import Callable, List, Tuple

import numpy as np
import torch

from game.board import Board
from game.enums import Direction, MoveType, loc_after_direction

from .featurizer import make_state_tensors
from .net import ChickenNet
from agents.Fluffy.trapdoor_belief import TrapdoorBelief

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _action_index(mv: Tuple[Direction, MoveType]) -> int:
    d, mt = mv
    return int(d) * 3 + int(mt)


class PlayerAgent:
    """
    Jamal: Hybrid heuristic + neural agent.
    - Uses the CNN policy (trained via self-play + offline logs) to produce priors.
    - Light heuristics re-weighting biases toward open space, fast egging, and safe cut-ins.
    - No heavy MCTS at inference for speed/consistency.
    """

    def __init__(self, board: Board, time_left: Callable[[], float]):
        del time_left
        self.game_map = board.game_map
        self.size = self.game_map.MAP_SIZE
        self.belief = TrapdoorBelief(self.game_map)
        self.net = ChickenNet(board_size=self.size).to(DEVICE)
        self._maybe_load_weights()
        self.net.eval()
        self.rng = np.random.default_rng()

        # Heuristic weights
        self.frontier_w = 0.6
        self.open_space_w = 0.9
        self.center_cutin_w = 0.6
        self.egg_soon_w = 0.8
        self.mobility_low_pen = 1.3
        self.risk_w = 12.0
        self.temperature = 0.0  # greedy at inference

        # Exploration bookkeeping (lightweight)
        self.visit_counts = np.zeros((self.size, self.size), dtype=np.uint8)
        self.prev_pos = board.chicken_player.get_location()

    def _maybe_load_weights(self) -> None:
        # Look for weights in this package; if not found, run with random weights.
        weights_path = os.path.join(os.path.dirname(__file__), "jamal_weights.pt")
        if os.path.exists(weights_path):
            try:
                state = torch.load(weights_path, map_location=DEVICE)
                self.net.load_state_dict(state)
            except Exception:
                pass

    # --------- Heuristic helpers (intentionally simple) ----------
    def _distance_to_center(self, loc: Tuple[int, int]) -> int:
        c = (self.size - 1) // 2
        return abs(loc[0] - c) + abs(loc[1] - c)

    def _local_open_space(self, board: Board, loc: Tuple[int, int]) -> float:
        # count available neighbors within manhattan radius 2
        x0, y0 = loc
        total = 0
        open_cells = 0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) + abs(dy) > 2:
                    continue
                x, y = x0 + dx, y0 + dy
                if not (0 <= x < self.size and 0 <= y < self.size):
                    continue
                total += 1
                if not board.is_cell_blocked((x, y)):
                    open_cells += 1
        if total == 0:
            return 0.0
        return open_cells / total

    def _immediate_mobility(self, board: Board, loc: Tuple[int, int]) -> int:
        count = 0
        for d in Direction:
            nxt = loc_after_direction(loc, d)
            if 0 <= nxt[0] < self.size and 0 <= nxt[1] < self.size:
                if not board.is_cell_blocked(nxt):
                    count += 1
        return count

    def _egg_in_two(self, board: Board, origin: Tuple[int, int]) -> bool:
        if board.can_lay_egg_at_loc(origin):
            return True
        for d in Direction:
            nxt = loc_after_direction(origin, d)
            if 0 <= nxt[0] < self.size and 0 <= nxt[1] < self.size:
                if board.is_cell_blocked(nxt):
                    continue
                if board.can_lay_egg_at_loc(nxt):
                    return True
        return False

    def _risk_at(self, board: Board, loc: Tuple[int, int]) -> float:
        # Simple risk proxy: enemy turd zone and discovered trapdoors are dangerous
        risk = 0.0
        if board.is_cell_in_enemy_turd_zone(loc):
            risk += 1.0
        for t in getattr(board, "found_trapdoors", ()):
            if abs(t[0] - loc[0]) + abs(t[1] - loc[1]) <= 1:
                risk += 1.0
        return risk

    # --------- Core API ----------
    def play(
        self,
        board: Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable[[], float],
    ) -> Tuple[Direction, MoveType]:
        # Update belief
        self._ingest_found_trapdoors(board)
        self.belief.update(board.chicken_player, sensor_data)

        # NN priors
        board_tensor, scalar_tensor = make_state_tensors(board, self.belief)
        with torch.no_grad():
            logits, _ = self.net(board_tensor.to(DEVICE), scalar_tensor.to(DEVICE))
        priors = torch.softmax(logits[0], dim=0).cpu().numpy()

        legal_moves = board.get_valid_moves()
        if not legal_moves:
            return Direction.UP, MoveType.PLAIN
        legal_indices = [_action_index(mv) for mv in legal_moves]

        # Heuristic reweighting
        cur = board.chicken_player.get_location()
        scores: List[Tuple[float, Tuple[Direction, MoveType]]] = []
        for mv, idx in zip(legal_moves, legal_indices):
            d, mt = mv
            nxt = loc_after_direction(cur, d)
            p = float(priors[idx])
            risk = self._risk_at(board, nxt)
            mobility = self._immediate_mobility(board, nxt)
            open_local = self._local_open_space(board, nxt)

            # Heuristic scalar
            h = 0.0
            h += self.open_space_w * open_local
            if mt == MoveType.PLAIN:
                toward_center = self._distance_to_center(cur) - self._distance_to_center(nxt)
                if toward_center > 0 and (board.can_lay_egg_at_loc(nxt) or self._egg_in_two(board, nxt)):
                    h += self.center_cutin_w * float(toward_center)
            if mt == MoveType.EGG:
                h += self.egg_soon_w
            if mobility <= 1:
                h -= self.mobility_low_pen
            h -= self.risk_w * risk

            # Combine NN prior with heuristics (log-linear)
            # Use small floor to avoid zeros
            score = np.log(max(p, 1e-6)) + h
            scores.append((score, mv))

        scores.sort(key=lambda x: x[0], reverse=True)
        choice = scores[0][1]

        # Light exploration bookkeeping
        if choice[1] == MoveType.EGG:
            pass
        self.prev_pos = loc_after_direction(cur, choice[0])
        return choice

    def _ingest_found_trapdoors(self, board: Board) -> None:
        for loc in getattr(board, "found_trapdoors", ()):
            self.belief.register_known_trapdoor(loc)


__all__ = ["PlayerAgent"]
