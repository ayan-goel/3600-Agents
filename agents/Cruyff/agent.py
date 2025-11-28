from __future__ import annotations

import time
from typing import Callable, List, Optional, Tuple

import numpy as np

from game.board import Board
from game.enums import Direction, MoveType, Result, loc_after_direction


class SearchTimeout(Exception):
    """Raised when the search runs out of time."""
    pass


class PlayerAgent:
    """
    An agent that uses iterative deepening and alpha-beta pruning.
    The heuristic values (1) difference in eggs laid, (2) number of
    reachable squares, and (3) distance to centre to avoid being trapped.
    """

    OPENING_TURNS = 6
    LATE_GAME_MAP_SIZE = 8

    def __init__(self, board: Board, time_left: Callable[[], float]) -> None:
        self.board = board
        self.time_left = time_left
        self.player_id = board.current_turn
        self.size = board.game_map.MAP_SIZE

    def get_move(self) -> Tuple[MoveType, Direction]:
        start_time = self.time_left()
        best_move: Optional[Tuple[MoveType, Direction]] = None
        depth = 1
        try:
            while True:
                best_move = self._alphabeta_search(depth)
                depth += 1
        except SearchTimeout:
            if best_move is None:
                return MoveType.PLAIN_STEP, Direction.UP
            return best_move

    def _alphabeta_search(self, depth: int) -> Tuple[MoveType, Direction]:
        def max_value(board: Board, depth: int, alpha: float, beta: float) -> float:
            if self.time_left() < 0.05:
                raise SearchTimeout()
            if depth == 0 or board.is_done():
                return self._evaluate(board)
            value = -np.inf
            for move in board.legal_actions():
                next_board = board.next_state(move)
                value = max(value, min_value(next_board, depth - 1, alpha, beta))
                if value >= beta:
                    return value
                alpha = max(alpha, value)
            return value

        def min_value(board: Board, depth: int, alpha: float, beta: float) -> float:
            if self.time_left() < 0.05:
                raise SearchTimeout()
            if depth == 0 or board.is_done():
                return self._evaluate(board)
            value = np.inf
            for move in board.legal_actions():
                next_board = board.next_state(move)
                value = min(value, max_value(next_board, depth - 1, alpha, beta))
                if value <= alpha:
                    return value
                beta = min(beta, value)
            return value

        best_score = -np.inf
        best_move: Tuple[MoveType, Direction] = (MoveType.PLAIN_STEP, Direction.UP)
        for move in self.board.legal_actions():
            if self.time_left() < 0.05:
                raise SearchTimeout()
            next_board = self.board.next_state(move)
            score = min_value(next_board, depth - 1, -np.inf, np.inf)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _evaluate(self, board: Board) -> float:
        result = board.check_game_result()
        if result == Result.A_WIN:
            return float('inf') if self.player_id == 0 else -float('inf')
        if result == Result.B_WIN:
            return -float('inf') if self.player_id == 0 else float('inf')
        if result == Result.DRAW:
            return 0.0
        eggs_self = board.player_egg_count(self.player_id)
        eggs_opponent = board.player_egg_count(1 - self.player_id)
        egg_diff = eggs_self - eggs_opponent

        def reachable_locations(pos: Tuple[int, int], pid: int) -> int:
            visited = set([pos])
            frontier = [pos]
            count = 0
            while frontier:
                loc = frontier.pop()
                count += 1
                for direction in Direction:
                    new_loc = loc_after_direction(loc, direction)
                    if not board.is_in_bounds(new_loc):
                        continue
                    # treat opponent eggs/turds and opponent piece as blocked
                    if board.is_blocked(new_loc, pid):
                        continue
                    if new_loc not in visited:
                        visited.add(new_loc)
                        frontier.append(new_loc)
            return count

        my_pos = board.player_positions()[self.player_id]
        opp_pos = board.player_positions()[1 - self.player_id]
        reach_self = reachable_locations(my_pos, self.player_id)
        reach_opp = reachable_locations(opp_pos, 1 - self.player_id)
        reach_diff = reach_self - reach_opp

        centre = (self.size // 2, self.size // 2)
        dist_to_centre = -abs(my_pos[0] - centre[0]) - abs(my_pos[1] - centre[1])

        return 3.0 * egg_diff + 1.5 * reach_diff + 0.2 * dist_to_centre
