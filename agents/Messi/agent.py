from __future__ import annotations

import time
from collections import deque
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from game.board import Board
from game.enums import Direction, MoveType, Result, loc_after_direction

from .trapdoor_belief import TrapdoorBelief

INF = 1_000_000.0


class SearchTimeout(Exception):
    """Raised when the lookahead budget runs out."""


class PlayerAgent:
    """Heuristic agent tuned to exploit Fluffy's deterministic play."""

    OPENING_TURNS = 6
    LATE_GAME_TURNS = 8

    def __init__(self, board: Board, time_left: Callable[[], float]):
        del time_left
        self.game_map = board.game_map
        self.size = self.game_map.MAP_SIZE
        self.trap_belief = TrapdoorBelief(self.game_map)
        self.my_parity = board.chicken_player.even_chicken
        self.enemy_parity = board.chicken_enemy.even_chicken
        self.spawn = board.chicken_player.get_spawn()
        self.enemy_spawn = board.chicken_enemy.get_spawn()
        self.moves_since_last_egg = 0
        self.phase = "opening"
        self._risk_grid = np.zeros((self.size, self.size), dtype=np.float32)
        self._search_deadline = 0.0
        self.safety_buffer = 0.08
        self.min_budget = 0.03
        self._lane_dir = Direction.RIGHT if self.spawn[0] <= self.size // 2 else Direction.LEFT
        self._vertical_bias = (
            Direction.UP if self.spawn[1] > (self.size // 2) else Direction.DOWN
        )
        self._opening_script = self._build_opening_script()
        self.loop_targets = self._build_loop_targets()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def play(
        self,
        board: Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable[[], float],
    ) -> Tuple[Direction, MoveType]:
        self._register_known_trapdoors(board)
        self.trap_belief.update(board.chicken_player, sensor_data)
        self._update_phase(board)
        self._risk_grid = self._build_risk_grid(board)

        legal_moves = board.get_valid_moves()
        if not legal_moves:
            return Direction.UP, MoveType.PLAIN

        opening_move = self._opening_move(board, legal_moves)
        if opening_move is not None:
            choice = opening_move
        else:
            try:
                choice = self._select_move(board, legal_moves, time_left)
            except SearchTimeout:
                choice = self._fast_greedy(board, legal_moves)

        if choice[1] == MoveType.EGG:
            self.moves_since_last_egg = 0
        else:
            self.moves_since_last_egg += 1
        return choice

    # ------------------------------------------------------------------ #
    # Search and move selection
    # ------------------------------------------------------------------ #
    def _select_move(
        self,
        board: Board,
        legal_moves: Sequence[Tuple[Direction, MoveType]],
        time_left: Callable[[], float],
    ) -> Tuple[Direction, MoveType]:
        depth = self._adaptive_depth(board, legal_moves)
        budget = max(time_left() - self.safety_buffer, self.min_budget)
        self._search_deadline = time.perf_counter() + max(budget, self.min_budget)
        ordered = self._order_moves(board, legal_moves)
        best_move = ordered[0]
        best_score = -INF
        limit = min(len(ordered), self._branch_limit(depth))
        for move in ordered[:limit]:
            score = self._rollout_after_move(board, move, depth)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _adaptive_depth(
        self, board: Board, legal_moves: Sequence[Tuple[Direction, MoveType]]
    ) -> int:
        depth = 3
        if board.turn_count < 3:
            depth = 2
        if self.phase == "endgame":
            depth += 1
        if len(legal_moves) <= 4:
            depth += 1
        return min(depth, 4)

    def _branch_limit(self, depth: int) -> int:
        if depth >= 4:
            return 5
        if depth == 3:
            return 6
        return 8

    def _order_moves(
        self, board: Board, legal_moves: Sequence[Tuple[Direction, MoveType]]
    ) -> List[Tuple[Direction, MoveType]]:
        scored = []
        for move in legal_moves:
            scored.append((self._score_move(board, move), move))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mv for _, mv in scored]

    def _fast_greedy(
        self, board: Board, legal_moves: Sequence[Tuple[Direction, MoveType]]
    ) -> Tuple[Direction, MoveType]:
        ordered = self._order_moves(board, legal_moves)
        return ordered[0]

    def _rollout_after_move(
        self,
        board: Board,
        move: Tuple[Direction, MoveType],
        horizon: int,
    ) -> float:
        if time.perf_counter() >= self._search_deadline:
            raise SearchTimeout

        child = board.get_copy()
        if not child.apply_move(*move):
            return -INF

        winner = child.get_winner()
        if winner == Result.PLAYER:
            return INF
        if winner == Result.ENEMY:
            return -INF

        base_score = self._static_eval(child)
        if horizon <= 1:
            return base_score

        reply = child.get_copy()
        reply.reverse_perspective()
        enemy_move = self._enemy_policy(reply, horizon)
        if enemy_move is None:
            return base_score
        if not reply.apply_move(*enemy_move):
            reply.reverse_perspective()
            return base_score - 10.0
        reply.reverse_perspective()

        winner = reply.get_winner()
        if winner == Result.PLAYER:
            return INF * 0.5
        if winner == Result.ENEMY:
            return -INF * 0.5

        if horizon <= 2:
            return 0.65 * self._static_eval(reply) + 0.35 * base_score

        next_moves = reply.get_valid_moves()
        if not next_moves:
            return self._static_eval(reply)

        ordered = self._order_moves(reply, next_moves)
        limit = min(len(ordered), self._branch_limit(horizon - 1))
        best = -INF
        for follow in ordered[:limit]:
            val = self._rollout_after_move(reply, follow, horizon - 1)
            if val > best:
                best = val
        return 0.5 * best + 0.5 * base_score

    def _enemy_policy(
        self, board: Board, horizon: int
    ) -> Optional[Tuple[Direction, MoveType]]:
        legal = board.get_valid_moves()
        if not legal:
            return None
        scored = []
        for move in legal:
            scored.append((self._score_enemy_move(board, move), move))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    # ------------------------------------------------------------------ #
    # Move scoring heuristics
    # ------------------------------------------------------------------ #
    def _score_move(self, board: Board, move: Tuple[Direction, MoveType]) -> float:
        dir_, mt = move
        cur = board.chicken_player.get_location()
        next_loc = loc_after_direction(cur, dir_)
        risk_here = self._risk_at(cur)
        risk_next = self._risk_at(next_loc)
        eggs_self = board.chicken_player.get_eggs_laid()
        eggs_opp = board.chicken_enemy.get_eggs_laid()
        egg_diff = eggs_self - eggs_opp
        dist_center = self._distance_to_center(next_loc)

        if mt == MoveType.EGG:
            base = 120.0
            if self.moves_since_last_egg >= 2:
                base += 15.0
            elif self.moves_since_last_egg >= 1:
                base += 6.0
            chain = self._egg_chain_strength(board, next_loc, depth=3)
            base += 10.0 * chain
            base += self._corner_bonus(cur)
            base -= 60.0 * risk_here
            base -= 25.0 * max(0.0, risk_next - 0.5)
            base -= 3.0 * dist_center
            base += 6.0 * max(0, -egg_diff)
            return base

        if mt == MoveType.TURD:
            block = self._turd_block_value(board, cur)
            base = 12.0 * block
            if egg_diff >= 2:
                base -= 12.0
            else:
                base += 4.0
            base -= 18.0 * risk_here
            base -= 8.0 * risk_next
            return base

        lane_progress = self._lane_progress(cur, next_loc)
        future_egg = self._future_egg_turns(board, next_loc)
        choke = self._enemy_choke_bonus(board, next_loc)
        path_pull = self._path_progress(board, next_loc)
        parity_bonus = 12.0 if self._is_my_parity(next_loc) else 0.0

        base = 32.0 + parity_bonus
        base += 5.0 * lane_progress
        base += 7.0 * choke
        base += path_pull
        base -= 18.0 * risk_next
        base -= 2.5 * dist_center
        base -= 4.0 * future_egg
        return base

    def _score_enemy_move(self, board: Board, move: Tuple[Direction, MoveType]) -> float:
        dir_, mt = move
        cur = board.chicken_player.get_location()
        next_loc = loc_after_direction(cur, dir_)
        risk = self._risk_at(next_loc)

        if mt == MoveType.EGG:
            chain = self._egg_chain_strength(board, next_loc, depth=2)
            score = 110.0 + 8.0 * chain - 40.0 * risk
            if self._is_corner(cur):
                score += 12.0
            return score
        if mt == MoveType.TURD:
            return -35.0 - 15.0 * risk

        target = self._enemy_lane_target(board)
        dist = self._manhattan(next_loc, target)
        return 28.0 - 2.8 * dist - 22.0 * risk

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    def _static_eval(self, board: Board) -> float:
        eggs_self = board.chicken_player.get_eggs_laid()
        eggs_opp = board.chicken_enemy.get_eggs_laid()
        egg_diff = eggs_self - eggs_opp

        mobility_self = len(board.get_valid_moves())
        mobility_opp = len(board.get_valid_moves(enemy=True))
        territory_self = self._available_sites(board, friendly=True)
        territory_opp = self._available_sites(board, friendly=False)
        risk_here = self._risk_at(board.chicken_player.get_location())
        choke = self._enemy_choke_bonus(board, board.chicken_player.get_location())

        score = 150.0 * egg_diff
        score += 6.0 * (territory_self - territory_opp)
        score += 3.0 * (mobility_self - mobility_opp)
        score -= 45.0 * risk_here
        score += 12.0 * choke
        score -= 0.5 * board.turns_left_player
        return score

    # ------------------------------------------------------------------ #
    # Opening guidance
    # ------------------------------------------------------------------ #
    def _opening_move(
        self, board: Board, legal_moves: Sequence[Tuple[Direction, MoveType]]
    ) -> Optional[Tuple[Direction, MoveType]]:
        if self.phase != "opening":
            return None
        if board.turn_count >= len(self._opening_script):
            return None
        desired_dir = self._opening_script[board.turn_count]
        candidates = [mv for mv in legal_moves if mv[0] == desired_dir]
        if not candidates:
            return None
        if board.can_lay_egg():
            egg_moves = [mv for mv in candidates if mv[1] == MoveType.EGG]
            if egg_moves and self._risk_at(board.chicken_player.get_location()) < 0.9:
                return egg_moves[0]
        plain_moves = [mv for mv in candidates if mv[1] == MoveType.PLAIN]
        if plain_moves:
            cur = board.chicken_player.get_location()
            plain_moves.sort(
                key=lambda mv: self._risk_at(loc_after_direction(cur, mv[0]))
            )
            return plain_moves[0]
        return None

    def _build_opening_script(self) -> List[Direction]:
        forward = Direction.RIGHT if self.spawn[0] == 0 else Direction.LEFT
        script = [forward, forward]
        script.append(self._vertical_bias)
        script.append(forward)
        script.append(self._vertical_bias)
        script.append(forward)
        return script

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _register_known_trapdoors(self, board: Board) -> None:
        for loc in getattr(board, "found_trapdoors", set()):
            self.trap_belief.register_known_trapdoor(loc)

    def _update_phase(self, board: Board) -> None:
        if board.turn_count < self.OPENING_TURNS:
            self.phase = "opening"
        elif board.turns_left_player <= self.LATE_GAME_TURNS:
            self.phase = "endgame"
        else:
            self.phase = "midgame"

    def _build_loop_targets(self) -> List[Tuple[int, int]]:
        center = (self.size - 1) / 2.0
        scored: List[Tuple[Tuple[int, int], float]] = []
        for x in range(self.size):
            for y in range(self.size):
                if (x + y) % 2 != self.my_parity:
                    continue
                dist_spawn = abs(x - self.spawn[0]) + abs(y - self.spawn[1])
                dist_center = abs(x - center) + abs(y - center)
                priority = 0.6 * dist_center + 0.4 * dist_spawn
                scored.append(((x, y), priority))
        scored.sort(key=lambda item: (item[1], item[0][1], item[0][0]))
        return [loc for loc, _ in scored]

    def _build_risk_grid(self, board: Board) -> np.ndarray:
        grid = np.zeros((self.size, self.size), dtype=np.float32)
        trap_scale = abs(self.game_map.TRAPDOOR_PENALTY) + 1.5
        for y in range(self.size):
            for x in range(self.size):
                grid[y, x] = trap_scale * self.trap_belief.trapdoor_prob_at((x, y))
        for (tx, ty) in getattr(board, "found_trapdoors", set()):
            grid[ty, tx] = trap_scale * 4.0
        for (tx, ty) in board.turds_enemy:
            grid[ty, tx] += 3.5
            for dir_ in Direction:
                nx, ny = loc_after_direction((tx, ty), dir_)
                if self._in_bounds((nx, ny)):
                    grid[ny, nx] += 1.25
        for (tx, ty) in board.turds_player:
            grid[ty, tx] += 0.4
        return grid

    def _risk_at(self, loc: Tuple[int, int]) -> float:
        x, y = loc
        if 0 <= x < self.size and 0 <= y < self.size:
            return float(self._risk_grid[y, x])
        return 10.0

    def _distance_to_center(self, loc: Tuple[int, int]) -> float:
        center = (self.size - 1) / 2.0
        return abs(loc[0] - center) + abs(loc[1] - center)

    def _lane_progress(self, cur: Tuple[int, int], nxt: Tuple[int, int]) -> float:
        if self._lane_dir in (Direction.RIGHT, Direction.LEFT):
            forward = 1 if self._lane_dir == Direction.RIGHT else -1
            return forward * (nxt[0] - cur[0])
        forward = 1 if self._lane_dir == Direction.DOWN else -1
        return forward * (nxt[1] - cur[1])

    def _future_egg_turns(self, board: Board, loc: Tuple[int, int]) -> float:
        if board.can_lay_egg_at_loc(loc):
            return 1.0
        return 2.0

    def _path_progress(self, board: Board, loc: Tuple[int, int]) -> float:
        best = 0.0
        for idx, target in enumerate(self.loop_targets[:14]):
            if target in board.eggs_player:
                continue
            dist = self._manhattan(loc, target)
            if dist < 0:
                continue
            weight = max(0.0, 10.0 - 2.0 * dist - 0.3 * idx)
            best = max(best, weight)
        return best

    def _enemy_choke_bonus(self, board: Board, loc: Tuple[int, int]) -> float:
        enemy_loc = board.chicken_enemy.get_location()
        dist = self._manhattan(loc, enemy_loc)
        if dist <= 2:
            return 4.0 - dist
        lane = self._lane_distance(loc)
        return max(0.0, 3.0 - 0.3 * lane)

    def _turd_block_value(self, board: Board, loc: Tuple[int, int]) -> float:
        enemy_loc = board.chicken_enemy.get_location()
        dist = self._manhattan(loc, enemy_loc)
        choke = 0.0
        if dist <= 2:
            choke += 3.0 - dist
        lanes = 0.0
        for dir_ in Direction:
            nxt = loc_after_direction(loc, dir_)
            if not self._in_bounds(nxt):
                continue
            if nxt in board.turds_player or nxt in board.turds_enemy:
                continue
            lanes += 0.3
        return choke + lanes

    def _lane_distance(self, loc: Tuple[int, int]) -> int:
        target_x = self.size - 1 if self.spawn[0] == 0 else 0
        return abs(loc[0] - target_x)

    def _enemy_lane_target(self, board: Board) -> Tuple[int, int]:
        return self.spawn

    def _corner_bonus(self, loc: Tuple[int, int]) -> float:
        x, y = loc
        if (x in (0, self.size - 1)) and (y in (0, self.size - 1)):
            return 18.0
        return 0.0

    def _egg_chain_strength(
        self, board: Board, origin: Tuple[int, int], depth: int = 3
    ) -> float:
        visited = set([origin])
        q = deque([(origin, 0)])
        score = 0.0
        while q:
            loc, steps = q.popleft()
            if steps > depth:
                continue
            if steps % 2 == 0:
                if (
                    board.can_lay_egg_at_loc(loc)
                    and loc not in board.eggs_player
                    and loc not in board.turds_player
                    and loc not in board.turds_enemy
                ):
                    score += max(0.0, depth - steps + 1)
            for dir_ in Direction:
                nxt = loc_after_direction(loc, dir_)
                if not self._in_bounds(nxt):
                    continue
                if (
                    nxt in visited
                    or nxt in board.eggs_player
                    or nxt in board.turds_player
                    or nxt in board.turds_enemy
                ):
                    continue
                visited.add(nxt)
                q.append((nxt, steps + 1))
        return score * 0.25

    def _available_sites(self, board: Board, friendly: bool) -> int:
        parity = self.my_parity if friendly else self.enemy_parity
        eggs = board.eggs_player if friendly else board.eggs_enemy
        my_turds = board.turds_player if friendly else board.turds_enemy
        opp_turds = board.turds_enemy if friendly else board.turds_player
        count = 0
        for x in range(self.size):
            for y in range(self.size):
                if (x + y) % 2 != parity:
                    continue
                loc = (x, y)
                if loc in eggs or loc in my_turds or loc in opp_turds:
                    continue
                count += 1
        return count

    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _in_bounds(self, loc: Tuple[int, int]) -> bool:
        x, y = loc
        return 0 <= x < self.size and 0 <= y < self.size

    def _is_corner(self, loc: Tuple[int, int]) -> bool:
        x, y = loc
        return (x in (0, self.size - 1)) and (y in (0, self.size - 1))

    def _is_my_parity(self, loc: Tuple[int, int]) -> bool:
        return (loc[0] + loc[1]) % 2 == self.my_parity

