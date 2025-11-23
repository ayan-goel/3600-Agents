from __future__ import annotations

from collections import deque
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from game.board import Board
from game.enums import Direction, MoveType, loc_after_direction


class PlayerAgent:
    """
    Lamine: Heuristic boss focused on fast expansion and cutoffs.
    - Deterministic opening (diagonal/perimeter sweep away from enemy)
    - Strict anti-backtrack
    - Strong novelty/frontier bias
    - Simple cutoff pressure
    - Chokepoint TURD gate (region-reduction BFS)
    """

    def __init__(self, board: Board, time_left: Callable[[], float]):
        del time_left
        self.game_map = board.game_map
        self.size = self.game_map.MAP_SIZE

        # Exploration bookkeeping
        self.visit_counts = np.zeros((self.size, self.size), dtype=np.uint8)
        self.visited_tiles = set()
        self.frontier_target: Optional[Tuple[int, int]] = None
        self.frontier_refresh_turn: int = 0
        self.prev_pos = board.chicken_player.get_location()

        # Opening/perimeter parameters
        self.OPENING_TURNS = 10

        # Move scoring weights
        self.novelty_w = 3.0
        self.frontier_w = 6.5
        self.coverage_penalty_w = 1.6
        self.open_local_w = 2.2
        self.backtrack_pen = 20.0
        self.mobility_low_pen = 2.5
        self.risk_turdzone_pen = 3.5
        self.risk_trap_adj_pen = 6.0
        self.risk_trap_cell_pen = 100.0

        # Turd gate thresholds
        self.min_turn_for_turd = 6
        self.turd_region_k = 5

    # -------- Core API --------
    def play(
        self,
        board: Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable[[], float],
    ) -> Tuple[Direction, MoveType]:
        # Record current location
        self._record_visit(board.chicken_player.get_location())
        self._maybe_refresh_frontier(board)

        # Try deterministic opening bias first (still returns a scored move)
        move = self._choose_move(board)
        self.prev_pos = loc_after_direction(board.chicken_player.get_location(), move[0])
        return move

    # -------- Move selection --------
    def _choose_move(self, board: Board) -> Tuple[Direction, MoveType]:
        legal = board.get_valid_moves()
        if not legal:
            return Direction.UP, MoveType.PLAIN
        cur = board.chicken_player.get_location()
        enemy = board.chicken_enemy.get_location()

        # Check if any safe, novel plain moves exist (used to slightly discourage early eggs)
        self._safe_novel_exists = any(
            (mt == MoveType.PLAIN)
            and self._novelty_bonus(loc_after_direction(cur, d)) > 0.0
            and not board.is_cell_in_enemy_turd_zone(loc_after_direction(cur, d))
            for (d, mt) in legal
        )

        scores: List[Tuple[float, Tuple[Direction, MoveType]]] = []
        for (d, mt) in legal:
            nxt = loc_after_direction(cur, d)
            score = 0.0

            # Base directional/expansion scoring
            if mt == MoveType.PLAIN or mt == MoveType.EGG:
                score += self.novelty_w * self._novelty_bonus(nxt)
                score += self.frontier_w * self._frontier_step_bonus(cur, nxt)
                score -= self.coverage_penalty_w * self._coverage_penalty(nxt)
                score += self.open_local_w * self._local_open_space(board, nxt)

                if nxt == self.prev_pos:
                    score -= self.backtrack_pen

                if self._immediate_mobility(board, nxt) <= 1:
                    score -= self.mobility_low_pen

                if board.is_cell_in_enemy_turd_zone(nxt):
                    score -= self.risk_turdzone_pen
                if self._near_known_trapdoor(board, nxt):
                    score -= self.risk_trap_adj_pen
                if self._is_known_trapdoor(board, nxt):
                    score -= self.risk_trap_cell_pen

                # EGG gate: allow eggs when mobility is healthy; discourage in opening if safe novel exists
                if mt == MoveType.EGG:
                    mob = self._immediate_mobility(board, nxt)
                    if mob >= 2:
                        score += 0.8
                    else:
                        score -= 2.0
                    if board.turn_count < self.OPENING_TURNS and self._safe_novel_exists:
                        score -= 1.2

            elif mt == MoveType.TURD:
                # Strict chokepoint gate via region-reduction estimate; otherwise heavily penalize
                if board.turn_count >= self.min_turn_for_turd:
                    delta_opp, delta_self = self._region_reduction_with_turd(board, cur)
                    if delta_opp >= self.turd_region_k and delta_self <= 0:
                        score += 4.0 + 0.5 * (delta_opp - delta_self)
                    else:
                        score -= 50.0  # disable weak turds
                else:
                    score -= 50.0

                # Still apply mild safety checks for the landing square
                if nxt == self.prev_pos:
                    score -= self.backtrack_pen
                if self._immediate_mobility(board, nxt) <= 1:
                    score -= self.mobility_low_pen
                if board.is_cell_in_enemy_turd_zone(nxt):
                    score -= self.risk_turdzone_pen

            scores.append((score, (d, mt)))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1]

    # -------- Heuristic helpers --------
    def _record_visit(self, loc: Tuple[int, int]) -> None:
        x, y = loc
        self.visit_counts[y, x] = min(255, self.visit_counts[y, x] + 1)
        self.visited_tiles.add(loc)

    def _maybe_refresh_frontier(self, board: Board) -> None:
        cur_turn = board.turn_count
        if self.frontier_target is None or cur_turn >= self.frontier_refresh_turn:
            self.frontier_target = self._compute_frontier_target(board)
            self.frontier_refresh_turn = cur_turn + 6
        elif self.frontier_target in self.visited_tiles:
            self.frontier_target = self._compute_frontier_target(board)
            self.frontier_refresh_turn = cur_turn + 6

    def _compute_frontier_target(self, board: Board) -> Optional[Tuple[int, int]]:
        start = board.chicken_player.get_location()
        q = deque([(start, 0)])
        seen = set([start])
        best = None
        best_score = -1e9
        while q:
            loc, dist = q.popleft()
            vx, vy = loc
            # Favor low-visit cells, modest distance, avoid enemy turd zones
            novelty = 1.0 / (1.0 + float(self.visit_counts[vy, vx]))
            risk_pen = 0.5 if board.is_cell_in_enemy_turd_zone(loc) else 0.0
            s = 3.0 * novelty - 0.15 * dist - risk_pen
            if s > best_score:
                best_score = s
                best = loc
            for d in Direction:
                nxt = loc_after_direction(loc, d)
                if nxt in seen:
                    continue
                if not board.is_valid_cell(nxt):
                    continue
                if board.is_cell_blocked(nxt):
                    continue
                seen.add(nxt)
                q.append((nxt, dist + 1))
        return best

    def _frontier_step_bonus(self, cur: Tuple[int, int], nxt: Tuple[int, int]) -> float:
        if self.frontier_target is None:
            return 0.0
        return float(
            self._manhattan(cur, self.frontier_target) - self._manhattan(nxt, self.frontier_target)
        )

    def _novelty_bonus(self, loc: Tuple[int, int]) -> float:
        x, y = loc
        v = int(self.visit_counts[y, x])
        if v == 0:
            return 1.0
        if v == 1:
            return 0.5
        if v == 2:
            return 0.2
        return 0.0

    def _coverage_penalty(self, loc: Tuple[int, int]) -> float:
        x, y = loc
        v = int(self.visit_counts[y, x])
        if v >= 3:
            return 2.0
        if v == 2:
            return 1.0
        return 0.0

    def _local_open_space(self, board: Board, loc: Tuple[int, int]) -> float:
        x0, y0 = loc
        total = 0
        open_cells = 0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) + abs(dy) > 2:
                    continue
                x, y = x0 + dx, y0 + dy
                if not board.is_valid_cell((x, y)):
                    continue
                total += 1
                if not board.is_cell_blocked((x, y)):
                    open_cells += 1
        if total == 0:
            return 0.0
        return open_cells / total

    def _immediate_mobility(self, board: Board, loc: Tuple[int, int]) -> int:
        m = 0
        for d in Direction:
            nxt = loc_after_direction(loc, d)
            if board.is_valid_cell(nxt) and not board.is_cell_blocked(nxt):
                m += 1
        return m

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _near_known_trapdoor(self, board: Board, loc: Tuple[int, int]) -> bool:
        if not hasattr(board, "found_trapdoors"):
            return False
        for t in board.found_trapdoors:
            if self._manhattan(loc, t) <= 1:
                return True
        return False

    def _is_known_trapdoor(self, board: Board, loc: Tuple[int, int]) -> bool:
        if not hasattr(board, "found_trapdoors"):
            return False
        return loc in board.found_trapdoors

    # -------- TURD region-reduction gate (ported and simplified) --------
    def _in_turd_zone(self, loc: Tuple[int, int], turds: Iterable[Tuple[int, int]]) -> bool:
        if loc in turds:
            return True
        for direction in Direction:
            if loc_after_direction(loc, direction) in turds:
                return True
        return False

    def _region_reduction_with_turd(self, board: Board, turd_loc: Tuple[int, int]) -> Tuple[int, int]:
        opp_sites_before = self._reachable_egg_sites_for_opponent(board, plus_turd=None)
        opp_sites_after = self._reachable_egg_sites_for_opponent(board, plus_turd=turd_loc)
        self_sites_before = self._reachable_egg_sites_for_self(board, plus_turd=None)
        self_sites_after = self._reachable_egg_sites_for_self(board, plus_turd=turd_loc)
        return max(0, opp_sites_before - opp_sites_after), max(0, self_sites_before - self_sites_after)

    def _reachable_egg_sites_for_self(self, board: Board, plus_turd: Optional[Tuple[int, int]]) -> int:
        start = board.chicken_player.get_location()
        return self._reachable_egg_sites(
            board=board,
            start=start,
            my_parity=board.chicken_player.even_chicken,
            block_eggs=board.eggs_enemy,
            block_turds=board.turds_enemy,
            my_eggs=board.eggs_player,
            my_turds=board.turds_player,
            plus_turd=plus_turd,
        )

    def _reachable_egg_sites_for_opponent(self, board: Board, plus_turd: Optional[Tuple[int, int]]) -> int:
        start = board.chicken_enemy.get_location()
        return self._reachable_egg_sites(
            board=board,
            start=start,
            my_parity=board.chicken_enemy.even_chicken,
            block_eggs=board.eggs_player,
            block_turds=board.turds_player,
            my_eggs=board.eggs_enemy,
            my_turds=board.turds_enemy,
            plus_turd=plus_turd,
        )

    def _reachable_egg_sites(
        self,
        board: Board,
        start: Tuple[int, int],
        my_parity: int,
        block_eggs: Iterable[Tuple[int, int]],
        block_turds: Iterable[Tuple[int, int]],
        my_eggs: Iterable[Tuple[int, int]],
        my_turds: Iterable[Tuple[int, int]],
        plus_turd: Optional[Tuple[int, int]],
    ) -> int:
        visited = set()
        q = deque([start])
        sites = 0

        def in_turd_zone(loc: Tuple[int, int]) -> bool:
            if loc in block_turds:
                return True
            if plus_turd is not None and (loc == plus_turd):
                return True
            if self._in_turd_zone(loc, block_turds):
                return True
            if plus_turd is not None and self._in_turd_zone(loc, {plus_turd}):
                return True
            return False

        while q:
            loc = q.popleft()
            if loc in visited:
                continue
            if not board.is_valid_cell(loc):
                continue
            if loc in block_eggs:
                continue
            if in_turd_zone(loc):
                continue
            visited.add(loc)
            if (loc[0] + loc[1]) % 2 == my_parity and (loc not in my_eggs) and (loc not in my_turds):
                sites += 1
            for d in Direction:
                nxt = loc_after_direction(loc, d)
                if nxt not in visited:
                    q.append(nxt)
        return sites



