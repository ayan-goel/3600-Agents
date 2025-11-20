from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from game.board import Board
from game.enums import Direction, MoveType, Result, loc_after_direction

from .trapdoor_belief import TrapdoorBelief


INF = 1_000_000.0


@dataclass
class RegionStats:
    territory: int = 0
    egg_sites: int = 0
    egg_value: float = 0.0
    mobility: float = 0.0
    trap_risk: float = 0.0
    boundary: int = 0
    corner_control: int = 0
    endgame_loop: float = 0.0


class SearchTimeout(Exception):
    """Raised internally when the search budget is exhausted."""


class PlayerAgent:
    """
    Heuristic trapdoor-aware agent nicknamed "Fluffy".
    Implements deterministic alpha-beta search guided by a secure-egg evaluation.
    """

    OPENING_TURNS = 10
    ENDGAME_TURNS = 10

    def __init__(self, board: Board, time_left: Callable[[], float]):
        del time_left  # Not used during construction.
        self.game_map = board.game_map
        self.size = self.game_map.MAP_SIZE
        self.trap_belief = TrapdoorBelief(self.game_map)
        self.my_parity = board.chicken_player.even_chicken
        self.phase = "opening"
        self.search_margin = 0.04
        self.iterative_slack = 0.01
        self.max_depths = {
            "opening": 2,
            "midgame": 3,
            "endgame": 5,
        }
        self.trapdoor_penalty = abs(self.game_map.TRAPDOOR_PENALTY)
        self.risk_block_threshold = 0.85 * (1.0 / self.trapdoor_penalty + 0.2)
        self._cached_risk_map: Optional[np.ndarray] = None
        self.moves_since_last_egg = 0
        # Turd gating parameters
        self.turd_reduction_threshold = 3
        self.turd_self_tolerance = 1
        self.min_turn_for_turd = 8

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
        self._cached_risk_map = self._build_risk_map()

        legal_moves = board.get_valid_moves()
        if board.turn_count < 8:
            non_turd = [mv for mv in legal_moves if mv[1] != MoveType.TURD]
            if non_turd:
                legal_moves = non_turd
        if not legal_moves:
            return Direction.UP, MoveType.PLAIN

        # Immediate egg forcing: choose a legal EGG direction (safest next tile)
        my_eggs = board.chicken_player.get_eggs_laid()
        opp_eggs = board.chicken_enemy.get_eggs_laid()
        deficit = opp_eggs - my_eggs
        endgame = board.turns_left_player <= self.ENDGAME_TURNS
        cooldown = self.moves_since_last_egg >= 4
        egg_moves = [mv for mv in legal_moves if mv[1] == MoveType.EGG]
        if egg_moves and (endgame or cooldown or deficit > 0):
            return self._select_safest_egg(board, egg_moves)
        if egg_moves:
            # Opportunistic egg if the safest next tile is low risk
            next_risks = []
            cur_loc = board.chicken_player.get_location()
            for mv in egg_moves:
                nxt = loc_after_direction(cur_loc, mv[0])
                next_risks.append(self._risk_at(nxt))
            if next_risks and min(next_risks) <= 0.6:
                return self._select_safest_egg(board, egg_moves)

        # Greedy move selection: score each legal move and pick best
        try:
            move = self._choose_greedy(board, legal_moves)
        except Exception:
            move = legal_moves[0]

        if move[1] == MoveType.EGG:
            self.moves_since_last_egg = 0
        else:
            self.moves_since_last_egg += 1
        return move

    # ------------------------------------------------------------------ #
    # Search orchestration
    # ------------------------------------------------------------------ #
    def _choose_with_search(
        self,
        board: Board,
        legal_moves: Sequence[Tuple[Direction, MoveType]],
        time_left: Callable[[], float],
    ) -> Tuple[Direction, MoveType]:
        max_depth = self.max_depths[self.phase]
        now = time.perf_counter()
        budget = max(time_left() - self.search_margin, 0.02)
        deadline = now + budget

        best_move = legal_moves[0]
        best_score = -INF

        for depth in range(1, max_depth + 1):
            try:
                score, move = self._alpha_beta(
                    board,
                    depth=depth,
                    alpha=-INF,
                    beta=INF,
                    maximizing=True,
                    deadline=deadline,
                )
                if move is not None:
                    best_move = move
                    best_score = score
            except SearchTimeout:
                break

            if time.perf_counter() + self.iterative_slack >= deadline:
                break

        # In extremely flat positions prefer eggs during endgame.
        if (
            best_score < 1.0
            and self.phase == "endgame"
            and board.can_lay_egg()
            and any(mt == MoveType.EGG for _, mt in legal_moves)
        ):
            for move in legal_moves:
                if move[1] == MoveType.EGG:
                    best_move = move
                    break
        endgame_force = board.turns_left_player <= self.ENDGAME_TURNS
        cooldown_force = self.moves_since_last_egg >= 5
        if (endgame_force or cooldown_force) and board.can_lay_egg():
            egg_moves = [mv for mv in legal_moves if mv[1] == MoveType.EGG]
            if egg_moves:
                best_move = self._select_safest_egg(board, egg_moves)
        return best_move

    def _choose_greedy(
        self,
        board: Board,
        legal_moves: Sequence[Tuple[Direction, MoveType]],
    ) -> Tuple[Direction, MoveType]:
        # Strongly penalize turds; allow only if we already have ≥4 eggs and are in combat/chokepoint
        my_eggs = board.chicken_player.get_eggs_laid()
        scores: List[Tuple[float, Tuple[Direction, MoveType]]] = []
        egg_sites = self._candidate_egg_sites(board)

        for move in legal_moves:
            dir_, mt = move
            cur_loc = board.chicken_player.get_location()
            next_loc = loc_after_direction(cur_loc, dir_)
            next_risk = self._risk_at(next_loc)
            # Risk shaping: tighten when ahead, loosen when behind
            my_eggs = board.chicken_player.get_eggs_laid()
            opp_eggs = board.chicken_enemy.get_eggs_laid()
            lead = my_eggs - opp_eggs
            deficit = -lead if lead < 0 else 0
            risk_w = 8.0 if lead >= 3 else (5.0 if lead >= 1 else (4.0 if deficit >= 2 else 6.0))

            if mt == MoveType.EGG:
                risk_here = self._risk_at(cur_loc)
                base = 100.0
                if self._is_corner(cur_loc):
                    base += 5.0
                base -= 8.0 * risk_here
                base += 4.0 if self.moves_since_last_egg >= 3 else 0.0
                scores.append((base, move))
                continue

            if mt == MoveType.TURD:
                delta_opp, delta_self = self._region_reduction_with_turd(board, cur_loc)
                chokepoint = self._chokepoint_bonus(board, cur_loc)
                allow = (
                    board.turn_count >= self.min_turn_for_turd
                    and delta_opp >= self.turd_reduction_threshold
                    and delta_self <= self.turd_self_tolerance
                )
                if not allow:
                    base = -40.0 - risk_w * next_risk
                else:
                    base = 10.0 * float(delta_opp) - 7.0 * float(delta_self) + 1.0 * chokepoint
                    base -= risk_w * next_risk
                scores.append((base, move))
                continue

            # PLAIN move: parity alignment, distance to next egg site, and risk
            # Prefer landing on our parity and reducing distance to a viable egg tile
            parity_bonus = 6.0 if self._is_my_parity(next_loc) else 0.0
            dist = self._min_distance_to_any(next_loc, egg_sites)
            # If no egg sites detected (rare), fall back to 0
            dist_pen = 0.0 if dist < 0 else 3.0 * float(dist)
            corner_pull = 1.5 if self._is_corner(next_loc) else 0.0
            cooldown_pull = 2.0 if self.moves_since_last_egg >= 4 else 0.0
            # Endgame 2-step egg scheduler: prefer paths that enable egg in 1–2
            one_away = 1.0 if board.can_lay_egg_at_loc(next_loc) else 0.0
            two_away = 0.0
            if not one_away:
                for d2 in Direction:
                    nxt2 = loc_after_direction(next_loc, d2)
                    if board.is_valid_cell(nxt2) and board.can_lay_egg_at_loc(nxt2):
                        two_away = 1.0
                        break
            # Turd adjacency penalty to reduce self-blocking
            adj_pen = 1.0 if self._in_turd_zone(next_loc, board.turds_player) else 0.0
            base = 15.0 + parity_bonus + corner_pull + cooldown_pull
            base -= dist_pen
            base += 8.0 * one_away + 3.0 * two_away
            base -= risk_w * next_risk
            base -= 1.5 * adj_pen
            scores.append((base, move))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1]

    def _alpha_beta(
        self,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        deadline: float,
    ) -> Tuple[float, Optional[Tuple[Direction, MoveType]]]:
        if time.perf_counter() >= deadline:
            raise SearchTimeout

        winner = board.get_winner()
        if winner is not None:
            if winner == Result.PLAYER:
                return INF, None
            if winner == Result.ENEMY:
                return -INF, None
            return 0.0, None

        if depth == 0:
            return self._evaluate(board), None

        legal_moves = board.get_valid_moves()
        if board.turn_count < 8:
            filtered = [mv for mv in legal_moves if mv[1] != MoveType.TURD]
            if filtered:
                legal_moves = filtered
        if not legal_moves:
            return self._evaluate(board), None

        ordered_moves = self._order_moves(board, legal_moves)

        best_move: Optional[Tuple[Direction, MoveType]] = None

        if maximizing:
            value = -INF
            for move in ordered_moves:
                child = board.get_copy()
                if not child.apply_move(*move):
                    continue
                child.reverse_perspective()
                score, _ = self._alpha_beta(
                    child,
                    depth - 1,
                    alpha,
                    beta,
                    False,
                    deadline,
                )
                if score > value:
                    value = score
                    best_move = move
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value, best_move

        value = INF
        for move in ordered_moves:
            child = board.get_copy()
            if not child.apply_move(*move):
                continue
            child.reverse_perspective()
            score, _ = self._alpha_beta(
                child,
                depth - 1,
                alpha,
                beta,
                True,
                deadline,
            )
            if score < value:
                value = score
                best_move = move
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value, best_move

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    def _evaluate(self, board: Board) -> float:
        if board.turns_left_player <= 0 and board.turns_left_enemy <= 0:
            eggs_self = board.chicken_player.get_eggs_laid()
            eggs_opp = board.chicken_enemy.get_eggs_laid()
            if eggs_self > eggs_opp:
                return INF
            if eggs_self < eggs_opp:
                return -INF
            return 0.0

        if not board.has_moves_left():
            return -0.9 * INF
        if not board.has_moves_left(enemy=True):
            return 0.9 * INF

        stats_self = self._analyze_region(board, for_enemy=False)
        stats_opp = self._analyze_region(board, for_enemy=True)

        egg_diff = (
            board.chicken_player.get_eggs_laid()
            - board.chicken_enemy.get_eggs_laid()
        )

        score = 0.0
        score += 12.0 * egg_diff
        score += 6.0 * (stats_self.egg_value - stats_opp.egg_value)
        score += 2.5 * (stats_self.territory - stats_opp.territory)
        score += 1.25 * (stats_self.mobility - stats_opp.mobility)
        score += 1.2 * (stats_self.boundary - stats_opp.boundary)
        score += 3.0 * (stats_self.corner_control - stats_opp.corner_control)
        score += -4.0 * (stats_self.trap_risk - stats_opp.trap_risk)
        score += 0.75 * (
            board.chicken_player.get_turds_left()
            - board.chicken_enemy.get_turds_left()
        )
        if self.phase == "endgame":
            score += 7.0 * (stats_self.endgame_loop - stats_opp.endgame_loop)
        if board.turn_count < 10:
            own_turds_used = self.game_map.MAX_TURDS - board.chicken_player.get_turds_left()
            opp_turds_used = self.game_map.MAX_TURDS - board.chicken_enemy.get_turds_left()
            score -= 1.8 * own_turds_used
            score += 1.8 * opp_turds_used
        if board.can_lay_egg():
            score += 1.5
        elif self._is_my_parity(board.chicken_player.get_location()):
            score += 0.4

        return score

    def _analyze_region(self, board: Board, for_enemy: bool) -> RegionStats:
        stats = RegionStats()
        if self._cached_risk_map is None:
            self._cached_risk_map = self._build_risk_map()
        risk_map = self._cached_risk_map

        player = board.chicken_enemy if for_enemy else board.chicken_player
        opponent = board.chicken_player if for_enemy else board.chicken_enemy
        start = player.get_location()
        parity = player.even_chicken

        my_eggs = board.eggs_enemy if for_enemy else board.eggs_player
        opp_eggs = board.eggs_player if for_enemy else board.eggs_enemy
        opp_turds = board.turds_player if for_enemy else board.turds_enemy

        visited = set()
        queue = deque([start])

        def is_blocked(loc: Tuple[int, int]) -> bool:
            if not board.is_valid_cell(loc):
                return True
            if loc == opponent.get_location():
                return True
            if loc in opp_eggs:
                return True
            if self._in_turd_zone(loc, opp_turds):
                return True
            return False

        while queue:
            loc = queue.popleft()
            if loc in visited or is_blocked(loc):
                stats.boundary += 1 if loc in opp_eggs else 0
                continue
            visited.add(loc)
            stats.territory += 1
            stats.trap_risk += risk_map[loc[1], loc[0]]
            if self._is_corner(loc):
                stats.corner_control += 1

            if self._can_lay_egg_at(loc, parity, my_eggs):
                stats.egg_sites += 1
                base = 1.0 + (0.5 if self._is_corner(loc) else 0.0)
                risk_penalty = max(0.1, 1.0 - 2.2 * risk_map[loc[1], loc[0]])
                stats.egg_value += base * risk_penalty

            for direction in Direction:
                nxt = loc_after_direction(loc, direction)
                if not board.is_valid_cell(nxt):
                    continue
                if nxt in visited:
                    continue
                if (
                    not for_enemy
                    and risk_map[nxt[1], nxt[0]] > self.risk_block_threshold
                    and stats.territory > 4
                ):
                    continue
                queue.append(nxt)

        stats.mobility = self._compute_mobility(board, for_enemy)

        if stats.territory > 0:
            stats.trap_risk /= stats.territory
        stats.endgame_loop = min(
            stats.egg_sites,
            board.turns_left_enemy if for_enemy else board.turns_left_player,
        )
        return stats

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _compute_mobility(self, board: Board, for_enemy: bool) -> float:
        moves = board.get_valid_moves(enemy=for_enemy)
        value = 0.0
        for _, move_type in moves:
            if move_type == MoveType.PLAIN:
                value += 1.0
            elif move_type == MoveType.EGG:
                value += 0.8
            else:
                value += 0.6
        return value

    def _order_moves(
        self,
        board: Board,
        moves: Sequence[Tuple[Direction, MoveType]],
    ) -> List[Tuple[Direction, MoveType]]:
        scored = []
        allow_early_turds = board.turn_count >= 10 or len(moves) == 1
        for move in moves:
            dir_, move_type = move
            loc = board.chicken_player.get_location()
            next_loc = loc_after_direction(loc, dir_)
            risk = self._cached_risk_map[next_loc[1], next_loc[0]] if self._cached_risk_map is not None and board.is_valid_cell(next_loc) else 0.0

            base = 0.0
            if move_type == MoveType.EGG:
                base = 7.0
                if self._is_corner(loc):
                    base += 1.0
                base += 1.2 * self._parity_alignment_bonus(loc)
                base += 2.5 if self.moves_since_last_egg >= 4 else 0.0
                base -= 1.0 * risk
            elif move_type == MoveType.TURD:
                if not allow_early_turds or not self._turd_allowed(board, loc):
                    base = -20.0 - risk
                else:
                    base = 3.0 + self._turd_pressure(board, loc) + self._chokepoint_bonus(board, loc) - risk
            else:
                base = 2.5 - 1.0 * risk
                if self._is_my_parity(next_loc):
                    base += 1.2
                base += self._parity_alignment_bonus(next_loc)
                if not board.can_lay_egg() and self._is_my_parity(next_loc):
                    base += 0.8

            scored.append((base, move))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [move for _, move in scored]

    def _turd_pressure(self, board: Board, loc: Tuple[int, int]) -> float:
        enemy_loc = board.chicken_enemy.get_location()
        distance = abs(enemy_loc[0] - loc[0]) + abs(enemy_loc[1] - loc[1])
        return max(0.0, 4.0 - distance)

    def _turd_allowed(self, board: Board, loc: Tuple[int, int]) -> bool:
        eggs = board.chicken_player.get_eggs_laid()
        enemy_loc = board.chicken_enemy.get_location()
        distance = abs(enemy_loc[0] - loc[0]) + abs(enemy_loc[1] - loc[1])
        if eggs >= 4:
            return True
        if distance <= 2:
            return True
        return self._chokepoint_bonus(board, loc) >= 2.0

    def _chokepoint_bonus(self, board: Board, loc: Tuple[int, int]) -> float:
        open_neighbors = 0
        for direction in Direction:
            nxt = loc_after_direction(loc, direction)
            if board.is_valid_cell(nxt) and nxt not in board.turds_enemy and nxt not in board.turds_player:
                open_neighbors += 1
        if open_neighbors <= 1:
            return 3.0
        if open_neighbors == 2:
            return 2.0
        return 0.5

    def _can_lay_egg_at(
        self,
        loc: Tuple[int, int],
        parity: int,
        my_eggs: Iterable[Tuple[int, int]],
    ) -> bool:
        if (loc[0] + loc[1]) % 2 != parity:
            return False
        if loc in my_eggs:
            return False
        return True

    def _in_turd_zone(self, loc: Tuple[int, int], turds: Iterable[Tuple[int, int]]) -> bool:
        if loc in turds:
            return True
        for direction in Direction:
            if loc_after_direction(loc, direction) in turds:
                return True
        return False

    def _is_corner(self, loc: Tuple[int, int]) -> bool:
        return loc in {
            (0, 0),
            (0, self.size - 1),
            (self.size - 1, 0),
            (self.size - 1, self.size - 1),
        }

    def _build_risk_map(self) -> np.ndarray:
        grid = np.zeros((self.size, self.size), dtype=np.float64)
        for y in range(self.size):
            for x in range(self.size):
                prob = self.trap_belief.trapdoor_prob_at((x, y))
                grid[y, x] = prob * self.trapdoor_penalty
        return grid

    def _risk_at(self, loc: Tuple[int, int]) -> float:
        if self._cached_risk_map is None:
            self._cached_risk_map = self._build_risk_map()
        x, y = loc
        if 0 <= x < self.size and 0 <= y < self.size:
            return float(self._cached_risk_map[y, x])
        return 1.0

    def _update_phase(self, board: Board) -> None:
        turns_played = board.turn_count
        if board.turns_left_player <= self.ENDGAME_TURNS:
            self.phase = "endgame"
        elif turns_played >= self.OPENING_TURNS:
            self.phase = "midgame"
        else:
            self.phase = "opening"

    def _register_known_trapdoors(self, board: Board) -> None:
        for trap in getattr(board, "found_trapdoors", ()):
            self.trap_belief.register_known_trapdoor(trap)

    def _select_safest_egg(
        self, board: Board, egg_moves: Sequence[Tuple[Direction, MoveType]]
    ) -> Tuple[Direction, MoveType]:
        best = egg_moves[0]
        best_risk = float("inf")
        for move in egg_moves:
            cur = board.chicken_player.get_location()
            nxt = loc_after_direction(cur, move[0])
            risk = self._risk_at(nxt)
            if risk < best_risk:
                best_risk = risk
                best = move
        return best

    def _is_my_parity(self, loc: Tuple[int, int]) -> bool:
        return (loc[0] + loc[1]) % 2 == self.my_parity

    def _parity_alignment_bonus(self, loc: Tuple[int, int]) -> float:
        return 1.0 if self._is_my_parity(loc) else 0.0

    def _candidate_egg_sites(self, board: Board) -> List[Tuple[int, int]]:
        sites: List[Tuple[int, int]] = []
        for y in range(self.size):
            for x in range(self.size):
                loc = (x, y)
                if board.can_lay_egg_at_loc(loc):
                    sites.append(loc)
        return sites

    def _min_distance_to_any(self, start: Tuple[int, int], targets: Sequence[Tuple[int, int]]) -> int:
        if not targets:
            return -1
        best = 10**9
        sx, sy = start
        for (tx, ty) in targets:
            d = abs(tx - sx) + abs(ty - sy)
            if d < best:
                best = d
        return best

    def _enemy_distance(self, board: Board, loc: Tuple[int, int]) -> int:
        ex, ey = board.chicken_enemy.get_location()
        return abs(ex - loc[0]) + abs(ey - loc[1])

    # ---- Region reduction gating for TURD moves ----
    def _region_reduction_with_turd(self, board: Board, turd_loc: Tuple[int, int]) -> Tuple[int, int]:
        """Return (delta_opp_sites, delta_self_sites) if we hypothetically place a turd at turd_loc."""
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
            block_eggs=board.eggs_player,   # our eggs block them
            block_turds=board.turds_player, # our turds (and hypothetical) block them
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
            # Adjacent to any blocking turd
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
            # Egg site if parity matches and square not occupied by our own eggs/turds
            if (loc[0] + loc[1]) % 2 == my_parity and (loc not in my_eggs) and (loc not in my_turds):
                sites += 1
            for d in Direction:
                nxt = loc_after_direction(loc, d)
                if nxt not in visited:
                    q.append(nxt)
        return sites

