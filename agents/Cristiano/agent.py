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
        self.min_turn_for_turd = 6
        self.rng = np.random.default_rng()
        # Opening aggression
        self.opening_aggressive_turns = 8
        self.min_open_eggs = 2
        self.outward_weight = 1.2
        self.spawn_loc = board.chicken_player.get_spawn()

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

        # Opening sweep: aggressively expand along a cardinal direction, egging every other step
        sweep_move = self._opening_sweep_move(board, legal_moves)
        if sweep_move is not None:
            if sweep_move[1] == MoveType.EGG:
                self.moves_since_last_egg = 0
            else:
                self.moves_since_last_egg += 1
            return sweep_move

        # Immediate egg forcing: choose a legal EGG direction (safest next tile)
        my_eggs = board.chicken_player.get_eggs_laid()
        opp_eggs = board.chicken_enemy.get_eggs_laid()
        deficit = opp_eggs - my_eggs
        endgame = board.turns_left_player <= self.ENDGAME_TURNS
        cooldown = self.moves_since_last_egg >= 3
        egg_moves = [mv for mv in legal_moves if mv[1] == MoveType.EGG]
        opening = self.phase == "opening"
        if egg_moves:
            if opening and my_eggs >= self.min_open_eggs and not endgame and deficit <= 0 and not cooldown:
                # In opening, after a couple of eggs, only take eggs that keep the chain going or corners
                best = None
                best_chain = -1.0
                cur = board.chicken_player.get_location()
                for mv in egg_moves:
                    nxt = loc_after_direction(cur, mv[0])
                    chain = self._egg_chain_score(board, nxt, max_depth=2)
                    if self._is_corner(cur):
                        chain += 0.5
                    if chain > best_chain:
                        best_chain = chain
                        best = mv
                if best is not None and best_chain >= 0.7:
                    return best
            else:
                if endgame or cooldown or deficit > 0:
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
        # Precompute opponent region before move for cutoff deltas
        opp_region_before = self._opponent_region_size_current(board)

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
            # Trapdoor no-go in midgame: avoid stepping next to discovered trapdoors unless endgame
            if self.phase == "midgame" and getattr(board, "found_trapdoors", None):
                for tloc in board.found_trapdoors:
                    if abs(tloc[0] - next_loc[0]) + abs(tloc[1] - next_loc[1]) <= 1:
                        next_risk += 0.25

            if mt == MoveType.EGG:
                risk_here = self._risk_at(cur_loc)
                base = 100.0
                # Prefer corner if it does not harm follow-up egg chain
                chain_after = self._egg_chain_score(board, next_loc, max_depth=2)
                base += 6.0 * chain_after
                if self._is_corner(cur_loc):
                    base += 5.0
                base -= 8.0 * risk_here
                # Cutoff bonus: how much this egg (at current loc) + step reduces opp region
                cut_delta = self._opponent_region_cutoff_after_move(board, move, opp_region_before)
                if self.phase in ("opening", "midgame"):
                    base += (12.0 if self.phase == "opening" else 8.0) * float(cut_delta)
                base += 4.0 if self.moves_since_last_egg >= 3 else 0.0
                scores.append((base, move))
                continue

            if mt == MoveType.TURD:
                delta_opp, delta_self = self._region_reduction_with_turd(board, cur_loc)
                chokepoint = self._chokepoint_bonus(board, cur_loc)
                # Dynamic threshold based on lead/deficit
                dyn_thresh = self.turd_reduction_threshold + (1 if lead >= 2 else 0) - (1 if deficit >= 2 else 0)
                allow = (
                    board.turn_count >= self.min_turn_for_turd
                    and delta_opp >= max(1, dyn_thresh)
                    and delta_self <= self.turd_self_tolerance
                )
                # Defend cluster: if enemy close to our dense egg cluster, allow with modest reduction
                if not allow:
                    enemy_close = self._enemy_distance(board, cur_loc) <= 3
                    my_cluster = self._count_nearby(board.eggs_player, cur_loc, radius=2)
                    if enemy_close and my_cluster >= 3 and delta_opp >= 2:
                        allow = True
                
                # Path blocking: allow turd if it severs enemy path to their largest cluster
                if not allow and self._blocks_enemy_best_path(board, move):
                    allow = True
                
                # Cluster breaking: allow turd if adjacent to enemy's largest cluster center
                if not allow:
                    clusters = self._find_egg_clusters(board, for_enemy=True)
                    if clusters:
                        largest = max(clusters, key=lambda c: len(c))
                        center = self._cluster_center(largest)
                        if abs(cur_loc[0] - center[0]) + abs(cur_loc[1] - center[1]) <= 1:
                            allow = True

                if not allow:
                    base = -40.0 - risk_w * next_risk
                else:
                    base = 10.0 * float(delta_opp) - 7.0 * float(delta_self) + 1.0 * chokepoint
                    if self._blocks_enemy_best_path(board, move):
                        base += 8.0
                    # Bonus for turd near enemy cluster center
                    clusters = self._find_egg_clusters(board, for_enemy=True)
                    if clusters:
                        largest = max(clusters, key=lambda c: len(c))
                        center = self._cluster_center(largest)
                        if abs(cur_loc[0] - center[0]) + abs(cur_loc[1] - center[1]) <= 1:
                            base += 6.0
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
            chain3 = 0.0
            if not one_away:
                two_away = 1.0 if self._egg_in_k_steps(board, next_loc, k=2) else 0.0
                if not two_away:
                    chain3 = 1.0 if self._egg_in_k_steps(board, next_loc, k=3) else 0.0
            # Turd adjacency penalty to reduce self-blocking
            adj_pen = 1.0 if self._in_turd_zone(next_loc, board.turds_player) else 0.0
            # Approximate mobility after moving
            mobility_next = self._approx_mobility(board, next_loc)
            mobility_pen = 2.0 if mobility_next <= 1 else (1.0 if mobility_next == 2 else 0.0)
            base = 15.0 + parity_bonus + corner_pull + cooldown_pull
            base -= dist_pen
            base += 8.0 * one_away + 4.0 * two_away + 2.0 * chain3
            # Opening outward expansion bonus
            if self.phase == "opening":
                sx, sy = self.spawn_loc
                cur_out = abs(cur_loc[0] - sx) + abs(cur_loc[1] - sy)
                nxt_out = abs(next_loc[0] - sx) + abs(next_loc[1] - sy)
                base += self.outward_weight * (nxt_out - cur_out)
                # Center avoidance in opening
                cx, cy = self.size // 2, self.size // 2
                dist_center = abs(next_loc[0] - cx) + abs(next_loc[1] - cy)
                if dist_center <= 2:
                    base -= 2.0
            # Cutoff and Voronoi advantage after this plain move
            cut_delta = self._opponent_region_cutoff_after_move(board, move, opp_region_before)
            if self.phase in ("opening", "midgame"):
                base += (12.0 if self.phase == "opening" else 9.0) * float(cut_delta)
                vor_adv = self._voronoi_advantage_after_move(board, move)
                base += 2.5 * float(vor_adv)
                
            # Path blocking bonus
            if self.phase != "endgame" and self._blocks_enemy_best_path(board, move):
                base += 8.0

            # Anti-camping penalty for enemy
            if self._enemy_camping(board):
                # Heavily reward cutting off camping enemy
                if cut_delta >= 1:
                    base += 5.0

            base -= risk_w * next_risk
            base -= 1.5 * adj_pen
            base -= mobility_pen
            # Tiny jitter to avoid tie oscillations
            base += float(self.rng.uniform(-0.05, 0.05))
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

    # ---- Opening sweep expansion ----
    def _opening_sweep_move(
        self,
        board: Board,
        legal_moves: Sequence[Tuple[Direction, MoveType]],
    ) -> Optional[Tuple[Direction, MoveType]]:
        # Only during the opening aggression window
        if board.turn_count >= self.opening_aggressive_turns:
            return None
        # Ensure we still respect extreme risk
        cur = board.chicken_player.get_location()
        dirs = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

        # Pre-index legal moves by (dir, type)
        legal = {(d, mt) for d, mt in legal_moves}

        # Evaluate runway in each direction
        best_dir = None
        best_score = -1e9
        runway_info = {}
        for d in dirs:
            steps, eggables, avg_risk = self._runway_stats(board, cur, d, max_steps=6)
            # Prefer many eggable tiles and longer runway, penalize risk
            score = 3.0 * eggables + 1.2 * steps - 2.5 * avg_risk
            # Small tie-break by how far from spawn this points
            nxt = loc_after_direction(cur, d)
            score += 0.2 * (abs(nxt[0] - self.spawn_loc[0]) + abs(nxt[1] - self.spawn_loc[1]))
            runway_info[d] = (steps, eggables, avg_risk, score)
            if score > best_score:
                best_score = score
                best_dir = d

        # Consider a secondary dir if primary is too risky right now
        sorted_dirs = sorted(dirs, key=lambda dd: runway_info[dd][3], reverse=True)
        risk_cap = 0.8

        # Try up to two directions
        for candidate_dir in sorted_dirs[:2]:
            steps, eggables, avg_risk, _ = runway_info[candidate_dir]
            if steps <= 0:
                continue
            next_loc = loc_after_direction(cur, candidate_dir)
            next_risk = self._risk_at(next_loc)
            if next_risk > risk_cap:
                continue

            # If we can lay an egg now and moving along candidate_dir is legal as EGG, do it
            if board.can_lay_egg() and ((candidate_dir, MoveType.EGG) in legal):
                return (candidate_dir, MoveType.EGG)

            # Otherwise, move plain along the candidate_dir if legal, to set up next egg
            if (candidate_dir, MoveType.PLAIN) in legal:
                return (candidate_dir, MoveType.PLAIN)

        return None

    def _runway_stats(
        self,
        board: Board,
        start: Tuple[int, int],
        direction: Direction,
        max_steps: int = 6,
    ) -> Tuple[int, int, float]:
        steps = 0
        eggables = 0
        risks: List[float] = []
        loc = start
        parity = board.chicken_player.even_chicken

        for _ in range(max_steps):
            nxt = loc_after_direction(loc, direction)
            if not board.is_valid_cell(nxt):
                break
            if board.is_cell_blocked(nxt):
                break
            # We consider hitting the enemy a blocker; do not simulate stepping onto them
            if nxt == board.chicken_enemy.get_location():
                break
            steps += 1
            risks.append(self._risk_at(nxt))
            # Eggable tile is the CURRENT tile when we choose EGG, but we are projecting:
            # On the turn we move to nxt, on the following turn (if parity matches) we can egg at nxt.
            if (nxt[0] + nxt[1]) % 2 == parity:
                eggables += 1
            loc = nxt

        avg_risk = float(np.mean(risks)) if risks else 0.0
        return steps, eggables, avg_risk

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

    # ---- Opponent cutoff and Voronoi helpers ----
    def _opponent_region_size_current(self, board: Board) -> int:
        """Compute opponent reachable region size given current obstacles (our eggs/turds) and our current location blocking."""
        enemy_start = board.chicken_enemy.get_location()
        return self._opponent_region_size(board, enemy_start, our_loc=board.chicken_player.get_location(), our_eggs=set(board.eggs_player), our_turds=set(board.turds_player))

    def _opponent_region_cutoff_after_move(self, board: Board, move: Tuple[Direction, MoveType], opp_region_before: int) -> int:
        """Compute how much opponent region shrinks after applying our move (without mutating the board)."""
        dir_, mt = move
        cur = board.chicken_player.get_location()
        next_loc = loc_after_direction(cur, dir_)
        our_eggs = set(board.eggs_player)
        our_turds = set(board.turds_player)
        # EGG and TURD lay on current tile before moving
        if mt == MoveType.EGG:
            our_eggs.add(cur)
        elif mt == MoveType.TURD:
            # Respect turd placement rule: if illegal it won't be in legal_moves anyway
            our_turds.add(cur)
        enemy_start = board.chicken_enemy.get_location()
        opp_after = self._opponent_region_size(board, enemy_start, our_loc=next_loc, our_eggs=our_eggs, our_turds=our_turds)
        return max(0, opp_region_before - opp_after)

    def _opponent_region_size(
        self,
        board: Board,
        enemy_start: Tuple[int, int],
        our_loc: Tuple[int, int],
        our_eggs: Iterable[Tuple[int, int]],
        our_turds: Iterable[Tuple[int, int]],
    ) -> int:
        visited = set()
        q = deque([enemy_start])
        our_turds_set = set(our_turds)
        our_eggs_set = set(our_eggs)

        def blocked(loc: Tuple[int, int]) -> bool:
            if not board.is_valid_cell(loc):
                return True
            if loc == our_loc:
                return True
            if loc in our_eggs_set:
                return True
            if self._in_turd_zone(loc, our_turds_set):
                return True
            return False

        size = 0
        while q:
            loc = q.popleft()
            if loc in visited or blocked(loc):
                continue
            visited.add(loc)
            size += 1
            for d in Direction:
                nxt = loc_after_direction(loc, d)
                if nxt not in visited:
                    q.append(nxt)
        return size

    def _voronoi_advantage_after_move(self, board: Board, move: Tuple[Direction, MoveType]) -> int:
        """Approximate number of cells closer to us than to the enemy after the move (ignoring risk)."""
        dir_, mt = move
        cur = board.chicken_player.get_location()
        next_loc = loc_after_direction(cur, dir_)
        enemy_loc = board.chicken_enemy.get_location()
        # Simulate egg/turd at current loc for obstacles
        our_eggs = set(board.eggs_player)
        our_turds = set(board.turds_player)
        if mt == MoveType.EGG:
            our_eggs.add(cur)
        elif mt == MoveType.TURD:
            our_turds.add(cur)

        # Dist maps BFS constrained by obstacles (our eggs/turds) and player/enemy positions as passable for themselves
        d_self = self._distance_map(board, start=next_loc, block_eggs=board.eggs_enemy, block_turds=board.turds_enemy)
        d_enemy = self._distance_map(board, start=enemy_loc, block_eggs=our_eggs, block_turds=our_turds, block_opponent_loc=next_loc)

        adv = 0
        for y in range(self.size):
            for x in range(self.size):
                # Only count safe tiles for advantage
                if self._risk_at((x, y)) > 0.7:
                    continue
                ds = d_self[y][x]
                de = d_enemy[y][x]
                if ds >= 0 and (de < 0 or ds < de):
                    adv += 1
        return adv

    def _distance_map(
        self,
        board: Board,
        start: Tuple[int, int],
        block_eggs: Iterable[Tuple[int, int]],
        block_turds: Iterable[Tuple[int, int]],
        block_opponent_loc: Optional[Tuple[int, int]] = None,
    ) -> List[List[int]]:
        dist = [[-1 for _ in range(self.size)] for _ in range(self.size)]
        q = deque()
        if board.is_valid_cell(start) and not board.is_cell_blocked(start):
            dist[start[1]][start[0]] = 0
            q.append(start)
        blocks_eggs = set(block_eggs)
        blocks_turds = set(block_turds)
        while q:
            loc = q.popleft()
            for d in Direction:
                nxt = loc_after_direction(loc, d)
                if not board.is_valid_cell(nxt):
                    continue
                if dist[nxt[1]][nxt[0]] != -1:
                    continue
                if block_opponent_loc is not None and nxt == block_opponent_loc:
                    continue
                if nxt in blocks_eggs:
                    continue
                if self._in_turd_zone(nxt, blocks_turds):
                    continue
                if board.is_cell_blocked(nxt):
                    continue
                dist[nxt[1]][nxt[0]] = dist[loc[1]][loc[0]] + 1
                q.append(nxt)
        return dist

    # ---- Path blocking logic ----
    def _blocks_enemy_best_path(self, board: Board, move: Tuple[Direction, MoveType]) -> bool:
        """Returns True if the move (especially TURD) blocks the enemy's shortest path to their largest reachable egg cluster."""
        enemy_loc = board.chicken_enemy.get_location()
        
        # Find enemy's best target cluster
        opp_clusters = self._find_egg_clusters(board, for_enemy=True)
        if not opp_clusters:
            return False
        
        # Sort clusters by size, take largest
        target_cluster = max(opp_clusters, key=lambda c: len(c))
        target_center = self._cluster_center(target_cluster)
        
        # Compute path before our move
        path_before = self._shortest_path(board, enemy_loc, target_center, block_eggs=board.eggs_player, block_turds=board.turds_player)
        if not path_before:
            return False
            
        # Compute path after our move (simulate obstacle)
        dir_, mt = move
        cur = board.chicken_player.get_location()
        next_loc = loc_after_direction(cur, dir_) # For plain moves, we become the obstacle at next_loc
        
        our_eggs = set(board.eggs_player)
        our_turds = set(board.turds_player)
        obstacle_loc = None

        if mt == MoveType.TURD:
            our_turds.add(cur)
            obstacle_loc = cur
        elif mt == MoveType.EGG:
            our_eggs.add(cur)
            obstacle_loc = cur # Egg blocks at current
        else:
            obstacle_loc = next_loc # Body block at next

        path_after = self._shortest_path(
            board, 
            enemy_loc, 
            target_center, 
            block_eggs=our_eggs, 
            block_turds=our_turds, 
            extra_block=obstacle_loc if mt == MoveType.PLAIN else None
        )

        # Blocked if path length increases significantly or becomes impossible
        if not path_after:
            return True
        return len(path_after) > len(path_before) + 2

    def _find_egg_clusters(self, board: Board, for_enemy: bool) -> List[List[Tuple[int, int]]]:
        """Find clusters of empty eggable spots for the given player."""
        parity = board.chicken_enemy.even_chicken if for_enemy else board.chicken_player.even_chicken
        my_eggs = board.eggs_enemy if for_enemy else board.eggs_player
        
        spots = []
        for y in range(self.size):
            for x in range(self.size):
                loc = (x, y)
                if (x + y) % 2 == parity and loc not in my_eggs and board.is_valid_cell(loc) and not board.is_cell_blocked(loc):
                    spots.append(loc)
        
        clusters = []
        visited = set()
        for s in spots:
            if s in visited:
                continue
            # BFS to find cluster
            cluster = []
            q = deque([s])
            visited.add(s)
            while q:
                curr = q.popleft()
                cluster.append(curr)
                for d in Direction:
                    nxt = loc_after_direction(curr, d)
                    if nxt in spots and nxt not in visited:
                        visited.add(nxt)
                        q.append(nxt)
            if len(cluster) >= 3:
                clusters.append(cluster)
        return clusters

    def _cluster_center(self, cluster: List[Tuple[int, int]]) -> Tuple[int, int]:
        if not cluster:
            return (0, 0)
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        return (int(np.mean(xs)), int(np.mean(ys)))

    def _shortest_path(
        self, 
        board: Board, 
        start: Tuple[int, int], 
        end: Tuple[int, int], 
        block_eggs: Iterable[Tuple[int, int]], 
        block_turds: Iterable[Tuple[int, int]],
        extra_block: Optional[Tuple[int, int]] = None
    ) -> Optional[List[Tuple[int, int]]]:
        q = deque([(start, [start])])
        visited = {start}
        blocks_eggs = set(block_eggs)
        blocks_turds = set(block_turds)
        
        while q:
            curr, path = q.popleft()
            if curr == end:
                return path
            
            # Heuristic sort: prefer neighbors closer to end
            neighbors = []
            for d in Direction:
                nxt = loc_after_direction(curr, d)
                if not board.is_valid_cell(nxt): continue
                if nxt in visited: continue
                if nxt in blocks_eggs: continue
                if self._in_turd_zone(nxt, blocks_turds): continue
                if extra_block and nxt == extra_block: continue
                if board.is_cell_blocked(nxt): continue
                
                dist = abs(nxt[0] - end[0]) + abs(nxt[1] - end[1])
                neighbors.append((dist, nxt))
            
            neighbors.sort(key=lambda x: x[0])
            
            for _, nxt in neighbors:
                visited.add(nxt)
                q.append((nxt, path + [nxt]))
                
        return None

    # ---- Egg chain helpers ----
    def _egg_in_k_steps(self, board: Board, loc: Tuple[int, int], k: int) -> bool:
        if k <= 0:
            return board.can_lay_egg_at_loc(loc)
        if board.can_lay_egg_at_loc(loc):
            return True
        for d in Direction:
            nxt = loc_after_direction(loc, d)
            if not board.is_valid_cell(nxt):
                continue
            if board.is_cell_blocked(nxt):
                continue
            if self._egg_in_k_steps(board, nxt, k - 1):
                return True
        return False

    def _egg_chain_score(self, board: Board, next_loc: Tuple[int, int], max_depth: int = 3) -> float:
        score = 0.0
        if board.can_lay_egg_at_loc(next_loc):
            return 1.0
        if self._egg_in_k_steps(board, next_loc, k=2):
            score += 0.7
        elif max_depth >= 3 and self._egg_in_k_steps(board, next_loc, k=3):
            score += 0.4
        return score

    def _approx_mobility(self, board: Board, loc: Tuple[int, int]) -> int:
        # Count neighbors that are not blocked for a crude mobility estimate
        m = 0
        for d in Direction:
            nxt = loc_after_direction(loc, d)
            if board.is_valid_cell(nxt) and not board.is_cell_blocked(nxt):
                m += 1
        return m

    def _count_nearby(self, items: Iterable[Tuple[int, int]], center: Tuple[int, int], radius: int) -> int:
        cx, cy = center
        c = 0
        for (x, y) in items:
            if abs(x - cx) + abs(y - cy) <= radius:
                c += 1
        return c

    def _enemy_camping(self, board: Board) -> bool:
        """Return True if enemy has made >4 moves but stayed within a small radius of recent average loc."""
        if board.turn_count < 10:
            return False
        # Crude heuristic: check history if available, else check if enemy is boxed in corner/edge
        # Since we don't have easy history access here without parsing, we use current mobility and corner proximity
        enemy_loc = board.chicken_enemy.get_location()
        # If enemy near corner and high egg density there, they are farming
        if self._is_corner(enemy_loc) or (min(enemy_loc[0], self.size - 1 - enemy_loc[0]) <= 1 and min(enemy_loc[1], self.size - 1 - enemy_loc[1]) <= 1):
             # Check egg density
             nearby_eggs = self._count_nearby(board.eggs_enemy, enemy_loc, 2)
             if nearby_eggs >= 3:
                 return True
        return False

