from __future__ import annotations

import time
from collections import deque
from typing import Callable, List, Optional, Sequence, Tuple, Set

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
    CORNER_SPRINT_TURNS = 10

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
        self.enemy_history: deque[Tuple[int, int]] = deque(maxlen=8)
        self.enemy_heading: Tuple[float, float] = (0.0, 0.0)
        self.opening_egg_goal = 3
        self.corner_target: Tuple[int, int] = self._best_diagonal_corner(board)
        # Exploration state
        self.visit_counts = np.zeros((self.size, self.size), dtype=np.uint8)
        self.visited_tiles: Set[Tuple[int, int]] = {self.spawn}
        self.frontier_target: Optional[Tuple[int, int]] = None
        self.frontier_refresh_turn: int = -1
        # Previous board position (to penalize immediate backtracks)
        self.prev_pos: Tuple[int, int] = self.spawn
        # Loop prevention state
        self.recent_positions: deque[Tuple[int, int]] = deque(maxlen=12)
        self.stagnation_count: int = 0
        # Outward expansion aggressiveness
        self.outward_weight: float = 1.25
        # Cached flag set each turn in _order_moves
        self._safe_novel_exists: bool = False

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
        # Exploration bookkeeping
        cur_loc = board.chicken_player.get_location()
        self._record_visit(cur_loc)
        self._maybe_refresh_frontier(board)
        self.corner_target = self._best_diagonal_corner(board)
        enemy_loc = board.chicken_enemy.get_location()
        self._record_enemy(enemy_loc)

        legal_moves = board.get_valid_moves()
        if not legal_moves:
            return Direction.UP, MoveType.PLAIN

        opening_move = self._opening_move(board, legal_moves)
        if opening_move is not None:
            choice = opening_move
        else:
            sprint_move = self._corner_sprint_move(board, legal_moves)
            if sprint_move is not None:
                choice = sprint_move
            else:
                try:
                    choice = self._select_move(board, legal_moves, time_left)
                except SearchTimeout:
                    choice = self._fast_greedy(board, legal_moves)

        if choice[1] == MoveType.EGG:
            self.moves_since_last_egg = 0
        else:
            self.moves_since_last_egg += 1
        # Update loop/stagnation trackers
        # Record current location in recent path if changed
        if not self.recent_positions or self.recent_positions[-1] != cur_loc:
            self.recent_positions.append(cur_loc)
        # Predict next location and update stagnation pressure
        nxt_loc = loc_after_direction(cur_loc, choice[0])
        novelty_next = nxt_loc not in self.visited_tiles
        try:
            terr_cur = self._territory_diff_bfs(board, cur_loc, radius=5)
            terr_nxt = self._territory_diff_bfs(board, nxt_loc, radius=5)
            terr_delta = terr_nxt - terr_cur
        except Exception:
            terr_delta = 0
        if novelty_next or terr_delta > 0:
            self.stagnation_count = max(0, self.stagnation_count - 1)
        else:
            self.stagnation_count = min(8, self.stagnation_count + 1)
        # Update previous position (for anti-backtracking next turn)
        self.prev_pos = cur_loc
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
        # Keep depth >=3 even on very early turns to better foresee cutoffs
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
        # Compute if there exists a safe novel plain move this turn
        self._safe_novel_exists = False
        cur = board.chicken_player.get_location()
        for dir_, mt in legal_moves:
            if mt != MoveType.PLAIN:
                continue
            nxt = loc_after_direction(cur, dir_)
            if not board.is_valid_cell(nxt):
                continue
            # Prefer truly novel, reasonably safe tiles
            if self._risk_at(nxt) <= 0.85 and int(self.visit_counts[nxt[1], nxt[0]]) == 0:
                self._safe_novel_exists = True
                break

        scored: List[Tuple[float, Tuple[Direction, MoveType]]] = []
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
        enemy_forecast = self._predict_enemy_next_loc(board)
        enemy_forecast_adjacent = (
            enemy_forecast is not None
            and (abs(next_loc[0] - enemy_forecast[0]) + abs(next_loc[1] - enemy_forecast[1]) == 1)
        )
        enemy_route_target = self._enemy_route_target(board)
        # Compute current local territory once per call (for mode selection and deltas)
        territory_cur_now = self._territory_diff_bfs(board, cur, radius=5)
        mode = self._select_mode(board, territory_cur_now)
        eggs_self = board.chicken_player.get_eggs_laid()
        eggs_opp = board.chicken_enemy.get_eggs_laid()
        egg_diff = eggs_self - eggs_opp
        dist_center = self._distance_to_center(next_loc)
        # No special blotch repulsion here; handled via risk grid
        stuck_pressure = min(1.0, float(self.stagnation_count) / 5.0)
        stagnating = self._stagnation_flag()

        if mt == MoveType.EGG:
            base = 115.0
            if self.moves_since_last_egg >= 2:
                base += 15.0
            elif self.moves_since_last_egg >= 1:
                base += 6.0
            chain = self._egg_chain_strength(board, next_loc, depth=3)
            base += 12.0 * chain
            base += self._corner_bonus(cur)
            base -= 60.0 * risk_here
            base -= 25.0 * max(0.0, risk_next - 0.5)
            base -= 3.0 * dist_center
            base += 6.0 * max(0, -egg_diff)
            # Favor parity-correct eggs and novelty of current tile
            if self._is_my_parity(cur):
                base += 6.0
            base += 2.5 * self._novelty_bonus(cur)
            # Opening throttling: prioritize coverage over early egg spam
            if self.phase == "opening":
                force_opening = self._should_force_opening_egg(board, cur, next_loc)
                base -= 24.0
                if self.moves_since_last_egg < 2:
                    base -= (18.0 - 6.0 * self.moves_since_last_egg)
                if self._safe_novel_exists:
                    base -= 8.0
                if force_opening:
                    base += 28.0
            else:
                base += 10.0
            # Mode adjustments: simplify priorities
            if mode == "expand":
                base -= 20.0
            elif mode == "egg":
                base += 12.0
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
        # Parity benefit damped when stagnating to allow freer movement out of ruts
        if stagnating:
            parity_bonus = 0.0
        else:
            parity_weight = 12.0 * (1.0 - 0.3 * stuck_pressure)
            parity_bonus = parity_weight if self._is_my_parity(next_loc) else 0.0
        # Outward expansion from spawn
        sx, sy = self.spawn
        cur_out = abs(cur[0] - sx) + abs(cur[1] - sy)
        nxt_out = abs(next_loc[0] - sx) + abs(next_loc[1] - sy)
        outward_gain_multiplier = 2.2 if self.phase == "opening" else 1.0
        outward_gain = self.outward_weight * outward_gain_multiplier * (nxt_out - cur_out)
        # Exploration incentives
        novelty = self._novelty_bonus(next_loc)
        frontier_step = self._frontier_step_bonus(cur, next_loc)
        coverage = self._coverage_penalty(next_loc)
        branching = self._branching_factor(board, next_loc)
        # Enemy turd proximity (radius-2)
        enemy_turds_near = self._count_enemy_turds_within(board, next_loc, radius=2)
        # Local open space (radius-2)
        open_space = self._local_open_space(board, next_loc)
        area_delta = self._area_delta(board, cur, next_loc)
        # Lightweight cut-in trigger: move toward interior when we can egg soon
        next_is_eggable = board.can_lay_egg_at_loc(next_loc)
        egg_in_two = self._egg_in_two_steps(board, next_loc)
        # Edge escape: encourage moving away from borders when cramped
        cur_edge = self._distance_to_border(cur)
        nxt_edge = self._distance_to_border(next_loc)
        edge_escape = max(0, nxt_edge - cur_edge)
        cur_center_d = self._distance_to_center(cur)
        nxt_center_d = self._distance_to_center(next_loc)
        cut_in_bonus = 0.0
        if nxt_center_d < cur_center_d and (next_is_eggable or egg_in_two):
            cut_in_bonus = 2.3 + (0.7 if (egg_in_two and not next_is_eggable) else 0.0)
        # Diagonal corner pursuit: aim toward best corner away from enemy
        diag_bonus = 0.0
        sep_bonus_raw = 0.0
        vor_bonus_raw = 0.0
        corner = self.corner_target
        cur_d_corner = self._manhattan(cur, corner)
        nxt_d_corner = self._manhattan(next_loc, corner)
        if self.phase == "opening":
            diag_bonus = 3.0 * float(cur_d_corner - nxt_d_corner)
            enemy_loc = board.chicken_enemy.get_location()
            sep_now = self._manhattan(cur, enemy_loc)
            sep_next = self._manhattan(next_loc, enemy_loc)
            sep_bonus_raw = 0.6 * float(sep_next - sep_now)
            vor_bonus_raw = 0.2 * float(self._local_voronoi_advantage(board, next_loc, radius=5))
        else:
            if stagnating and cur_d_corner > 0:
                diag_wt = 1.6 + 1.0 * stuck_pressure
                diag_bonus = diag_wt * float(cur_d_corner - nxt_d_corner)
        # Stronger local territory (BFS Voronoi) bonus to push cutoffs/coverage
        territory_adv = self._territory_diff_bfs(board, next_loc, radius=5)
        territory_cur = self._territory_diff_bfs(board, cur, radius=5)
        territory_delta = float(territory_adv - territory_cur)
        if self.phase == "opening":
            territory_weight = 0.45
        elif self.phase == "midgame":
            territory_weight = 0.35
        else:
            territory_weight = 0.25
        # Gate separation: only reward separation when it increases our territory
        sep_bonus = sep_bonus_raw if territory_delta > 0.0 else 0.0
        # Add explicit reward for improving territory vs current
        delta_weight = 0.6 if self.phase == "opening" else (0.35 if self.phase == "midgame" else 0.25)
        # Harmonize: clamp territory metrics and risk-modulate their influence
        t_adv = self._clamp(float(territory_adv), -8.0, 8.0)
        t_delta = self._clamp(float(territory_delta), -4.0, 4.0)
        risk_scale = 1.0 / (1.0 + 2.0 * max(0.0, risk_next - 0.2))
        territory_contrib = 0.0
        # Harmonize: if BFS territory is already strong, damp local Voronoi bonus to avoid double counting
        vor_bonus = 0.0 if abs(t_adv) >= 4.0 else 0.5 * vor_bonus_raw

        base = 32.0 + parity_bonus
        lane_wt = 5.0 * (1.0 - 0.4 * stuck_pressure)
        if stagnating:
            lane_wt = 0.0
        base += lane_wt * lane_progress
        # Stronger choke to encourage cutoffs and separation
        base += 9.5 * choke
        base += path_pull
        base += outward_gain
        # Novelty boosted slightly when stagnating
        base += (novelty * (1.0 + 0.4 * stuck_pressure)) + 2.2
        # Stuck pressure increases frontier drive; add edge bias
        frontier_wt = (7.2 if self.phase == "opening" else 6.2) + 3.0 * stuck_pressure
        dist_border = self._distance_to_border(cur)
        if dist_border <= 1:
            frontier_wt += 1.0
        elif dist_border <= 2:
            frontier_wt += 0.5
        if mode == "expand":
            frontier_wt += 2.0
        elif mode == "egg":
            frontier_wt += 0.8
        base += frontier_wt * frontier_step
        base += 2.0 * open_space
        if mode == "expand":
            base += 0.8 * novelty + 0.8 * open_space
        base += 1.4 * area_delta
        base += cut_in_bonus
        base += diag_bonus + sep_bonus + vor_bonus
        if enemy_route_target is not None:
            dist_cur_route = self._manhattan(cur, enemy_route_target)
            dist_next_route = self._manhattan(next_loc, enemy_route_target)
            route_gain = float(dist_cur_route - dist_next_route)
            if route_gain > 0:
                base += (1.5 + 0.5 * stuck_pressure) * route_gain
        base += self._stuck_escape_bonus(board, cur, next_loc)
        # Collision avoidance: penalize stepping into predicted enemy next tile
        if enemy_forecast is not None and next_loc == enemy_forecast:
            # Allow if it meaningfully improves territory under low risk
            if not (territory_delta > 2.0 and risk_next < 0.6):
                base -= 10.0
        # Intercept adjacency: only reward if it protects our corner path
        if (
            enemy_forecast_adjacent
            and territory_delta > 0.0
            and self.corner_target is not None
        ):
            if self._manhattan(enemy_forecast, self.corner_target) <= self._manhattan(cur, self.corner_target):
                base += min(3.0, 1.0 + 0.5 * territory_delta)
        base -= 18.0 * risk_next
        # Extra penalty for proximity to enemy turds in opening (mobility preservation)
        if self.phase == "opening" and enemy_turds_near > 0:
            base -= min(8.0, 2.5 * enemy_turds_near) * (1.0 if branching <= 2 else 0.6)
        base -= 2.5 * dist_center
        # Edge escape encouragement
        if self.phase != "endgame":
            base += 1.5 * (1.0 + 0.5 * stuck_pressure) * float(edge_escape)
        # Delay plain moves that don't lead to near-term egg in the opening
        if self.phase == "opening":
            base -= 1.5 * future_egg
        else:
            base -= 1.8 * future_egg
        # Mode-specific shaping
        if mode == "cutoff":
            base += 3.0 * choke + 1.5 * path_pull
        elif mode == "egg":
            base += 0.8 * future_egg  # soften anti-egg penalty when we want to egg
        # (no escape mode)
        # Corridor/dead-end awareness: discourage moving into dead-ends early unless it cuts off space or enables quick egg
        if self.phase == "opening" and branching <= 1:
            if not (territory_delta > 0.0 or egg_in_two):
                base -= 6.0
        # Corner approach planning: encourage stepping into a safe corner we can soon egg
        if (
            mt == MoveType.PLAIN
            and self._is_corner(next_loc)
            and risk_next < 0.8
        ):
            if next_loc not in board.eggs_player and next_loc not in board.turds_player and next_loc not in board.turds_enemy:
                corner_plan = 2.5 + 1.2 * self._novelty_bonus(next_loc)
                if egg_in_two:
                    corner_plan += 1.0
                base += corner_plan
        # Discourage revisits, especially if a safe novel option exists; stronger when stagnating
        base -= 1.8 * coverage * (1.0 + 0.6 * stuck_pressure)
        if self._safe_novel_exists:
            if coverage >= 2.0:
                base -= 5.0
            elif coverage >= 1.0:
                base -= 1.6 * (1.0 + 0.5 * stuck_pressure)
        if self.phase == "opening":
            visits = int(self.visit_counts[next_loc[1], next_loc[0]])
            if visits >= 1:
                base -= (6.0 + 4.0 * visits)
        # Avoid immediate backtracks to the previous position
        if next_loc == self.prev_pos:
            base -= 16.0
        # Cycle avoidance: penalize re-entering short cycles (A-B-A, A-B-C-A) more when stagnating
        cycle_pen = self._cycle_penalty(next_loc)
        base -= min(12.0, cycle_pen * (1.0 + 0.5 * stuck_pressure))
        # Opening anti-regression: avoid moving back toward spawn or against lane
        if self.phase == "opening":
            reg_out = max(0, cur_out - nxt_out)
            reg_lane = max(0.0, -lane_progress)
            if (reg_out > 0 or reg_lane > 0) and not (risk_next + 0.15 < risk_here):
                base -= 10.0 + 6.0 * (reg_out + reg_lane)
            if board.turn_count <= 3 and (reg_out > 0 or reg_lane > 0) and not (risk_next + 0.05 < risk_here):
                base -= 10.0
        # Heading inertia: mildly prefer continuing current heading to avoid oscillation
        if mt == MoveType.PLAIN and self.prev_pos != cur:
            dx = cur[0] - self.prev_pos[0]
            dy = cur[1] - self.prev_pos[1]
            last_dir: Optional[Direction] = None
            if dx == 1 and dy == 0:
                last_dir = Direction.RIGHT
            elif dx == -1 and dy == 0:
                last_dir = Direction.LEFT
            elif dx == 0 and dy == 1:
                last_dir = Direction.DOWN
            elif dx == 0 and dy == -1:
                last_dir = Direction.UP
            if last_dir is not None and dir_ == last_dir:
                base += 0.8
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
        terr_adv = self._territory_diff_bfs(board, next_loc, radius=5)
        return 28.0 - 2.8 * dist - 22.0 * risk + 0.3 * float(terr_adv)

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

    def _corner_sprint_move(
        self, board: Board, legal_moves: Sequence[Tuple[Direction, MoveType]]
    ) -> Optional[Tuple[Direction, MoveType]]:
        if board.turn_count >= self.CORNER_SPRINT_TURNS:
            return None
        if self.corner_target is None:
            return None
        cur = board.chicken_player.get_location()
        dist_cur = self._manhattan(cur, self.corner_target)
        if dist_cur <= 1:
            return None
        best_move: Optional[Tuple[Direction, MoveType]] = None
        best_score = float("inf")
        for dir_, mt in legal_moves:
            nxt = loc_after_direction(cur, dir_)
            if not board.is_valid_cell(nxt):
                continue
            if board.is_cell_blocked(nxt):
                continue
            next_dist = self._manhattan(nxt, self.corner_target)
            if next_dist >= dist_cur and dist_cur > 2:
                continue
            risk = self._risk_at(nxt)
            score = next_dist + 0.6 * risk
            if mt == MoveType.EGG:
                if not board.can_lay_egg_at_loc(nxt):
                    continue
                score += 1.5
            if score < best_score:
                best_score = score
                best_move = (dir_, mt)
        return best_move

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
    
    # Removed perimeter orientation and hazard-distance helpers to restore prior behavior

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

    # ------------------------------------------------------------------ #
    # Exploration helpers
    # ------------------------------------------------------------------ #
    def _record_visit(self, loc: Tuple[int, int]) -> None:
        x, y = loc
        if 0 <= x < self.size and 0 <= y < self.size:
            self.visit_counts[y, x] = min(255, self.visit_counts[y, x] + 1)
            self.visited_tiles.add(loc)

    def _maybe_refresh_frontier(self, board: Board) -> None:
        # Refresh if no target, already reached, or stale
        if (
            self.frontier_target is None
            or self.frontier_target in self.visited_tiles
            or board.turn_count >= self.frontier_refresh_turn
        ):
            self.frontier_target = self._compute_frontier_target(board)
            self.frontier_refresh_turn = board.turn_count + 6

    def _compute_frontier_target(self, board: Board) -> Optional[Tuple[int, int]]:
        start = board.chicken_player.get_location()
        best_target: Optional[Tuple[int, int]] = None
        best_score = -1e9
        seen = set([start])
        q: deque[Tuple[Tuple[int, int], int]] = deque([(start, 0)])
        max_depth = 18
        center = (self.size - 1) / 2.0
        sx, sy = self.spawn
        while q:
            loc, dist = q.popleft()
            if dist > max_depth:
                continue
            x, y = loc
            if not self._in_bounds(loc):
                continue
            # Scoring: prefer novel tiles, closer distance, lower risk
            visits = int(self.visit_counts[y, x])
            novelty = 3.0 / (1.0 + visits)
            risk = self._risk_at(loc)
            center_dist = abs(x - center) + abs(y - center)
            outward = abs(x - sx) + abs(y - sy)
            score = 6.0 * novelty - 0.3 * dist - 0.6 * risk
            # Opening bias: nudge toward centerline and outward expansion
            if self.phase == "opening":
                score += -0.12 * center_dist + 0.05 * outward
            if score > best_score:
                best_score = score
                best_target = loc
            for dir_ in Direction:
                nxt = loc_after_direction(loc, dir_)
                if nxt in seen:
                    continue
                if not board.is_valid_cell(nxt):
                    continue
                seen.add(nxt)
                q.append((nxt, dist + 1))
        return best_target

    def _novelty_bonus(self, loc: Tuple[int, int]) -> float:
        x, y = loc
        if 0 <= x < self.size and 0 <= y < self.size:
            v = int(self.visit_counts[y, x])
            return 3.0 / (1.0 + v)
        return 0.0

    def _coverage_penalty(self, loc: Tuple[int, int]) -> float:
        x, y = loc
        if 0 <= x < self.size and 0 <= y < self.size:
            v = int(self.visit_counts[y, x])
            if v == 0:
                return 0.0
            if v == 1:
                return 1.0
            return 2.0
        return 0.0

    def _frontier_step_bonus(self, cur: Tuple[int, int], nxt: Tuple[int, int]) -> float:
        if self.frontier_target is None:
            return 0.0
        cur_d = abs(cur[0] - self.frontier_target[0]) + abs(cur[1] - self.frontier_target[1])
        nxt_d = abs(nxt[0] - self.frontier_target[0]) + abs(nxt[1] - self.frontier_target[1])
        return max(0.0, float(cur_d - nxt_d))

    def _local_open_space(self, board: Board, loc: Tuple[int, int]) -> float:
        # Count nearby accessible cells within manhattan radius 2
        count = 0
        total = 0
        x0, y0 = loc
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) + abs(dy) > 2:
                    continue
                x, y = x0 + dx, y0 + dy
                if not self._in_bounds((x, y)):
                    continue
                total += 1
                if not board.is_cell_blocked((x, y)):
                    count += 1
        if total == 0:
            return 0.0
        # Normalize around [0, ~1]
        return (count / total)
    
    def _local_voronoi_advantage(self, board: Board, loc: Tuple[int, int], radius: int = 5) -> int:
        """Approximate local territory advantage: cells closer to us than enemy within a radius."""
        ex, ey = board.chicken_enemy.get_location()
        lx, ly = loc
        score = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) + abs(dy) > radius:
                    continue
                x = lx + dx
                y = ly + dy
                if not self._in_bounds((x, y)):
                    continue
                if board.is_cell_blocked((x, y)):
                    continue
                d_self = abs(dx) + abs(dy)
                d_enemy = abs(x - ex) + abs(y - ey)
                if d_self < d_enemy:
                    score += 1
                elif d_self > d_enemy:
                    score -= 1
        return score
    
    def _bfs_dist_map(self, board: Board, start: Tuple[int, int], radius: int) -> dict:
        """Compute BFS distances from start up to a given manhattan radius on passable cells."""
        dist = {start: 0}
        q = deque([start])
        while q:
            loc = q.popleft()
            d = dist[loc]
            if d >= radius:
                continue
            for dir_ in Direction:
                nxt = loc_after_direction(loc, dir_)
                if not self._in_bounds(nxt):
                    continue
                if board.is_cell_blocked(nxt):
                    continue
                if nxt in dist:
                    continue
                dist[nxt] = d + 1
                q.append(nxt)
        return dist
    
    def _territory_diff_bfs(self, board: Board, origin: Tuple[int, int], radius: int = 5) -> int:
        """Approximate local territory difference using BFS-based Voronoi within a radius."""
        self_map = self._bfs_dist_map(board, origin, radius)
        enemy_loc = board.chicken_enemy.get_location()
        enemy_map = self._bfs_dist_map(board, enemy_loc, radius)
        ox, oy = origin
        score = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) + abs(dy) > radius:
                    continue
                x = ox + dx
                y = oy + dy
                if not self._in_bounds((x, y)):
                    continue
                if board.is_cell_blocked((x, y)):
                    continue
                ds = self_map.get((x, y))
                de = enemy_map.get((x, y))
                if ds is None and de is None:
                    continue
                if de is None:
                    score += 1
                elif ds is None:
                    score -= 1
                else:
                    if ds < de:
                        score += 1
                    elif ds > de:
                        score -= 1
        return score
    
    # (Local Voronoi advantage helper removed to restore earlier behavior)

    # (Open-region, mobility, and blotch-specific helpers removed in revert)

    def _risk_at(self, loc: Tuple[int, int]) -> float:
        x, y = loc
        if 0 <= x < self.size and 0 <= y < self.size:
            return float(self._risk_grid[y, x])
        return 10.0

    def _distance_to_center(self, loc: Tuple[int, int]) -> float:
        center = (self.size - 1) / 2.0
        return abs(loc[0] - center) + abs(loc[1] - center)
    
    def _best_diagonal_corner(self, board: Board) -> Tuple[int, int]:
        """Choose a corner target that is far from enemy and reasonably far from us to sweep toward."""
        corners = [(0, 0), (self.size - 1, 0), (0, self.size - 1), (self.size - 1, self.size - 1)]
        cur = board.chicken_player.get_location()
        enemy = board.chicken_enemy.get_location()
        best = corners[0]
        best_score = -1e9
        for cx, cy in corners:
            dist_enemy = abs(cx - enemy[0]) + abs(cy - enemy[1])
            dist_cur = abs(cx - cur[0]) + abs(cy - cur[1])
            score = 1.0 * dist_enemy + 0.5 * dist_cur
            if score > best_score:
                best_score = score
                best = (cx, cy)
        return best

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
    
    def _egg_in_two_steps(self, board: Board, origin: Tuple[int, int]) -> bool:
        """Return True if from origin we can reach an eggable tile in <=2 moves.
        Uses simple neighbor scan without simulating enemy responses."""
        if board.can_lay_egg_at_loc(origin):
            return True
        for dir_ in Direction:
            nxt = loc_after_direction(origin, dir_)
            if not self._in_bounds(nxt):
                continue
            if board.is_cell_blocked(nxt):
                continue
            if board.can_lay_egg_at_loc(nxt):
                return True
        return False

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
        # Corridor awareness: more valuable if current tile is a corridor/low-branching point
        branching_here = self._branching_factor(board, loc)
        corridor_bonus = 0.0
        if branching_here <= 1:
            corridor_bonus += 1.4
        elif branching_here == 2:
            corridor_bonus += 0.6
        # Local lane availability (unchanged base, scaled a bit)
        lanes = 0.0
        for dir_ in Direction:
            nxt = loc_after_direction(loc, dir_)
            if not self._in_bounds(nxt):
                continue
            if nxt in board.turds_player or nxt in board.turds_enemy:
                continue
            lanes += 0.25
        # More valuable if within moderate range of enemy (threat window)
        proximity = max(0.0, 3.0 - min(3.0, float(dist)))
        return choke + lanes + corridor_bonus + 0.4 * proximity

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
    
    def _branching_factor(self, board: Board, loc: Tuple[int, int]) -> int:
        """Count passable immediate neighbors. Low branching implies corridors/dead-ends."""
        cnt = 0
        for dir_ in Direction:
            nxt = loc_after_direction(loc, dir_)
            if not self._in_bounds(nxt):
                continue
            if board.is_cell_blocked(nxt):
                continue
            cnt += 1
        return cnt
    
    def _count_enemy_turds_within(self, board: Board, loc: Tuple[int, int], radius: int = 2) -> int:
        x0, y0 = loc
        count = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) + abs(dy) > radius:
                    continue
                x = x0 + dx
                y = y0 + dy
                if not self._in_bounds((x, y)):
                    continue
                if (x, y) in board.turds_enemy:
                    count += 1
        return count

    def _stagnation_flag(self) -> bool:
        if len(self.recent_positions) < 4:
            return False
        recent = list(self.recent_positions)[-4:]
        xs = [p[0] for p in recent]
        ys = [p[1] for p in recent]
        if max(xs) - min(xs) <= 1 or max(ys) - min(ys) <= 1:
            return True
        if len(set(recent)) <= 2:
            return True
        return False

    def _area_score(self, board: Board, start: Tuple[int, int], radius: int = 4) -> float:
        seen = set([start])
        q: deque[Tuple[Tuple[int, int], int]] = deque([(start, 0)])
        novel = 0
        total = 0
        while q:
            loc, dist = q.popleft()
            if dist > radius:
                continue
            if not self._in_bounds(loc):
                continue
            if board.is_cell_blocked(loc):
                continue
            total += 1
            x, y = loc
            if int(self.visit_counts[y, x]) == 0:
                novel += 1
            for dir_ in Direction:
                nxt = loc_after_direction(loc, dir_)
                if nxt in seen:
                    continue
                seen.add(nxt)
                q.append((nxt, dist + 1))
        if total == 0:
            return 0.0
        explored = total - novel
        return float(novel) + 0.35 * float(explored)

    def _area_delta(self, board: Board, cur: Tuple[int, int], nxt: Tuple[int, int]) -> float:
        return self._area_score(board, nxt) - self._area_score(board, cur)
    
    def _record_enemy(self, loc: Tuple[int, int]) -> None:
        self.enemy_history.append(loc)
        if len(self.enemy_history) < 2:
            self.enemy_heading = (0.0, 0.0)
            return
        dx = self.enemy_history[-1][0] - self.enemy_history[0][0]
        dy = self.enemy_history[-1][1] - self.enemy_history[0][1]
        steps = max(1, len(self.enemy_history) - 1)
        self.enemy_heading = (dx / steps, dy / steps)
    
    def _enemy_route_target(self, board: Board, horizon: int = 3) -> Optional[Tuple[int, int]]:
        if not self.enemy_history:
            return None
        hx, hy = self.enemy_heading
        if abs(hx) < 0.1 and abs(hy) < 0.1:
            return board.chicken_enemy.get_location()
        tx, ty = board.chicken_enemy.get_location()
        target = (tx, ty)
        for _ in range(horizon):
            tx = round(target[0] + hx)
            ty = round(target[1] + hy)
            nxt = (tx, ty)
            if not self._in_bounds(nxt) or board.is_cell_blocked(nxt):
                break
            target = nxt
        return target
    
    def _should_force_opening_egg(
        self, board: Board, cur: Tuple[int, int], nxt: Tuple[int, int]
    ) -> bool:
        if board.turn_count > 8:
            return False
        eggs = board.chicken_player.get_eggs_laid()
        if eggs >= self.opening_egg_goal:
            return False
        if not board.can_lay_egg_at_loc(nxt):
            return False
        if self._risk_at(nxt) >= 0.85:
            return False
        enemy_loc = board.chicken_enemy.get_location()
        if self._manhattan(nxt, enemy_loc) <= 1:
            return False
        return True
    
    def _stuck_escape_bonus(
        self, board: Board, cur: Tuple[int, int], nxt: Tuple[int, int]
    ) -> float:
        if not self._stagnation_flag():
            return 0.0
        bonus = 0.0
        if self.frontier_target is not None:
            cur_d = self._manhattan(cur, self.frontier_target)
            nxt_d = self._manhattan(nxt, self.frontier_target)
            bonus += 2.6 * max(0.0, float(cur_d - nxt_d))
        bonus += 1.2 * max(
            0.0, float(self._distance_to_border(cur) - self._distance_to_border(nxt))
        )
        bonus += 1.0 * max(0.0, self._area_score(board, nxt) - self._area_score(board, cur))
        return bonus
    
    def _predict_enemy_next_loc(self, board: Board) -> Optional[Tuple[int, int]]:
        """One-ply enemy forecast from current state using enemy policy (territory-aware)."""
        try:
            sim = board.get_copy()
            sim.reverse_perspective()
            mv = self._enemy_policy(sim, horizon=1)
            if mv is None:
                return None
            enemy_cur = board.chicken_enemy.get_location()
            return loc_after_direction(enemy_cur, mv[0])
        except Exception:
            return None

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
    
    def _clamp(self, value: float, lo: float, hi: float) -> float:
        if value < lo:
            return lo
        if value > hi:
            return hi
        return value
    
    def _cycle_penalty(self, next_loc: Tuple[int, int]) -> float:
        """Return penalty for stepping back into a short cycle relative to recent positions."""
        pen = 0.0
        n = len(self.recent_positions)
        if n >= 2 and self.recent_positions[-2] == next_loc:
            pen += 6.0
        if n >= 3 and self.recent_positions[-3] == next_loc:
            pen += 3.0
        # If the same tile appears frequently in recent path, add mild penalty
        if n >= 6:
            repeats = sum(1 for p in list(self.recent_positions)[-6:] if p == next_loc)
            if repeats >= 2:
                pen += 2.0
        return pen
    
    def _distance_to_border(self, loc: Tuple[int, int]) -> int:
        x, y = loc
        return min(x, y, self.size - 1 - x, self.size - 1 - y)
    
    def _select_mode(self, board: Board, territory_cur: int) -> str:
        """Pick a simple high-level mode to avoid competing heuristics."""
        if self.phase == "endgame":
            return "egg"
        if self.phase == "opening":
            if board.can_lay_egg() and self.moves_since_last_egg >= 2 and not self._safe_novel_exists:
                return "egg"
            return "expand"
        # midgame
        if self.stagnation_count >= 3 or territory_cur <= 0:
            return "expand"
        if board.can_lay_egg() and self.moves_since_last_egg >= 1:
            return "egg"
        return "cutoff"

