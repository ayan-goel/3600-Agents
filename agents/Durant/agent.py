from __future__ import annotations

import time
from collections import deque
from typing import Callable, Deque, List, Optional, Sequence, Set, Tuple

import numpy as np

from game.board import Board
from game.enums import Direction, MoveType, Result, loc_after_direction

from .trapdoor_belief import TrapdoorBelief

INF = 1_000_000.0


class SearchTimeout(Exception):
    """Raised when the lookahead budget runs out."""


Coord = Tuple[int, int]


class PlayerAgent:
    """
    DURANT: strong Bayesian + search agent.

    Design goals (vs Maldini):

    - Use the same high-quality trapdoor belief model.
    - Keep a real lookahead search (3-ply rollout with enemy policy).
    - Use a dense, well-structured evaluation:
        * egg diff
        * BFS-based local territory advantage
        * mobility
        * trap risk
        * choke on opponent / cutoffs
    - Still handle:
        * opening vs mid vs endgame phases
        * exploration vs egging tension
        * loop / oscillation avoidance
        * trapdoor recovery after teleports
        * opponent tracking + lightweight interception
    """

    OPENING_TURNS = 8
    LATE_GAME_TURNS = 10

    def __init__(self, board: Board, time_left: Callable[[], float]):
        # Static map info
        self.game_map = board.game_map
        self.size = self.game_map.MAP_SIZE

        # Bayesian trapdoor model
        self.trap_belief = TrapdoorBelief(self.game_map)

        # Parity & spawn info
        self.my_parity = board.chicken_player.even_chicken
        self.enemy_parity = board.chicken_enemy.even_chicken
        self.spawn = board.chicken_player.get_spawn()
        self.enemy_spawn = board.chicken_enemy.get_spawn()

        # Phase control
        self.phase: str = "opening"
        self.moves_since_last_egg: int = 0

        # Risk grid (from trapdoors + turds)
        self._risk_grid = np.zeros((self.size, self.size), dtype=np.float32)

        # Time management
        self._search_deadline: float = 0.0
        self.safety_buffer: float = 0.08
        self.min_budget: float = 0.03

        # Opening bias
        self._lane_dir = (
            Direction.RIGHT if self.spawn[0] <= self.size // 2 else Direction.LEFT
        )
        self._vertical_bias = (
            Direction.UP
            if self.spawn[1] > (self.size // 2)
            else Direction.DOWN
        )
        self._opening_script: List[Direction] = self._build_opening_script()

        # Exploration state
        self.visit_counts = np.zeros((self.size, self.size), dtype=np.uint8)
        self.visited_tiles: Set[Coord] = {self.spawn}
        self.frontier_target: Optional[Coord] = None
        self.frontier_refresh_turn: int = -1
        self.recent_positions: Deque[Coord] = deque(maxlen=12)
        self.prev_pos: Coord = self.spawn
        self.stagnation_count: int = 0

        # Outward expansion strength
        self.outward_weight: float = 1.25
        self._safe_novel_exists: bool = False

        # Opponent tracking
        self.enemy_positions: Deque[Coord] = deque(maxlen=10)
        self.enemy_last_pos: Optional[Coord] = None

        # Trapdoor recovery
        self.last_loc_before = board.chicken_player.get_location()
        self.trapdoor_recovery_mode: bool = False
        self.recovery_turns_left: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def play(
        self,
        board: Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable[[], float],
    ) -> Tuple[Direction, MoveType]:
        """Main entry point called by the engine each turn."""

        # 1. Register any trapdoors the engine has explicitly revealed
        self._register_known_trapdoors(board)

        # 2. Update Bayesian belief from sensors
        self.trap_belief.update(board.chicken_player, sensor_data)

        # 3. Position / teleport detection
        cur_loc = board.chicken_player.get_location()
        if self.last_loc_before is not None:
            # Teleport if we "jumped" more than one tile
            if self._manhattan(self.last_loc_before, cur_loc) > 2:
                self.trapdoor_recovery_mode = True
                self.recovery_turns_left = 8
        self.last_loc_before = cur_loc

        if self.recovery_turns_left > 0:
            self.recovery_turns_left -= 1
        else:
            self.trapdoor_recovery_mode = False

        # 4. Phase + risk grid
        self._update_phase(board)
        self._risk_grid = self._build_risk_grid(board)

        # 5. Exploration bookkeeping
        self._record_visit(cur_loc)
        self._maybe_refresh_frontier(board)

        # 6. Opponent tracking
        enemy_loc = board.chicken_enemy.get_location()
        if self.enemy_last_pos is None or enemy_loc != self.enemy_last_pos:
            self.enemy_positions.append(enemy_loc)
            self.enemy_last_pos = enemy_loc

        # 7. Legal moves
        legal_moves = board.get_valid_moves()
        if not legal_moves:
            return Direction.UP, MoveType.PLAIN

        # Opening book moves for first few turns
        opening = self._opening_move(board, legal_moves)
        if opening is not None:
            choice = opening
        else:
            try:
                choice = self._select_move(board, legal_moves, time_left)
            except SearchTimeout:
                choice = self._fast_greedy(board, legal_moves)

        # 8. Egg bookkeeping
        if choice[1] == MoveType.EGG:
            self.moves_since_last_egg = 0
        else:
            self.moves_since_last_egg += 1

        # 9. Loop / stagnation bookkeeping
        if not self.recent_positions or self.recent_positions[-1] != cur_loc:
            self.recent_positions.append(cur_loc)

        nxt_loc = loc_after_direction(cur_loc, choice[0])
        novelty_next = nxt_loc not in self.visited_tiles
        terr_cur = self._territory_diff_bfs(board, cur_loc, radius=5)
        terr_nxt = self._territory_diff_bfs(board, nxt_loc, radius=5)
        terr_delta = terr_nxt - terr_cur
        if novelty_next or terr_delta > 0:
            self.stagnation_count = max(0, self.stagnation_count - 1)
        else:
            self.stagnation_count = min(8, self.stagnation_count + 1)

        self.prev_pos = cur_loc
        return choice

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def _select_move(
        self,
        board: Board,
        legal_moves: Sequence[Tuple[Direction, MoveType]],
        time_left: Callable[[], float],
    ) -> Tuple[Direction, MoveType]:
        """Pick a move via shallow lookahead rollout."""

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
        """Adjust search depth based on phase & branching."""
        depth = 3
        if self.phase == "endgame":
            depth += 1
        if len(legal_moves) <= 4:
            depth += 1
        return min(depth, 4)

    def _branch_limit(self, depth: int) -> int:
        if depth >= 4:
            return 5
        if depth == 3:
            return 7
        return 9

    def _order_moves(
        self, board: Board, legal_moves: Sequence[Tuple[Direction, MoveType]]
    ) -> List[Tuple[Direction, MoveType]]:
        """Score moves with a static heuristic for move ordering."""

        cur = board.chicken_player.get_location()
        self._safe_novel_exists = False
        for dir_, mt in legal_moves:
            if mt != MoveType.PLAIN:
                continue
            nxt = loc_after_direction(cur, dir_)
            if not board.is_valid_cell(nxt):
                continue
            if (
                self._risk_at(nxt) <= 0.9
                and int(self.visit_counts[nxt[1], nxt[0]]) == 0
            ):
                self._safe_novel_exists = True
                break

        scored: List[Tuple[float, Tuple[Direction, MoveType]]] = []
        for mv in legal_moves:
            scored.append((self._score_move(board, mv), mv))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

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
        """One-step us, one-step enemy, then recurse with static eval mix."""
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

        # Enemy reply from their perspective
        reply = child.get_copy()
        reply.reverse_perspective()
        enemy_move = self._enemy_policy(reply)
        if enemy_move is None:
            return base_score
        if not reply.apply_move(*enemy_move):
            reply.reverse_perspective()
            return base_score - 8.0

        reply.reverse_perspective()
        winner = reply.get_winner()
        if winner == Result.PLAYER:
            return INF * 0.5
        if winner == Result.ENEMY:
            return -INF * 0.5

        if horizon <= 2:
            return 0.65 * self._static_eval(reply) + 0.35 * base_score

        # One more ply: we move again with ordered subset
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
        self, board: Board
    ) -> Optional[Tuple[Direction, MoveType]]:
        """Simple greedy enemy model (maximizes their static eval)."""
        legal = board.get_valid_moves()
        if not legal:
            return None
        scored = []
        for mv in legal:
            scored.append((self._score_enemy_move(board, mv), mv))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    # ------------------------------------------------------------------
    # Static evaluation & move scoring
    # ------------------------------------------------------------------
    def _static_eval(self, board: Board) -> float:
        """Heuristic score of a board from our perspective."""
        eggs_self = board.chicken_player.get_eggs_laid()
        eggs_opp = board.chicken_enemy.get_eggs_laid()
        egg_diff = eggs_self - eggs_opp

        mobility_self = len(board.get_valid_moves())
        mobility_opp = len(board.get_valid_moves(enemy=True))

        terr_self = self._available_sites(board, friendly=True)
        terr_opp = self._available_sites(board, friendly=False)

        risk_here = self._risk_at(board.chicken_player.get_location())
        choke = self._enemy_choke_bonus(board, board.chicken_player.get_location())

        score = 150.0 * egg_diff
        score += 5.0 * (terr_self - terr_opp)
        score += 3.0 * (mobility_self - mobility_opp)
        score -= 40.0 * risk_here
        score += 10.0 * choke
        score -= 0.4 * board.turns_left_player
        return score

    def _score_move(self, board: Board, move: Tuple[Direction, MoveType]) -> float:
        """Static score for move ordering & greedy fallback."""
        dir_, mt = move
        cur = board.chicken_player.get_location()
        nxt = loc_after_direction(cur, dir_)

        risk_cur = self._risk_at(cur)
        risk_nxt = self._risk_at(nxt)
        enemy_forecast = self._predict_enemy_next_loc(board)

        eggs_self = board.chicken_player.get_eggs_laid()
        eggs_opp = board.chicken_enemy.get_eggs_laid()
        egg_diff = eggs_self - eggs_opp

        dist_center_cur = self._distance_to_center(cur)
        dist_center_nxt = self._distance_to_center(nxt)

        # Novelty & coverage
        novelty = self._novelty_bonus(nxt)
        coverage = self._coverage_penalty(nxt)
        frontier_step = self._frontier_step_bonus(cur, nxt)
        open_space = self._local_open_space(board, nxt)

        # Trapdoor recovery: aggressive push into truly new tiles
        if self.trapdoor_recovery_mode and nxt in self.visited_tiles:
            novelty -= 2.0

        # Local territory change
        terr_cur = self._territory_diff_bfs(board, cur, radius=5)
        terr_nxt = self._territory_diff_bfs(board, nxt, radius=5)
        terr_delta = terr_nxt - terr_cur
        terr_adv = terr_nxt

        # Distance from spawn (outward expansion)
        sx, sy = self.spawn
        cur_out = abs(cur[0] - sx) + abs(cur[1] - sy)
        nxt_out = abs(nxt[0] - sx) + abs(nxt[1] - sy)
        outward_gain = (nxt_out - cur_out) * (
            1.7 if self.phase == "opening" else 1.1
        )

        # Simple enemy pressure / cutoff
        choke = self._enemy_choke_bonus(board, nxt)
        cutoff = self._enemy_cutoff_bonus(board, nxt)

        # Branching (corridor awareness)
        branching = self._branching_factor(board, nxt)

        # Distance to border
        edge_cur = self._distance_to_border(cur)
        edge_nxt = self._distance_to_border(nxt)
        edge_escape = max(0, edge_nxt - edge_cur)

        # Egg-specific scoring
        if mt == MoveType.EGG:
            base = 110.0
            if self.moves_since_last_egg >= 2:
                base += 15.0
            chain = self._egg_chain_strength(board, nxt, depth=3)
            base += 10.0 * chain
            base += self._corner_bonus(cur)
            base -= 55.0 * risk_cur
            base -= 25.0 * max(0.0, risk_nxt - 0.5)
            base -= 3.0 * dist_center_cur
            base += 6.0 * max(0, -egg_diff)
            if self._is_my_parity(cur):
                base += 6.0
            base += 2.5 * self._novelty_bonus(cur)

            # Opening throttling: still lay eggs early but not every move
            if self.phase == "opening":
                base -= 18.0
                if self.moves_since_last_egg < 1:
                    base -= 10.0
                if self._safe_novel_exists and chain < 1.0:
                    base -= 8.0

            return base

        if mt == MoveType.TURD:
            block = self._turd_block_value(board, cur)
            base = 12.0 * block
            if egg_diff >= 2:
                base -= 12.0
            else:
                base += 4.0
            base -= 18.0 * risk_cur
            base -= 8.0 * risk_nxt
            return base

        # Plain moves ----------------------------------------------------
        base = 30.0

        # Lane progress (go “forward” off spawn in opening)
        lane_prog = self._lane_progress(cur, nxt)
        base += 5.0 * lane_prog

        # Outward expansion
        base += self.outward_weight * outward_gain

        # Novelty & frontier targeting
        stuck_pressure = min(1.0, float(self.stagnation_count) / 4.0)
        frontier_weight = (7.0 if self.phase == "opening" else 6.0) + 1.5 * stuck_pressure
        base += 2.0 * novelty
        base += frontier_weight * frontier_step

        # Territory
        terr_adv_clamped = self._clamp(float(terr_adv), -8.0, 8.0)
        terr_delta_clamped = self._clamp(float(terr_delta), -4.0, 4.0)
        terr_weight = 0.45 if self.phase == "opening" else (0.35 if self.phase == "midgame" else 0.25)
        delta_weight = 0.6 if self.phase == "opening" else (0.4 if self.phase == "midgame" else 0.25)
        risk_scale = 1.0 / (1.0 + 2.0 * max(0.0, risk_nxt - 0.2))
        terr_contrib = risk_scale * (
            terr_weight * terr_adv_clamped + delta_weight * max(0.0, terr_delta_clamped)
        )
        terr_contrib = self._clamp(terr_contrib, -6.0, 6.0)
        base += terr_contrib

        # Open space / mobility feeling
        base += 2.0 * open_space

        # Enemy choke / cutoff
        base += 9.0 * choke
        base += cutoff

        # Center: avoid early, allowed mid/endgame
        if self.phase == "opening":
            base -= 2.5 * dist_center_nxt
        else:
            base -= 1.3 * dist_center_nxt

        # Edge escape
        if self.phase != "endgame":
            base += 1.4 * float(edge_escape)

        # Penalize high risk
        base -= 18.0 * risk_nxt

        # Stronger penalty near predicted enemy square (collision)
        if enemy_forecast is not None and nxt == enemy_forecast:
            if not (terr_delta > 2.0 and risk_nxt < 0.6):
                base -= 10.0

        # Coverage / revisits
        base -= 2.3 * coverage * (1.0 + 0.5 * stuck_pressure)
        if self._safe_novel_exists and coverage >= 2.0:
            base -= 8.0
        if self.phase != "opening" and self.stagnation_count >= 4 and coverage >= 2.0:
            base -= 4.0 + 0.7 * float(self.stagnation_count - 3)

        # Extra penalty for revisiting in recovery mode
        if self.trapdoor_recovery_mode and nxt in self.visited_tiles:
            base -= 12.0

        # Avoid immediate backtracks
        if nxt == self.prev_pos:
            base -= 14.0

        # Cycle / oscillation penalties
        base -= min(12.0, self._cycle_penalty(nxt))
        base -= self._oscillation_penalty(cur, nxt)

        # Heading inertia (slight preference to keep direction)
        if self.prev_pos != cur:
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
                base += 0.7

        # Parity bonus – enjoy tiles where we can egg soon
        if self._is_my_parity(nxt):
            base += 10.0

        # Slightly prefer moves that get us closer to an eggable tile in mid/endgame
        if self.phase != "opening":
            nearest_cur = self._nearest_egg_distance(board, cur, limit=6)
            nearest_nxt = self._nearest_egg_distance(board, nxt, limit=6)
            if nearest_nxt < nearest_cur:
                base += 4.0 * (nearest_cur - nearest_nxt)

        # Don't run deep into 1-branch dead-ends early unless they clearly help
        if self.phase == "opening" and branching <= 1 and terr_delta <= 0 and not board.can_lay_egg_at_loc(nxt):
            base -= 6.0

        return base

    def _score_enemy_move(self, board: Board, move: Tuple[Direction, MoveType]) -> float:
        """Enemy heuristic: similar structure but tuned “against us”."""
        dir_, mt = move
        cur = board.chicken_player.get_location()
        nxt = loc_after_direction(cur, dir_)
        risk = self._risk_at(nxt)

        if mt == MoveType.EGG:
            chain = self._egg_chain_strength(board, nxt, depth=2)
            score = 110.0 + 8.0 * chain - 40.0 * risk
            if self._is_corner(cur):
                score += 10.0
            return score

        if mt == MoveType.TURD:
            return -30.0 - 12.0 * risk

        target = self._enemy_lane_target(board)
        dist = self._manhattan(nxt, target)
        terr_adv = self._territory_diff_bfs(board, nxt, radius=5)
        return 28.0 - 2.8 * dist - 20.0 * risk + 0.3 * float(terr_adv)

    # ------------------------------------------------------------------
    # Phase / opening logic
    # ------------------------------------------------------------------
    def _update_phase(self, board: Board) -> None:
        if board.turn_count < self.OPENING_TURNS:
            self.phase = "opening"
        elif board.turns_left_player <= self.LATE_GAME_TURNS:
            self.phase = "endgame"
        else:
            self.phase = "midgame"

    def _build_opening_script(self) -> List[Direction]:
        """Simple scripted first few moves off spawn."""
        forward = Direction.RIGHT if self.spawn[0] == 0 else Direction.LEFT
        script = [forward, forward]
        script.append(self._vertical_bias)
        script.append(forward)
        script.append(self._vertical_bias)
        script.append(forward)
        return script

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

    # ------------------------------------------------------------------
    # Trapdoor risk grid & bookkeeping
    # ------------------------------------------------------------------
    def _register_known_trapdoors(self, board: Board) -> None:
        for loc in getattr(board, "found_trapdoors", set()):
            self.trap_belief.register_known_trapdoor(loc)

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

    def _risk_at(self, loc: Coord) -> float:
        x, y = loc
        if 0 <= x < self.size and 0 <= y < self.size:
            return float(self._risk_grid[y, x])
        return 10.0

    # ------------------------------------------------------------------
    # Exploration helpers
    # ------------------------------------------------------------------
    def _record_visit(self, loc: Coord) -> None:
        x, y = loc
        if 0 <= x < self.size and 0 <= y < self.size:
            self.visit_counts[y, x] = min(255, self.visit_counts[y, x] + 1)
            self.visited_tiles.add(loc)

    def _maybe_refresh_frontier(self, board: Board) -> None:
        stuck_refresh = (
            self.phase != "opening"
            and (self.stagnation_count >= 3 or self.moves_since_last_egg >= 4)
        )
        if (
            self.frontier_target is None
            or self.frontier_target in self.visited_tiles
            or board.turn_count >= self.frontier_refresh_turn
            or stuck_refresh
        ):
            self.frontier_target = self._compute_frontier_target(board)
            self.frontier_refresh_turn = board.turn_count + (4 if stuck_refresh else 6)

    def _compute_frontier_target(self, board: Board) -> Optional[Coord]:
        start = board.chicken_player.get_location()
        best_target: Optional[Coord] = None
        best_score = -INF
        seen = set([start])
        q: Deque[Tuple[Coord, int]] = deque([(start, 0)])
        max_depth = 18
        center = (self.size - 1) / 2.0
        sx, sy = self.spawn
        stuck_mode = self.phase != "opening" and (
            self.stagnation_count >= 3 or self.moves_since_last_egg >= 4
        )
        egg_seeking = self.phase != "opening" and self.moves_since_last_egg >= 2
        dist_penalty = 0.18 if stuck_mode else 0.3

        while q:
            loc, dist = q.popleft()
            if dist > max_depth:
                continue
            x, y = loc
            if not self._in_bounds(loc):
                continue
            visits = int(self.visit_counts[y, x])
            novelty = 3.0 / (1.0 + visits)
            risk = self._risk_at(loc)
            center_dist = abs(x - center) + abs(y - center)
            outward = abs(x - sx) + abs(y - sy)

            score = 6.0 * novelty - dist_penalty * dist - 0.6 * risk
            if self.phase == "opening":
                score += -0.12 * center_dist + 0.05 * outward
            elif stuck_mode:
                score += 0.3 * outward - 0.08 * center_dist + 0.2 * dist

            if egg_seeking and board.can_lay_egg_at_loc(loc):
                score += 4.0
                if dist >= 4:
                    score += 2.0

            if score > best_score:
                best_score = score
                best_target = loc

            for d in Direction:
                nxt = loc_after_direction(loc, d)
                if nxt in seen:
                    continue
                if not board.is_valid_cell(nxt):
                    continue
                seen.add(nxt)
                q.append((nxt, dist + 1))

        return best_target

    # basic novelty/coverage/frontier/local-open-space helpers
    def _novelty_bonus(self, loc: Coord) -> float:
        x, y = loc
        if 0 <= x < self.size and 0 <= y < self.size:
            v = int(self.visit_counts[y, x])
            return 3.0 / (1.0 + v)
        return 0.0

    def _coverage_penalty(self, loc: Coord) -> float:
        x, y = loc
        if 0 <= x < self.size and 0 <= y < self.size:
            v = int(self.visit_counts[y, x])
            if v == 0:
                return 0.0
            if v == 1:
                return 1.0
            return 2.0
        return 0.0

    def _frontier_step_bonus(self, cur: Coord, nxt: Coord) -> float:
        if self.frontier_target is None:
            return 0.0
        cur_d = self._manhattan(cur, self.frontier_target)
        nxt_d = self._manhattan(nxt, self.frontier_target)
        return max(0.0, float(cur_d - nxt_d))

    def _local_open_space(self, board: Board, loc: Coord) -> float:
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
        return count / total

    # Territory / Voronoi-style helpers
    def _bfs_dist_map(self, board: Board, start: Coord, radius: int) -> dict:
        dist = {start: 0}
        q: Deque[Coord] = deque([start])
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

    def _territory_diff_bfs(
        self, board: Board, origin: Coord, radius: int = 5
    ) -> int:
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

    # ------------------------------------------------------------------
    # Opponent / cutoff helpers
    # ------------------------------------------------------------------
    def _enemy_choke_bonus(self, board: Board, loc: Coord) -> float:
        enemy_loc = board.chicken_enemy.get_location()
        dist = self._manhattan(loc, enemy_loc)
        if dist <= 2:
            return 4.0 - dist
        lane = self._lane_distance(loc)
        return max(0.0, 3.0 - 0.3 * lane)

    def _enemy_cutoff_bonus(self, board: Board, loc: Coord) -> float:
        """Reward moves that reduce enemy reachable unexplored space."""
        enemy_loc = board.chicken_enemy.get_location()
        my_loc = board.chicken_player.get_location()

        def count_reachable_open(start: Coord, blocked_by: Optional[Coord]) -> int:
            visited = {start}
            if blocked_by:
                visited.add(blocked_by)
            q: Deque[Coord] = deque([start])
            count = 0
            depth = 0
            max_depth = 4
            while q and depth < max_depth:
                nxt_layer: Deque[Coord] = deque()
                while q:
                    pos = q.popleft()
                    for d in Direction:
                        nxt = loc_after_direction(pos, d)
                        if nxt in visited:
                            continue
                        if not self._in_bounds(nxt):
                            continue
                        if board.is_cell_blocked(nxt):
                            continue
                        visited.add(nxt)
                        nxt_layer.append(nxt)
                        if nxt not in self.visited_tiles:
                            count += 1
                q = nxt_layer
                depth += 1
            return count

        open_with = count_reachable_open(enemy_loc, blocked_by=loc)
        open_without = count_reachable_open(enemy_loc, blocked_by=None)
        reduction = open_without - open_with
        bonus = 0.0
        if reduction > 0:
            bonus = min(10.0, 2.5 * reduction)

        # Extra bonus if we're moving closer to enemy's predicted path
        pred_dir = self._predict_enemy_direction()
        if pred_dir is not None:
            pred_enemy_next = loc_after_direction(enemy_loc, pred_dir)
            if self._manhattan(loc, pred_enemy_next) < self._manhattan(my_loc, pred_enemy_next):
                bonus += 3.0
        return bonus

    def _predict_enemy_direction(self) -> Optional[Direction]:
        if len(self.enemy_positions) < 2:
            return None
        positions = list(self.enemy_positions)[-3:]
        if len(positions) < 2:
            return None
        dx = positions[-1][0] - positions[0][0]
        dy = positions[-1][1] - positions[0][1]
        if abs(dx) > abs(dy):
            return Direction.RIGHT if dx > 0 else Direction.LEFT
        elif abs(dy) > 0:
            return Direction.DOWN if dy > 0 else Direction.UP
        return None

    def _predict_enemy_next_loc(self, board: Board) -> Optional[Coord]:
        try:
            sim = board.get_copy()
            sim.reverse_perspective()
            mv = self._enemy_policy(sim)
            if mv is None:
                return None
            enemy_cur = board.chicken_enemy.get_location()
            return loc_after_direction(enemy_cur, mv[0])
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Egg / turd helpers
    # ------------------------------------------------------------------
    def _turd_block_value(self, board: Board, loc: Coord) -> float:
        enemy_loc = board.chicken_enemy.get_location()
        dist = self._manhattan(loc, enemy_loc)
        choke = 0.0
        if dist <= 2:
            choke += 3.0 - dist
        branching_here = self._branching_factor(board, loc)
        corridor_bonus = 0.0
        if branching_here <= 1:
            corridor_bonus += 1.4
        elif branching_here == 2:
            corridor_bonus += 0.6
        lanes = 0.0
        for d in Direction:
            nxt = loc_after_direction(loc, d)
            if not self._in_bounds(nxt):
                continue
            if nxt in board.turds_player or nxt in board.turds_enemy:
                continue
            lanes += 0.25
        proximity = max(0.0, 3.0 - min(3.0, float(dist)))
        return choke + lanes + corridor_bonus + 0.4 * proximity

    def _egg_chain_strength(
        self, board: Board, origin: Coord, depth: int = 3
    ) -> float:
        visited = set([origin])
        q: Deque[Tuple[Coord, int]] = deque([(origin, 0)])
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
            for d in Direction:
                nxt = loc_after_direction(loc, d)
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

    def _nearest_egg_distance(
        self, board: Board, start: Coord, limit: int = 5
    ) -> int:
        visited = set([start])
        q: Deque[Tuple[Coord, int]] = deque([(start, 0)])
        while q:
            loc, dist = q.popleft()
            if dist > limit:
                break
            if board.can_lay_egg_at_loc(loc):
                return dist
            for d in Direction:
                nxt = loc_after_direction(loc, d)
                if nxt in visited or not self._in_bounds(nxt):
                    continue
                if board.is_cell_blocked(nxt):
                    continue
                visited.add(nxt)
                q.append((nxt, dist + 1))
        return limit + 1

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    def _lane_progress(self, cur: Coord, nxt: Coord) -> float:
        if self._lane_dir in (Direction.RIGHT, Direction.LEFT):
            forward = 1 if self._lane_dir == Direction.RIGHT else -1
            return forward * (nxt[0] - cur[0])
        forward = 1 if self._lane_dir == Direction.DOWN else -1
        return forward * (nxt[1] - cur[1])

    def _lane_distance(self, loc: Coord) -> int:
        target_x = self.size - 1 if self.spawn[0] == 0 else 0
        return abs(loc[0] - target_x)
    
    def _enemy_lane_target(self, board: Board) -> Coord:
        """
        Heuristic 'goal' square for the enemy when scoring their moves.

        We assume the enemy wants to advance along a horizontal lane starting
        from their spawn toward the opposite side of the map, just like we do.
        """
        sx, sy = self.enemy_spawn  # enemy spawn from original perspective
        # If they spawn on the left, target the right edge; otherwise target left.
        target_x = self.size - 1 if sx == 0 else 0
        return (target_x, sy)

    def _available_sites(self, board: Board, friendly: bool) -> int:
        parity = self.my_parity if friendly else self.enemy_parity
        eggs = board.eggs_player if friendly else board.eggs_enemy
        my_turds = board.turds_player if friendly else board.turds_enemy
        opp_turds = board.turds_enemy if friendly else board.turds_player
        cnt = 0
        for x in range(self.size):
            for y in range(self.size):
                if (x + y) % 2 != parity:
                    continue
                loc = (x, y)
                if loc in eggs or loc in my_turds or loc in opp_turds:
                    continue
                cnt += 1
        return cnt

    def _branching_factor(self, board: Board, loc: Coord) -> int:
        cnt = 0
        for d in Direction:
            nxt = loc_after_direction(loc, d)
            if not self._in_bounds(nxt):
                continue
            if board.is_cell_blocked(nxt):
                continue
            cnt += 1
        return cnt

    def _corner_bonus(self, loc: Coord) -> float:
        x, y = loc
        if (x in (0, self.size - 1)) and (y in (0, self.size - 1)):
            return 18.0
        return 0.0

    def _manhattan(self, a: Coord, b: Coord) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _distance_to_center(self, loc: Coord) -> float:
        center = (self.size - 1) / 2.0
        return abs(loc[0] - center) + abs(loc[1] - center)

    def _distance_to_border(self, loc: Coord) -> int:
        x, y = loc
        return min(x, y, self.size - 1 - x, self.size - 1 - y)

    def _in_bounds(self, loc: Coord) -> bool:
        x, y = loc
        return 0 <= x < self.size and 0 <= y < self.size

    def _is_corner(self, loc: Coord) -> bool:
        x, y = loc
        return (x in (0, self.size - 1)) and (y in (0, self.size - 1))

    def _is_my_parity(self, loc: Coord) -> bool:
        return (loc[0] + loc[1]) % 2 == self.my_parity

    def _clamp(self, v: float, lo: float, hi: float) -> float:
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    def _cycle_penalty(self, next_loc: Coord) -> float:
        pen = 0.0
        n = len(self.recent_positions)
        if n >= 2 and self.recent_positions[-2] == next_loc:
            pen += 6.0
        if n >= 3 and self.recent_positions[-3] == next_loc:
            pen += 3.0
        if n >= 6:
            repeats = sum(
                1 for p in list(self.recent_positions)[-6:] if p == next_loc
            )
            if repeats >= 2:
                pen += 2.0
        return pen

    def _oscillation_penalty(self, cur: Coord, next_loc: Coord) -> float:
        n = len(self.recent_positions)
        if n < 4:
            return 0.0
        recent = list(self.recent_positions)[-8:]
        unique_recent = set(recent)
        if len(unique_recent) <= 2 and len(recent) >= 4:
            if next_loc in unique_recent:
                duration = len(recent) - 3
                return 28.0 + 7.0 * duration
            else:
                return -12.0
        if n >= 4:
            if (
                recent[-1] == recent[-3]
                and recent[-2] == recent[-4]
                and next_loc == recent[-2]
            ):
                return 18.0
        return 0.0