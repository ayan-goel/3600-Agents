"""
BOBBY - Reactive/Defensive Agent
Strategy: Avoid center, control perimeter, let opponent take risks
This is the "ultra-defensive" style that waits and reacts
"""
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
    """DEFENSIVE agent - avoids center, controls perimeter, lets opponent take risks."""

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
        # Exploration state
        self.visit_counts = np.zeros((self.size, self.size), dtype=np.uint8)
        self.visited_tiles: Set[Tuple[int, int]] = {self.spawn}
        self.frontier_target: Optional[Tuple[int, int]] = None
        self.frontier_refresh_turn: int = -1
        self.prev_pos: Tuple[int, int] = self.spawn
        self.recent_positions: deque[Tuple[int, int]] = deque(maxlen=12)
        self.stagnation_count: int = 0
        self.outward_weight: float = 1.25
        self._safe_novel_exists: bool = False
        # Defensive tracking
        self.enemy_positions: deque[Tuple[int, int]] = deque(maxlen=10)
        self.enemy_last_pos: Optional[Tuple[int, int]] = None

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
        self._track_opponent(board)
        
        cur_loc = board.chicken_player.get_location()
        self._record_visit(cur_loc)
        self._maybe_refresh_frontier(board)

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
        
        # BOBBY: DEFENSIVE EARLY GAME - Stay on perimeter, avoid center!
        # Key: Let opponent rush center and hit trapdoors
        if 3 <= board.turn_count <= 30:
            enemy_loc = board.chicken_enemy.get_location()
            center = (self.size // 2, self.size // 2)
            
            our_dist_to_center = self._manhattan(cur_loc, center)
            enemy_dist_to_center = self._manhattan(enemy_loc, center)
            dist_to_enemy = self._manhattan(cur_loc, enemy_loc)
            
            if choice[1] != MoveType.EGG:
                best_defensive = None
                best_defensive_score = -1e9
                
                for mv in legal_moves:
                    def_loc = loc_after_direction(cur_loc, mv[0])
                    if not board.is_valid_cell(def_loc) or board.is_cell_blocked(def_loc):
                        continue
                    
                    score = 0.0
                    new_dist_to_center = self._manhattan(def_loc, center)
                    new_dist_to_enemy = self._manhattan(def_loc, enemy_loc)
                    
                    # STRONG penalty for moving toward center (trapdoor risk!)
                    if new_dist_to_center < our_dist_to_center:
                        score -= 100.0  # Heavy penalty
                    elif new_dist_to_center > our_dist_to_center:
                        score += 40.0  # Reward staying on perimeter
                    
                    # Bonus for edge tiles (safe!)
                    if def_loc[0] == 0 or def_loc[0] == self.size - 1:
                        score += 50.0
                    if def_loc[1] == 0 or def_loc[1] == self.size - 1:
                        score += 50.0
                    
                    # Bonus for corners (very safe, good chains)
                    if self._is_corner(def_loc):
                        score += 70.0
                    
                    # DON'T chase opponent - let them come to us
                    if new_dist_to_enemy < dist_to_enemy:
                        score -= 30.0  # Penalize chasing
                    elif new_dist_to_enemy > dist_to_enemy:
                        score += 20.0  # Reward staying away
                    
                    # Bonus for unvisited tiles
                    if def_loc not in self.visited_tiles:
                        score += 60.0
                    
                    # Bonus for egg moves
                    if mv[1] == MoveType.EGG:
                        score += 80.0
                    
                    # Bonus for parity
                    if self._is_my_parity(def_loc):
                        score += 25.0
                    
                    # STRONG penalty for risk (very defensive)
                    score -= 120.0 * self._risk_at(def_loc)
                    
                    if score > best_defensive_score:
                        best_defensive_score = score
                        best_defensive = mv
                
                if best_defensive is not None and best_defensive_score > -50.0:
                    choice = best_defensive
        
        # BOBBY: MID-GAME PERIMETER CONTROL
        # Continue controlling edges, only enter center if opponent is trapped
        if 30 <= board.turn_count <= 60 and choice[1] != MoveType.EGG:
            enemy_loc = board.chicken_enemy.get_location()
            center = (self.size // 2, self.size // 2)
            our_dist_to_center = self._manhattan(cur_loc, center)
            enemy_dist_to_center = self._manhattan(enemy_loc, center)
            
            # Only go toward center if enemy is stuck there (we've surrounded them)
            enemy_is_trapped = enemy_dist_to_center <= 2 and our_dist_to_center >= 3
            
            if not enemy_is_trapped:
                # Stay on perimeter
                best_perimeter = None
                best_perimeter_score = -1e9
                
                for mv in legal_moves:
                    per_loc = loc_after_direction(cur_loc, mv[0])
                    if not board.is_valid_cell(per_loc) or board.is_cell_blocked(per_loc):
                        continue
                    
                    score = 0.0
                    new_dist_to_center = self._manhattan(per_loc, center)
                    
                    # Prefer staying away from center
                    if new_dist_to_center >= our_dist_to_center:
                        score += 30.0
                    else:
                        score -= 40.0
                    
                    # Bonus for edges
                    if per_loc[0] == 0 or per_loc[0] == self.size - 1:
                        score += 35.0
                    if per_loc[1] == 0 or per_loc[1] == self.size - 1:
                        score += 35.0
                    
                    if per_loc not in self.visited_tiles:
                        score += 50.0
                    
                    if mv[1] == MoveType.EGG:
                        score += 60.0
                    
                    if self._is_my_parity(per_loc):
                        score += 20.0
                    
                    score -= 80.0 * self._risk_at(per_loc)
                    
                    if score > best_perimeter_score:
                        best_perimeter_score = score
                        best_perimeter = mv
                
                if best_perimeter is not None and best_perimeter_score > 0:
                    choice = best_perimeter
            else:
                # Enemy is trapped - NOW we can squeeze them
                best_squeeze = None
                best_squeeze_score = -1e9
                
                for mv in legal_moves:
                    sq_loc = loc_after_direction(cur_loc, mv[0])
                    if not board.is_valid_cell(sq_loc) or board.is_cell_blocked(sq_loc):
                        continue
                    
                    score = 0.0
                    new_dist_to_enemy = self._manhattan(sq_loc, enemy_loc)
                    
                    # Now we can close in
                    if new_dist_to_enemy < self._manhattan(cur_loc, enemy_loc):
                        score += 40.0
                    
                    if sq_loc not in self.visited_tiles:
                        score += 40.0
                    
                    if mv[1] == MoveType.EGG:
                        score += 50.0
                    
                    score -= 60.0 * self._risk_at(sq_loc)
                    
                    if score > best_squeeze_score:
                        best_squeeze_score = score
                        best_squeeze = mv
                
                if best_squeeze is not None and best_squeeze_score > 20.0:
                    choice = best_squeeze
        
        # BOBBY: NEVER TURD unless way ahead and safe
        if choice[1] == MoveType.TURD:
            my_eggs = board.chicken_player.get_eggs_laid()
            opp_eggs = board.chicken_enemy.get_eggs_laid()
            if my_eggs - opp_eggs < 4:  # Only turd if 4+ eggs ahead
                non_turd = [mv for mv in legal_moves if mv[1] != MoveType.TURD]
                if non_turd:
                    egg_moves = [mv for mv in non_turd if mv[1] == MoveType.EGG]
                    if egg_moves:
                        choice = egg_moves[0]
                    else:
                        choice = non_turd[0]
        
        # Oscillation override
        if len(self.recent_positions) >= 6:
            recent = list(self.recent_positions)[-6:]
            unique_tiles = set(recent)
            if len(unique_tiles) <= 2:
                next_loc = loc_after_direction(cur_loc, choice[0])
                if next_loc in unique_tiles:
                    for mv in legal_moves:
                        escape_loc = loc_after_direction(cur_loc, mv[0])
                        if escape_loc not in unique_tiles and board.is_valid_cell(escape_loc):
                            if not board.is_cell_blocked(escape_loc):
                                choice = mv
                                break

        if choice[1] == MoveType.EGG:
            self.moves_since_last_egg = 0
        else:
            self.moves_since_last_egg += 1
        
        if not self.recent_positions or self.recent_positions[-1] != cur_loc:
            self.recent_positions.append(cur_loc)
        
        self.prev_pos = cur_loc
        return choice

    def _track_opponent(self, board: Board) -> None:
        enemy_loc = board.chicken_enemy.get_location()
        if self.enemy_last_pos is not None and enemy_loc != self.enemy_last_pos:
            self.enemy_positions.append(enemy_loc)
        self.enemy_last_pos = enemy_loc

    # ------------------------------------------------------------------ #
    # Search and move selection (same as Messi)
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
        self._safe_novel_exists = False
        cur = board.chicken_player.get_location()
        for dir_, mt in legal_moves:
            if mt != MoveType.PLAIN:
                continue
            nxt = loc_after_direction(cur, dir_)
            if not board.is_valid_cell(nxt):
                continue
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
    # Move scoring - DEFENSIVE version
    # ------------------------------------------------------------------ #
    def _score_move(self, board: Board, move: Tuple[Direction, MoveType]) -> float:
        dir_, mt = move
        cur = board.chicken_player.get_location()
        next_loc = loc_after_direction(cur, dir_)
        risk_here = self._risk_at(cur)
        risk_next = self._risk_at(next_loc)
        center = (self.size // 2, self.size // 2)
        
        eggs_self = board.chicken_player.get_eggs_laid()
        eggs_opp = board.chicken_enemy.get_eggs_laid()
        egg_diff = eggs_self - eggs_opp

        if mt == MoveType.EGG:
            base = 120.0  # Higher egg priority
            if self.moves_since_last_egg >= 2:
                base += 20.0
            chain = self._egg_chain_strength(board, next_loc, depth=3)
            base += 12.0 * chain
            base += self._corner_bonus(cur) * 1.5  # Extra corner bonus
            base -= 80.0 * risk_here  # Very risk-averse
            if self._is_my_parity(cur):
                base += 8.0
            return base

        if mt == MoveType.TURD:
            # Very reluctant to turd
            if egg_diff < 4:
                return -100.0  # Almost never turd unless way ahead
            block = self._turd_block_value(board, cur)
            return 5.0 * block - 20.0

        # PLAIN moves - defensive scoring
        base = 32.0
        
        # AVOID CENTER - key defensive behavior
        cur_center_dist = self._manhattan(cur, center)
        next_center_dist = self._manhattan(next_loc, center)
        if next_center_dist < cur_center_dist:
            base -= 25.0  # Penalize moving toward center
        elif next_center_dist > cur_center_dist:
            base += 15.0  # Reward moving away from center
        
        # Strong edge preference
        if next_loc[0] == 0 or next_loc[0] == self.size - 1:
            base += 20.0
        if next_loc[1] == 0 or next_loc[1] == self.size - 1:
            base += 20.0
        
        # Corner bonus
        if self._is_corner(next_loc):
            base += 25.0
        
        # Parity and novelty
        if self._is_my_parity(next_loc):
            base += 12.0
        if next_loc not in self.visited_tiles:
            base += 15.0
        
        # Outward expansion (but on perimeter)
        sx, sy = self.spawn
        cur_out = abs(cur[0] - sx) + abs(cur[1] - sy)
        nxt_out = abs(next_loc[0] - sx) + abs(next_loc[1] - sy)
        base += 1.5 * (nxt_out - cur_out)
        
        # STRONG risk penalty (very defensive)
        base -= 30.0 * risk_next
        
        # Avoid backtracking
        if next_loc == self.prev_pos:
            base -= 15.0
        
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

    def _static_eval(self, board: Board) -> float:
        eggs_self = board.chicken_player.get_eggs_laid()
        eggs_opp = board.chicken_enemy.get_eggs_laid()
        egg_diff = eggs_self - eggs_opp

        mobility_self = len(board.get_valid_moves())
        mobility_opp = len(board.get_valid_moves(enemy=True))
        territory_self = self._available_sites(board, friendly=True)
        territory_opp = self._available_sites(board, friendly=False)
        risk_here = self._risk_at(board.chicken_player.get_location())
        
        # Extra weight on eggs and safety for defensive play
        score = 160.0 * egg_diff  # Higher egg weight
        score += 5.0 * (territory_self - territory_opp)
        score += 3.0 * (mobility_self - mobility_opp)
        score -= 60.0 * risk_here  # Higher risk penalty
        return score

    # ------------------------------------------------------------------ #
    # Opening guidance - DEFENSIVE version
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
            if egg_moves and self._risk_at(board.chicken_player.get_location()) < 0.7:  # More conservative
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
        # Defensive opening - stay on edges, don't rush center
        forward = Direction.RIGHT if self.spawn[0] == 0 else Direction.LEFT
        # Move along edge first, then outward
        edge_dir = self._vertical_bias
        script = [forward, edge_dir, forward, edge_dir, forward, edge_dir]
        return script

    # ------------------------------------------------------------------ #
    # Helpers (same as Messi)
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
                # Defensive: prefer tiles away from center
                priority = -0.3 * dist_center + 0.7 * dist_spawn
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
        # BOBBY: Extra risk for center tiles
        center = self.size // 2
        for y in range(self.size):
            for x in range(self.size):
                dist_to_center = abs(x - center) + abs(y - center)
                if dist_to_center <= 2:
                    grid[y, x] += 0.5  # Extra risk for center
        return grid

    def _record_visit(self, loc: Tuple[int, int]) -> None:
        x, y = loc
        if 0 <= x < self.size and 0 <= y < self.size:
            self.visit_counts[y, x] = min(255, self.visit_counts[y, x] + 1)
            self.visited_tiles.add(loc)

    def _maybe_refresh_frontier(self, board: Board) -> None:
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
            # Defensive: prefer tiles AWAY from center
            score = 6.0 * novelty - 0.3 * dist - 0.8 * risk + 0.2 * center_dist
            # Bonus for edge tiles
            if x == 0 or x == self.size - 1 or y == 0 or y == self.size - 1:
                score += 2.0
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

    def _risk_at(self, loc: Tuple[int, int]) -> float:
        x, y = loc
        if 0 <= x < self.size and 0 <= y < self.size:
            return float(self._risk_grid[y, x])
        return 10.0

    def _corner_bonus(self, loc: Tuple[int, int]) -> float:
        x, y = loc
        if (x in (0, self.size - 1)) and (y in (0, self.size - 1)):
            return 25.0  # Higher corner bonus for defensive play
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

    def _turd_block_value(self, board: Board, loc: Tuple[int, int]) -> float:
        enemy_loc = board.chicken_enemy.get_location()
        dist = self._manhattan(loc, enemy_loc)
        choke = 0.0
        if dist <= 2:
            choke += 3.0 - dist
        return choke

    def _enemy_lane_target(self, board: Board) -> Tuple[int, int]:
        return self.spawn

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




