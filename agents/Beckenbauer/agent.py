"""
BECKENBAUER - Aggressive Agent
Strategy: Rush to center, cut off opponent early, dominate territory
This is the "old aggressive" style that rushes to intercept opponents
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
    """AGGRESSIVE agent - rushes to cut off opponent, dominates center."""

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
        # Aggressive tracking
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
        
        # BECKENBAUER: AGGRESSIVE EARLY GAME - Rush to cut off opponent!
        # This is the key differentiator - we ALWAYS try to intercept
        if 3 <= board.turn_count <= 25:
            enemy_loc = board.chicken_enemy.get_location()
            center = (self.size // 2, self.size // 2)
            
            dist_to_enemy = self._manhattan(cur_loc, enemy_loc)
            our_dist_to_center = self._manhattan(cur_loc, center)
            enemy_dist_to_center = self._manhattan(enemy_loc, center)
            
            # Always try to get to center faster than opponent
            if choice[1] != MoveType.EGG:
                best_aggressive = None
                best_aggressive_score = -1e9
                
                for mv in legal_moves:
                    agg_loc = loc_after_direction(cur_loc, mv[0])
                    if not board.is_valid_cell(agg_loc) or board.is_cell_blocked(agg_loc):
                        continue
                    
                    score = 0.0
                    new_dist_to_center = self._manhattan(agg_loc, center)
                    new_dist_to_enemy = self._manhattan(agg_loc, enemy_loc)
                    
                    # STRONG bonus for moving toward center
                    center_progress = our_dist_to_center - new_dist_to_center
                    score += 80.0 * center_progress
                    
                    # Bonus for closing distance to enemy
                    enemy_progress = dist_to_enemy - new_dist_to_enemy
                    score += 50.0 * enemy_progress
                    
                    # Bonus for getting between enemy and center
                    if new_dist_to_center < enemy_dist_to_center:
                        score += 60.0
                    
                    # Bonus for cutting off enemy's path
                    cutoff = self._enemy_cutoff_bonus(board, agg_loc)
                    score += 40.0 * cutoff
                    
                    # Bonus for unvisited tiles
                    if agg_loc not in self.visited_tiles:
                        score += 30.0
                    
                    # Bonus for egg moves
                    if mv[1] == MoveType.EGG:
                        score += 50.0
                    
                    # Bonus for parity
                    if self._is_my_parity(agg_loc):
                        score += 15.0
                    
                    # Penalty for risk (but less than defensive agents)
                    score -= 25.0 * self._risk_at(agg_loc)
                    
                    if score > best_aggressive_score:
                        best_aggressive_score = score
                        best_aggressive = mv
                
                if best_aggressive is not None and best_aggressive_score > 20.0:
                    choice = best_aggressive
        
        # BECKENBAUER: MID-GAME CHASE - Keep pressure on opponent
        if 25 <= board.turn_count <= 60 and choice[1] != MoveType.EGG:
            enemy_loc = board.chicken_enemy.get_location()
            dist_to_enemy = self._manhattan(cur_loc, enemy_loc)
            
            # If enemy is getting away, chase them
            if dist_to_enemy >= 4:
                best_chase = None
                best_chase_score = -1e9
                
                for mv in legal_moves:
                    chase_loc = loc_after_direction(cur_loc, mv[0])
                    if not board.is_valid_cell(chase_loc) or board.is_cell_blocked(chase_loc):
                        continue
                    
                    score = 0.0
                    new_dist = self._manhattan(chase_loc, enemy_loc)
                    progress = dist_to_enemy - new_dist
                    
                    score += 60.0 * progress
                    
                    if chase_loc not in self.visited_tiles:
                        score += 40.0
                    
                    if mv[1] == MoveType.EGG:
                        score += 45.0
                    
                    score -= 30.0 * self._risk_at(chase_loc)
                    
                    if score > best_chase_score:
                        best_chase_score = score
                        best_chase = mv
                
                if best_chase is not None and best_chase_score > 30.0:
                    choice = best_chase
        
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

    def _enemy_cutoff_bonus(self, board: Board, loc: Tuple[int, int]) -> float:
        """Calculate bonus for blocking enemy's access to open space."""
        enemy_loc = board.chicken_enemy.get_location()
        
        # Count open tiles reachable by enemy before and after we move to loc
        enemy_open_before = 0
        enemy_open_after = 0
        
        for x in range(self.size):
            for y in range(self.size):
                tile = (x, y)
                if board.is_cell_blocked(tile):
                    continue
                if tile in self.visited_tiles:
                    continue
                
                dist_to_enemy = self._manhattan(tile, enemy_loc)
                dist_to_us = self._manhattan(tile, loc)
                
                if dist_to_enemy <= 3:
                    enemy_open_before += 1
                    if dist_to_us < dist_to_enemy:
                        # We'd be closer - we cut them off
                        pass
                    else:
                        enemy_open_after += 1
        
        reduction = enemy_open_before - enemy_open_after
        return min(5.0, reduction * 0.5)

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
    # Move scoring - AGGRESSIVE version
    # ------------------------------------------------------------------ #
    def _score_move(self, board: Board, move: Tuple[Direction, MoveType]) -> float:
        dir_, mt = move
        cur = board.chicken_player.get_location()
        next_loc = loc_after_direction(cur, dir_)
        risk_here = self._risk_at(cur)
        risk_next = self._risk_at(next_loc)
        enemy_loc = board.chicken_enemy.get_location()
        center = (self.size // 2, self.size // 2)
        
        eggs_self = board.chicken_player.get_eggs_laid()
        eggs_opp = board.chicken_enemy.get_eggs_laid()
        egg_diff = eggs_self - eggs_opp

        if mt == MoveType.EGG:
            base = 110.0
            if self.moves_since_last_egg >= 2:
                base += 15.0
            chain = self._egg_chain_strength(board, next_loc, depth=3)
            base += 10.0 * chain
            base += self._corner_bonus(cur)
            base -= 40.0 * risk_here  # Less risk-averse
            if self._is_my_parity(cur):
                base += 6.0
            return base

        if mt == MoveType.TURD:
            block = self._turd_block_value(board, cur)
            base = 15.0 * block  # More willing to turd
            if egg_diff >= 2:
                base += 10.0  # Use turds when ahead
            return base

        # PLAIN moves - aggressive scoring
        base = 32.0
        
        # Strong center-seeking
        cur_center_dist = self._manhattan(cur, center)
        next_center_dist = self._manhattan(next_loc, center)
        center_progress = cur_center_dist - next_center_dist
        base += 8.0 * center_progress
        
        # Enemy interception bonus
        cur_enemy_dist = self._manhattan(cur, enemy_loc)
        next_enemy_dist = self._manhattan(next_loc, enemy_loc)
        enemy_progress = cur_enemy_dist - next_enemy_dist
        base += 6.0 * enemy_progress
        
        # Cutoff bonus
        cutoff = self._enemy_cutoff_bonus(board, next_loc)
        base += 12.0 * cutoff
        
        # Parity and novelty
        if self._is_my_parity(next_loc):
            base += 10.0
        if next_loc not in self.visited_tiles:
            base += 8.0
        
        # Outward expansion
        sx, sy = self.spawn
        cur_out = abs(cur[0] - sx) + abs(cur[1] - sy)
        nxt_out = abs(next_loc[0] - sx) + abs(next_loc[1] - sy)
        base += 2.0 * (nxt_out - cur_out)
        
        # Less risk penalty (more aggressive)
        base -= 12.0 * risk_next
        
        # Avoid backtracking
        if next_loc == self.prev_pos:
            base -= 12.0
        
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
        
        # Extra weight on territory for aggressive play
        score = 150.0 * egg_diff
        score += 8.0 * (territory_self - territory_opp)
        score += 4.0 * (mobility_self - mobility_opp)
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
        # More aggressive - push toward center faster
        script = [forward, forward, self._vertical_bias, forward, self._vertical_bias, forward]
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
            score = 6.0 * novelty - 0.3 * dist - 0.6 * risk - 0.15 * center_dist
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

    def _turd_block_value(self, board: Board, loc: Tuple[int, int]) -> float:
        enemy_loc = board.chicken_enemy.get_location()
        dist = self._manhattan(loc, enemy_loc)
        choke = 0.0
        if dist <= 2:
            choke += 3.0 - dist
        return choke + 1.0

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

