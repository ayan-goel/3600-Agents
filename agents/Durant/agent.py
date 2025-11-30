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


    OPENING_TURNS = 14  # Extended opening for full perimeter sweep
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
        # MALDINI: Perimeter sweep strategy
        # Determine our sweep direction: go vertical first (up or down), then cut across
        # If we spawn on left (x=0), we go vertical on left edge, then sweep right
        # If we spawn on right (x=7), we go vertical on right edge, then sweep left
        self._sweep_vertical = self._vertical_bias  # UP or DOWN based on spawn y
        self._sweep_horizontal = Direction.RIGHT if self.spawn[0] == 0 else Direction.LEFT
        self._perimeter_phase = "vertical"  # "vertical" -> "cut_in" -> "done"
        self._cut_in_turn = -1  # Turn when we should cut inward
        
        self._opening_script = self._build_opening_script()
        self.loop_targets = self._build_loop_targets()
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
        # MALDINI: Opponent tracking
        self.enemy_positions: deque[Tuple[int, int]] = deque(maxlen=10)
        self.enemy_last_pos: Optional[Tuple[int, int]] = None
        # MALDINI: Trapdoor recovery detection
        self.last_known_loc: Optional[Tuple[int, int]] = None
        self.trapdoor_recovery_mode: bool = False
        self.recovery_turns_left: int = 0

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
        
        # MALDINI: Track opponent position
        enemy_loc = board.chicken_enemy.get_location()
        if self.enemy_last_pos is None or enemy_loc != self.enemy_last_pos:
            self.enemy_positions.append(enemy_loc)
            self.enemy_last_pos = enemy_loc
        
        # MALDINI: Detect trapdoor reset (we jumped far from last position)
        if self.last_known_loc is not None:
            dist_from_last = self._manhattan(cur_loc, self.last_known_loc)
            # If we moved more than 2 tiles, we likely hit a trapdoor
            if dist_from_last > 2:
                self.trapdoor_recovery_mode = True
                self.recovery_turns_left = 10  # Stay in recovery mode for 10 turns
        
        if self.recovery_turns_left > 0:
            self.recovery_turns_left -= 1
        else:
            self.trapdoor_recovery_mode = False
        
        self.last_known_loc = cur_loc

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
        
        # MALDINI: NEVER RETREAT POLICY
        # If we're about to move backward toward our spawn, reconsider
        # This prevents us from being pushed back by aggressive opponents
        if choice[1] == MoveType.PLAIN:
            next_loc = loc_after_direction(cur_loc, choice[0])
            spawn_dist_cur = self._manhattan(cur_loc, self.spawn)
            spawn_dist_next = self._manhattan(next_loc, self.spawn)
            
            # If this move brings us closer to spawn (retreating), find alternative
            if spawn_dist_next < spawn_dist_cur and spawn_dist_cur >= 2:
                # Try to find a non-retreating move
                best_alt = None
                best_alt_score = -1e9
                
                for mv in legal_moves:
                    alt_loc = loc_after_direction(cur_loc, mv[0])
                    if not board.is_valid_cell(alt_loc) or board.is_cell_blocked(alt_loc):
                        continue
                    
                    alt_spawn_dist = self._manhattan(alt_loc, self.spawn)
                    
                    # Skip retreating moves
                    if alt_spawn_dist < spawn_dist_cur:
                        continue
                    
                    score = 0.0
                    # Prefer advancing (away from spawn)
                    score += 10.0 * (alt_spawn_dist - spawn_dist_cur)
                    # Prefer eggs
                    if mv[1] == MoveType.EGG:
                        score += 50.0
                    # Penalize risk
                    score -= 20.0 * self._risk_at(alt_loc)
                    # Prefer unvisited
                    if alt_loc not in self.visited_tiles:
                        score += 30.0
                    
                    if score > best_alt_score:
                        best_alt_score = score
                        best_alt = mv
                
                if best_alt is not None:
                    choice = best_alt
        
        # MALDINI: EARLY GAME EGG CHAIN PREFERENCE
        # In early game (first 20 turns), strongly prefer egg moves that build chains
        if board.turn_count <= 20:
            egg_moves = [mv for mv in legal_moves if mv[1] == MoveType.EGG]
            if egg_moves and choice[1] != MoveType.EGG:
                # Check if any egg move would create a good chain
                best_chain_egg = None
                best_chain_score = 0.0
                for mv in egg_moves:
                    next_loc = loc_after_direction(cur_loc, mv[0])
                    if self._risk_at(next_loc) > 0.5:
                        continue  # Skip risky eggs
                    
                    # Score based on chain potential
                    chain_score = 0.0
                    # Bonus for adjacent to existing eggs
                    adj_eggs = self._count_adjacent_eggs(board, cur_loc)
                    chain_score += 2.0 * adj_eggs
                    # Bonus for setting up future eggs
                    chain_score += self._egg_chain_score(board, next_loc, max_depth=2)
                    # Bonus for corners
                    if self._is_corner(cur_loc):
                        chain_score += 1.5
                    
                    if chain_score > best_chain_score:
                        best_chain_score = chain_score
                        best_chain_egg = mv
                
                # Take the chain egg if it's good enough
                if best_chain_egg is not None and best_chain_score >= 1.5:
                    choice = best_chain_egg
        
        # MALDINI: TRAPDOOR RECOVERY - Aggressively explore NEW areas after reset
        # When we hit a trapdoor, we need to pivot to TRULY unexplored territory
        # Key insight: Go to the OPPOSITE side of the board from where we currently are
        if self.trapdoor_recovery_mode:
            # Find tiles that NEITHER player has claimed
            truly_unexplored = []
            for x in range(self.size):
                for y in range(self.size):
                    tile = (x, y)
                    if tile in self.visited_tiles:
                        continue
                    if tile in board.eggs_enemy:
                        continue
                    if board.is_cell_blocked(tile):
                        continue
                    truly_unexplored.append(tile)
            
            if truly_unexplored:
                # Find which half of the board we're on and target the OTHER half
                mid = self.size // 2
                on_right_half = cur_loc[0] >= mid
                on_bottom_half = cur_loc[1] >= mid
                
                # Filter unexplored tiles to prefer the opposite side
                opposite_side_tiles = []
                for tile in truly_unexplored:
                    # Prefer tiles on opposite horizontal half
                    if on_right_half and tile[0] < mid:
                        opposite_side_tiles.append(tile)
                    elif not on_right_half and tile[0] >= mid:
                        opposite_side_tiles.append(tile)
                
                # If no tiles on opposite side, use all unexplored
                target_tiles = opposite_side_tiles if opposite_side_tiles else truly_unexplored
                
                # Find the LARGEST cluster of unexplored tiles
                def count_nearby_unexplored(tile: Tuple[int, int]) -> int:
                    count = 0
                    for t in truly_unexplored:
                        if self._manhattan(tile, t) <= 3:
                            count += 1
                    return count
                
                # Pick target that has most unexplored tiles nearby, preferring opposite side
                best_target = max(target_tiles, key=lambda t: (count_nearby_unexplored(t), -self._manhattan(cur_loc, t)))
                
                cur_dist = self._manhattan(cur_loc, best_target)
                
                # ALWAYS find the best recovery move - don't rely on base choice
                best_recovery = None
                best_recovery_score = -1e9
                for mv in legal_moves:
                    recovery_loc = loc_after_direction(cur_loc, mv[0])
                    if not board.is_valid_cell(recovery_loc) or board.is_cell_blocked(recovery_loc):
                        continue
                    
                    score = 0.0
                    new_dist = self._manhattan(recovery_loc, best_target)
                    progress = cur_dist - new_dist
                    
                    # HUGE bonus for progress toward unexplored region
                    score += 150.0 * progress
                    
                    # MASSIVE bonus for unvisited tiles
                    if recovery_loc not in self.visited_tiles:
                        score += 300.0
                    
                    # Bonus for tiles with many unexplored neighbors
                    nearby_unexplored = count_nearby_unexplored(recovery_loc)
                    score += 20.0 * nearby_unexplored
                    
                    # Extra bonus for moving toward the opposite side of the board
                    if on_right_half and recovery_loc[0] < cur_loc[0]:
                        score += 80.0  # Moving left when on right side
                    elif not on_right_half and recovery_loc[0] > cur_loc[0]:
                        score += 80.0  # Moving right when on left side
                    
                    # Extra bonus if this tile is also not near enemy eggs
                    near_enemy_egg = any(
                        self._manhattan(recovery_loc, egg) <= 1 
                        for egg in board.eggs_enemy
                    )
                    if not near_enemy_egg:
                        score += 40.0
                    
                    # Bonus for egg moves (still want to score)
                    if mv[1] == MoveType.EGG:
                        score += 100.0
                    
                    # Bonus for parity
                    if self._is_my_parity(recovery_loc):
                        score += 30.0
                    
                    # STRONG penalty for already visited - we MUST explore new areas
                    if recovery_loc in self.visited_tiles:
                        score -= 200.0
                    
                    # Penalty for going toward spawn (we want to go AWAY)
                    spawn_dist_now = self._manhattan(cur_loc, self.spawn)
                    spawn_dist_next = self._manhattan(recovery_loc, self.spawn)
                    if spawn_dist_next < spawn_dist_now:
                        score -= 100.0  # Penalize moving toward spawn
                    
                    # Penalty for risk
                    score -= 50.0 * self._risk_at(recovery_loc)
                    
                    if score > best_recovery_score:
                        best_recovery_score = score
                        best_recovery = mv
                
                # ALWAYS use recovery move if we found one (unless current is an egg)
                if best_recovery is not None and choice[1] != MoveType.EGG:
                    choice = best_recovery
        
        # HARD OSCILLATION OVERRIDE: if stuck in small area, force escape
        # Check for oscillation in last 8 moves - trigger if 4 or fewer unique tiles
        if len(self.recent_positions) >= 8:
            recent = list(self.recent_positions)[-8:]
            recent_unique = set(recent)
            # Stuck in 4 or fewer tiles over 8 moves = oscillating/stuck
            if len(recent_unique) <= 4:
                next_loc = loc_after_direction(cur_loc, choice[0])
                # If current choice keeps us in the stuck area, find escape
                if next_loc in recent_unique:
                    best_escape = None
                    best_escape_score = -1e9
                    for mv in legal_moves:
                        escape_loc = loc_after_direction(cur_loc, mv[0])
                        if not board.is_valid_cell(escape_loc) or board.is_cell_blocked(escape_loc):
                            continue
                        score = 0.0
                        # Strong preference for tiles outside our stuck area
                        if escape_loc not in recent_unique:
                            score += 100.0
                        # Huge bonus for completely unvisited tiles
                        if escape_loc not in self.visited_tiles:
                            score += 80.0
                        # Bonus for egg moves
                        if mv[1] == MoveType.EGG:
                            score += 40.0
                        # Bonus for parity tiles
                        if self._is_my_parity(escape_loc):
                            score += 20.0
                        # Penalty for risk
                        score -= 60.0 * self._risk_at(escape_loc)
                        if score > best_escape_score:
                            best_escape_score = score
                            best_escape = mv
                    if best_escape is not None and best_escape_score > 0:
                        choice = best_escape
        
        # MALDINI: AREA EXPANSION - If stuck in a small row/column range, push outward
        # This catches cases where we're not technically oscillating but stuck in one region
        if len(self.recent_positions) >= 10 and 25 <= board.turn_count <= 70:
            recent = list(self.recent_positions)[-10:]
            min_row = min(p[1] for p in recent)
            max_row = max(p[1] for p in recent)
            min_col = min(p[0] for p in recent)
            max_col = max(p[0] for p in recent)
            
            row_range = max_row - min_row
            col_range = max_col - min_col
            
            # If we've been stuck in a 3x3 or smaller area for 10 moves
            if row_range <= 3 and col_range <= 3 and choice[1] != MoveType.EGG:
                # Find unexplored rows/columns
                unexplored_rows = set(range(self.size))
                unexplored_cols = set(range(self.size))
                for tile in self.visited_tiles:
                    unexplored_cols.discard(tile[0])
                    unexplored_rows.discard(tile[1])
                
                # Find closest unexplored area
                best_push = None
                best_push_score = -1e9
                for mv in legal_moves:
                    push_loc = loc_after_direction(cur_loc, mv[0])
                    if not board.is_valid_cell(push_loc) or board.is_cell_blocked(push_loc):
                        continue
                    
                    score = 0.0
                    
                    # Bonus for moving toward unexplored rows/columns
                    if push_loc[0] in unexplored_cols:
                        score += 60.0
                    if push_loc[1] in unexplored_rows:
                        score += 60.0
                    
                    # Bonus for expanding our range
                    if push_loc[1] < min_row or push_loc[1] > max_row:
                        score += 40.0
                    if push_loc[0] < min_col or push_loc[0] > max_col:
                        score += 40.0
                    
                    if push_loc not in self.visited_tiles:
                        score += 50.0
                    if mv[1] == MoveType.EGG:
                        score += 30.0
                    if self._is_my_parity(push_loc):
                        score += 15.0
                    
                    score -= 40.0 * self._risk_at(push_loc)
                    
                    if score > best_push_score:
                        best_push_score = score
                        best_push = mv
                
                if best_push is not None and best_push_score > 50.0:
                    choice = best_push
        
        # MALDINI: MID-GAME EXPANSION - proactively push into open space
        # Trigger: turns 20-60, check for large unexplored regions
        if 20 <= board.turn_count <= 60:
            # Find the largest unexplored region
            unexplored_target = self._get_largest_unexplored_target(board)
            if unexplored_target is not None:
                # Check if current choice is moving AWAY from the unexplored region
                next_loc = loc_after_direction(cur_loc, choice[0])
                cur_dist_to_target = self._manhattan(cur_loc, unexplored_target)
                next_dist_to_target = self._manhattan(next_loc, unexplored_target)
                
                # If we're not making progress toward unexplored area and not laying an egg
                if next_dist_to_target >= cur_dist_to_target and choice[1] != MoveType.EGG:
                    # Find move that gets us closer to the unexplored region
                    best_expand = None
                    best_expand_score = -1e9
                    for mv in legal_moves:
                        expand_loc = loc_after_direction(cur_loc, mv[0])
                        if not board.is_valid_cell(expand_loc) or board.is_cell_blocked(expand_loc):
                            continue
                        
                        new_dist = self._manhattan(expand_loc, unexplored_target)
                        progress = cur_dist_to_target - new_dist  # Positive = getting closer
                        
                        score = 50.0 * progress  # Strong bonus for progress toward target
                        
                        # Huge bonus for actually reaching unvisited tiles
                        if expand_loc not in self.visited_tiles:
                            score += 80.0
                        # Bonus for egg moves
                        if mv[1] == MoveType.EGG:
                            score += 40.0
                        # Bonus for parity (can egg soon)
                        if self._is_my_parity(expand_loc):
                            score += 15.0
                        # Penalty for risk
                        score -= 40.0 * self._risk_at(expand_loc)
                        
                        if score > best_expand_score:
                            best_expand_score = score
                            best_expand = mv
                    
                    # Override if we found a good expansion move
                    if best_expand is not None and best_expand_score > 30.0:
                        choice = best_expand
        
        # MALDINI: BOARD HALF BALANCE - Don't let opponent dominate an entire half
        # Check if opponent has significantly more eggs on one half of the board
        if 25 <= board.turn_count <= 65 and choice[1] != MoveType.EGG:
            mid = self.size // 2
            
            # Count eggs on left vs right half
            our_left = sum(1 for egg in board.eggs_player if egg[0] < mid)
            our_right = sum(1 for egg in board.eggs_player if egg[0] >= mid)
            opp_left = sum(1 for egg in board.eggs_enemy if egg[0] < mid)
            opp_right = sum(1 for egg in board.eggs_enemy if egg[0] >= mid)
            
            # Count eggs on top vs bottom half  
            our_top = sum(1 for egg in board.eggs_player if egg[1] < mid)
            our_bottom = sum(1 for egg in board.eggs_player if egg[1] >= mid)
            opp_top = sum(1 for egg in board.eggs_enemy if egg[1] < mid)
            opp_bottom = sum(1 for egg in board.eggs_enemy if egg[1] >= mid)
            
            # Determine if opponent is dominating a half we haven't explored
            target_x = None
            target_y = None
            
            # If opponent has 3+ more eggs on right and we have few there
            if opp_right - our_right >= 3 and our_right <= 2:
                target_x = self.size - 1  # Push right
            # If opponent has 3+ more eggs on left and we have few there
            elif opp_left - our_left >= 3 and our_left <= 2:
                target_x = 0  # Push left
            
            # Same for vertical
            if opp_bottom - our_bottom >= 3 and our_bottom <= 2:
                target_y = self.size - 1  # Push down
            elif opp_top - our_top >= 3 and our_top <= 2:
                target_y = 0  # Push up
            
            if target_x is not None or target_y is not None:
                # We need to push toward the underexplored half
                best_balance = None
                best_balance_score = -1e9
                
                for mv in legal_moves:
                    balance_loc = loc_after_direction(cur_loc, mv[0])
                    if not board.is_valid_cell(balance_loc) or board.is_cell_blocked(balance_loc):
                        continue
                    
                    score = 0.0
                    
                    # Bonus for moving toward target half
                    if target_x is not None:
                        if target_x > mid:  # Push right
                            if balance_loc[0] > cur_loc[0]:
                                score += 60.0
                        else:  # Push left
                            if balance_loc[0] < cur_loc[0]:
                                score += 60.0
                    
                    if target_y is not None:
                        if target_y > mid:  # Push down
                            if balance_loc[1] > cur_loc[1]:
                                score += 60.0
                        else:  # Push up
                            if balance_loc[1] < cur_loc[1]:
                                score += 60.0
                    
                    # Bonus for unvisited tiles
                    if balance_loc not in self.visited_tiles:
                        score += 80.0
                    
                    # Bonus for egg moves
                    if mv[1] == MoveType.EGG:
                        score += 50.0
                    
                    # Bonus for parity
                    if self._is_my_parity(balance_loc):
                        score += 20.0
                    
                    # Penalty for risk
                    score -= 50.0 * self._risk_at(balance_loc)
                    
                    if score > best_balance_score:
                        best_balance_score = score
                        best_balance = mv
                
                if best_balance is not None and best_balance_score > 40.0:
                    choice = best_balance
        
        # MALDINI: CORNER RUSH - If opponent is heading toward an unclaimed corner, race them!
        # This prevents opponent from getting easy 3-point corners
        if 15 <= board.turn_count <= 70:
            corners = [(0, 0), (0, self.size - 1), (self.size - 1, 0), (self.size - 1, self.size - 1)]
            enemy_loc = board.chicken_enemy.get_location()
            
            for corner in corners:
                # Skip if corner is already claimed
                if corner in board.eggs_player or corner in board.eggs_enemy:
                    continue
                if board.is_cell_blocked(corner):
                    continue
                
                # Check if opponent is closer to this corner than us
                enemy_dist = self._manhattan(enemy_loc, corner)
                our_dist = self._manhattan(cur_loc, corner)
                
                # If opponent is within 3 moves of an unclaimed corner and closer than us
                if enemy_dist <= 3 and enemy_dist < our_dist:
                    # We need to race them or cut them off!
                    next_loc = loc_after_direction(cur_loc, choice[0])
                    next_dist = self._manhattan(next_loc, corner)
                    
                    # If current choice isn't moving toward the contested corner
                    if next_dist >= our_dist and choice[1] != MoveType.EGG:
                        best_race = None
                        best_race_score = -1e9
                        for mv in legal_moves:
                            race_loc = loc_after_direction(cur_loc, mv[0])
                            if not board.is_valid_cell(race_loc) or board.is_cell_blocked(race_loc):
                                continue
                            
                            new_dist = self._manhattan(race_loc, corner)
                            progress = our_dist - new_dist
                            
                            score = 70.0 * progress  # Strong incentive to race for corner
                            
                            if race_loc not in self.visited_tiles:
                                score += 40.0
                            if mv[1] == MoveType.EGG:
                                score += 50.0
                            if self._is_my_parity(race_loc):
                                score += 20.0
                            
                            score -= 40.0 * self._risk_at(race_loc)
                            
                            if score > best_race_score:
                                best_race_score = score
                                best_race = mv
                        
                        if best_race is not None and best_race_score > 40.0:
                            choice = best_race
                            break  # Only race for one corner at a time
        
        # MALDINI: OPPONENT TRACKING & INTERCEPTION
        # Throughout the game, try to cut off opponent from open space
        # But balance with our own exploration needs
        if 15 <= board.turn_count <= 65:
            enemy_loc = board.chicken_enemy.get_location()
            my_eggs = board.chicken_player.get_eggs_laid()
            opp_eggs = board.chicken_enemy.get_eggs_laid()
            
            # Only intercept if:
            # 1. We're not too far behind on eggs
            # 2. Enemy is heading toward unexplored territory
            # 3. We can intercept without going too far out of our way
            if opp_eggs - my_eggs <= 3:  # Not too far behind
                predicted_dir = self._predict_enemy_direction()
                if predicted_dir is not None:
                    # Where is enemy heading?
                    predicted_enemy_next = loc_after_direction(enemy_loc, predicted_dir)
                    
                    # Is enemy heading toward open space we haven't visited?
                    enemy_heading_to_open = predicted_enemy_next not in self.visited_tiles
                    
                    if enemy_heading_to_open:
                        # Check if we can intercept
                        dist_to_intercept = self._manhattan(cur_loc, predicted_enemy_next)
                        
                        # Only intercept if reasonably close (within 4 moves)
                        if dist_to_intercept <= 4:
                            next_loc = loc_after_direction(cur_loc, choice[0])
                            cur_dist_to_enemy_path = self._manhattan(cur_loc, predicted_enemy_next)
                            next_dist_to_enemy_path = self._manhattan(next_loc, predicted_enemy_next)
                            
                            # If current choice isn't moving toward intercept, consider override
                            if next_dist_to_enemy_path >= cur_dist_to_enemy_path and choice[1] != MoveType.EGG:
                                best_intercept = None
                                best_intercept_score = -1e9
                                for mv in legal_moves:
                                    intercept_loc = loc_after_direction(cur_loc, mv[0])
                                    if not board.is_valid_cell(intercept_loc) or board.is_cell_blocked(intercept_loc):
                                        continue
                                    
                                    new_dist = self._manhattan(intercept_loc, predicted_enemy_next)
                                    progress = cur_dist_to_enemy_path - new_dist
                                    
                                    score = 30.0 * progress  # Bonus for getting closer to intercept
                                    
                                    # Bonus if this move also explores
                                    if intercept_loc not in self.visited_tiles:
                                        score += 40.0
                                    # Bonus for egg moves
                                    if mv[1] == MoveType.EGG:
                                        score += 25.0
                                    # Penalty for risk
                                    score -= 30.0 * self._risk_at(intercept_loc)
                                    
                                    if score > best_intercept_score:
                                        best_intercept_score = score
                                        best_intercept = mv
                                
                                # Only intercept if it's a good move overall
                                if best_intercept is not None and best_intercept_score > 20.0:
                                    choice = best_intercept
        
        # MALDINI: TERRITORIAL AWARENESS - Cover the opposite side from opponent
        # If opponent is dominating one side, we should ensure we're covering the other
        if 25 <= board.turn_count <= 70:
            my_eggs = board.chicken_player.get_eggs_laid()
            opp_eggs = board.chicken_enemy.get_eggs_laid()
            
            # Only do this if we're behind or close
            if opp_eggs >= my_eggs:
                # Determine which quadrant opponent is dominating
                enemy_eggs_left = sum(1 for egg in board.eggs_enemy if egg[0] < self.size // 2)
                enemy_eggs_right = sum(1 for egg in board.eggs_enemy if egg[0] >= self.size // 2)
                enemy_eggs_top = sum(1 for egg in board.eggs_enemy if egg[1] < self.size // 2)
                enemy_eggs_bottom = sum(1 for egg in board.eggs_enemy if egg[1] >= self.size // 2)
                
                # Find which quadrant we should target (opposite of enemy's strongest)
                target_quadrant = None
                if enemy_eggs_right > enemy_eggs_left + 2:
                    # Enemy dominating right, we should check if we've covered left
                    our_eggs_left = sum(1 for egg in board.eggs_player if egg[0] < self.size // 2)
                    if our_eggs_left < enemy_eggs_right // 2:
                        target_quadrant = "left"
                elif enemy_eggs_left > enemy_eggs_right + 2:
                    our_eggs_right = sum(1 for egg in board.eggs_player if egg[0] >= self.size // 2)
                    if our_eggs_right < enemy_eggs_left // 2:
                        target_quadrant = "right"
                
                if enemy_eggs_bottom > enemy_eggs_top + 2:
                    our_eggs_top = sum(1 for egg in board.eggs_player if egg[1] < self.size // 2)
                    if our_eggs_top < enemy_eggs_bottom // 2:
                        target_quadrant = "top"
                elif enemy_eggs_top > enemy_eggs_bottom + 2:
                    our_eggs_bottom = sum(1 for egg in board.eggs_player if egg[1] >= self.size // 2)
                    if our_eggs_bottom < enemy_eggs_top // 2:
                        target_quadrant = "bottom"
                
                if target_quadrant is not None:
                    # Find unexplored tiles in target quadrant
                    target_tiles = []
                    mid = self.size // 2
                    for x in range(self.size):
                        for y in range(self.size):
                            tile = (x, y)
                            if tile in self.visited_tiles:
                                continue
                            if board.is_cell_blocked(tile):
                                continue
                            
                            in_target = False
                            if target_quadrant == "left" and x < mid:
                                in_target = True
                            elif target_quadrant == "right" and x >= mid:
                                in_target = True
                            elif target_quadrant == "top" and y < mid:
                                in_target = True
                            elif target_quadrant == "bottom" and y >= mid:
                                in_target = True
                            
                            if in_target:
                                target_tiles.append(tile)
                    
                    if target_tiles:
                        # Find closest target tile
                        closest_target = min(target_tiles, key=lambda t: self._manhattan(cur_loc, t))
                        next_loc = loc_after_direction(cur_loc, choice[0])
                        cur_dist = self._manhattan(cur_loc, closest_target)
                        next_dist = self._manhattan(next_loc, closest_target)
                        
                        # If current choice doesn't help us get to underexplored quadrant
                        if next_dist >= cur_dist and choice[1] != MoveType.EGG:
                            best_territory = None
                            best_territory_score = -1e9
                            for mv in legal_moves:
                                terr_loc = loc_after_direction(cur_loc, mv[0])
                                if not board.is_valid_cell(terr_loc) or board.is_cell_blocked(terr_loc):
                                    continue
                                
                                new_dist = self._manhattan(terr_loc, closest_target)
                                progress = cur_dist - new_dist
                                
                                score = 40.0 * progress
                                
                                if terr_loc not in self.visited_tiles:
                                    score += 60.0
                                if mv[1] == MoveType.EGG:
                                    score += 50.0
                                if self._is_my_parity(terr_loc):
                                    score += 15.0
                                
                                score -= 40.0 * self._risk_at(terr_loc)
                                
                                if score > best_territory_score:
                                    best_territory_score = score
                                    best_territory = mv
                            
                            if best_territory is not None and best_territory_score > 25.0:
                                choice = best_territory
        
        # MALDINI: ENDGAME EXPLORATION & CORNER SEEKING
        # In late game, prioritize reaching unexplored regions OR corners
        turns_left = board.turns_left_player
        if turns_left <= 25:  # Late game
            # First check for large unexplored regions we can still reach
            unexplored_target = self._get_largest_unexplored_target(board)
            target_to_seek = None
            target_type = None
            
            # Check corners
            corners = [(0, 0), (0, self.size - 1), (self.size - 1, 0), (self.size - 1, self.size - 1)]
            best_corner = None
            best_corner_dist = 999
            for corner in corners:
                if corner in board.eggs_player or corner in board.eggs_enemy:
                    continue
                if board.is_cell_blocked(corner):
                    continue
                if self._risk_at(corner) > 0.7:
                    continue
                dist = self._manhattan(cur_loc, corner)
                if dist < turns_left and dist < best_corner_dist:
                    best_corner_dist = dist
                    best_corner = corner
            
            # Decide: unexplored region or corner?
            if unexplored_target is not None:
                unexplored_dist = self._manhattan(cur_loc, unexplored_target)
                # Prefer unexplored regions if they're reasonably close
                if unexplored_dist <= turns_left:
                    if best_corner is None or unexplored_dist <= best_corner_dist + 2:
                        target_to_seek = unexplored_target
                        target_type = "unexplored"
            
            if target_to_seek is None and best_corner is not None and best_corner_dist <= 8:
                target_to_seek = best_corner
                target_type = "corner"
            
            if target_to_seek is not None:
                cur_dist = self._manhattan(cur_loc, target_to_seek)
                next_loc = loc_after_direction(cur_loc, choice[0])
                next_dist = self._manhattan(next_loc, target_to_seek)
                
                # If not making progress, find a better move
                if next_dist >= cur_dist and choice[1] != MoveType.EGG:
                    best_move = None
                    best_score = -999
                    for mv in legal_moves:
                        mv_next = loc_after_direction(cur_loc, mv[0])
                        if not board.is_valid_cell(mv_next) or board.is_cell_blocked(mv_next):
                            continue
                        new_dist = self._manhattan(mv_next, target_to_seek)
                        progress = cur_dist - new_dist
                        score = progress * 15  # Progress toward target
                        if mv[1] == MoveType.EGG:
                            score += 8
                        if mv_next not in self.visited_tiles:
                            score += 20  # Bonus for novel tiles
                        score -= self._risk_at(mv_next) * 15
                        if score > best_score:
                            best_score = score
                            best_move = mv
                    
                    if best_move is not None and best_score > 0:
                        choice = best_move
        
        # MID-GAME EXPLORATION: if we haven't laid eggs in a while and are
        # revisiting old tiles, push toward unexplored areas
        if (self.phase != "opening" and 
            self.moves_since_last_egg >= 5 and 
            len(self.recent_positions) >= 10):
            # Check if we're stuck in a small region (5 or fewer unique tiles in last 10)
            recent_10 = list(self.recent_positions)[-10:]
            unique_recent = set(recent_10)
            if len(unique_recent) <= 5:
                # We're stagnating - find a move toward unexplored territory
                next_loc = loc_after_direction(cur_loc, choice[0])
                # Only override if current choice keeps us in the stagnant area
                if next_loc in unique_recent or next_loc in self.visited_tiles:
                    best_explore = None
                    best_score = -1e9
                    for mv in legal_moves:
                        explore_loc = loc_after_direction(cur_loc, mv[0])
                        if not board.is_valid_cell(explore_loc) or board.is_cell_blocked(explore_loc):
                            continue
                        score = 0.0
                        # Big bonus for unvisited tiles
                        if explore_loc not in self.visited_tiles:
                            score += 100.0
                        # Bonus for tiles not in recent stagnation area
                        if explore_loc not in unique_recent:
                            score += 40.0
                        # Bonus for egg moves
                        if mv[1] == MoveType.EGG:
                            score += 60.0
                        # Bonus for parity (can egg next)
                        if self._is_my_parity(explore_loc):
                            score += 20.0
                        # Penalty for risk
                        score -= 50.0 * self._risk_at(explore_loc)
                        if score > best_score:
                            best_score = score
                            best_explore = mv
                    # Only take exploration move if it's significantly better
                    if best_explore is not None and best_score > 20.0:
                        choice = best_explore

        # MALDINI: HARD TURD OVERRIDE - Never turd when behind or in recovery mode
        if choice[1] == MoveType.TURD:
            my_eggs = board.chicken_player.get_eggs_laid()
            opp_eggs = board.chicken_enemy.get_eggs_laid()
            egg_diff = my_eggs - opp_eggs
            turds_left = board.chicken_player.get_turds_left()
            
            # NEVER turd if:
            # 1. We're behind on eggs
            # 2. In trapdoor recovery mode
            # 3. Early game (before turn 25)
            # 4. Endgame (less than 15 turns left)
            # 5. Only 1 turd left (save it)
            should_block_turd = (
                egg_diff < 2 or
                self.trapdoor_recovery_mode or
                board.turn_count < 25 or
                board.turns_left_player < 15 or
                turds_left <= 1
            )
            
            if should_block_turd:
                # Find best non-turd move
                non_turd_moves = [mv for mv in legal_moves if mv[1] != MoveType.TURD]
                if non_turd_moves:
                    # Prefer eggs, then plain moves
                    egg_moves = [mv for mv in non_turd_moves if mv[1] == MoveType.EGG]
                    if egg_moves:
                        choice = egg_moves[0]
                    else:
                        # Pick the best plain move (toward unexplored)
                        best_plain = None
                        best_plain_score = -1e9
                        for mv in non_turd_moves:
                            mv_loc = loc_after_direction(cur_loc, mv[0])
                            if not board.is_valid_cell(mv_loc) or board.is_cell_blocked(mv_loc):
                                continue
                            score = 0.0
                            if mv_loc not in self.visited_tiles:
                                score += 100.0
                            if mv[1] == MoveType.EGG:
                                score += 50.0
                            score -= 30.0 * self._risk_at(mv_loc)
                            if score > best_plain_score:
                                best_plain_score = score
                                best_plain = mv
                        if best_plain is not None:
                            choice = best_plain
        
        # =================================================================
        # MALDINI: PURE ENDGAME EGG HUNTING MODE
        # =================================================================
        # In the endgame, FORGET about the opponent, territory, oscillation, etc.
        # Just find and lay eggs as fast as possible!
        # This overrides ALL other logic when in endgame
        turns_left = board.turns_left_player
        if turns_left <= 20:  # Endgame threshold
            # Step 1: If we can lay an egg RIGHT NOW, do it
            egg_moves = [mv for mv in legal_moves if mv[1] == MoveType.EGG]
            if egg_moves:
                # Pick the best egg move (lowest risk, best position)
                best_egg = None
                best_egg_score = -1e9
                for mv in egg_moves:
                    egg_loc = loc_after_direction(cur_loc, mv[0])
                    if not board.is_valid_cell(egg_loc) or board.is_cell_blocked(egg_loc):
                        continue
                    
                    score = 200.0  # Base score - eggs are EVERYTHING
                    
                    # Bonus for corners (3 points!)
                    if self._is_corner(cur_loc):
                        score += 100.0
                    
                    # Bonus for setting up more eggs
                    if self._is_my_parity(egg_loc):
                        score += 30.0  # Next tile is also eggable
                    
                    # Bonus for unvisited territory (more eggs likely)
                    if egg_loc not in self.visited_tiles:
                        score += 40.0
                    
                    # Risk penalty (but reduced - we need eggs!)
                    score -= 30.0 * self._risk_at(egg_loc)
                    
                    if score > best_egg_score:
                        best_egg_score = score
                        best_egg = mv
                
                if best_egg is not None:
                    choice = best_egg
            else:
                # Step 2: Can't egg now - find the FASTEST path to an egg opportunity
                # Use BFS to find nearest eggable tile
                best_egg_hunt = None
                best_hunt_score = -1e9
                
                for mv in legal_moves:
                    if mv[1] == MoveType.TURD:
                        continue  # Never turd in endgame
                    
                    next_loc = loc_after_direction(cur_loc, mv[0])
                    if not board.is_valid_cell(next_loc) or board.is_cell_blocked(next_loc):
                        continue
                    
                    score = 0.0
                    
                    # How far to nearest egg opportunity from this tile?
                    dist_to_egg = self._nearest_egg_distance(board, next_loc, limit=8)
                    cur_dist_to_egg = self._nearest_egg_distance(board, cur_loc, limit=8)
                    
                    # HUGE bonus for getting closer to egg opportunities
                    egg_progress = cur_dist_to_egg - dist_to_egg
                    score += 50.0 * egg_progress
                    
                    # Can we egg immediately from next_loc?
                    if board.can_lay_egg_at_loc(next_loc):
                        score += 100.0
                    
                    # Bonus for corners (rush to get them!)
                    corners = [(0, 0), (0, self.size - 1), (self.size - 1, 0), (self.size - 1, self.size - 1)]
                    for corner in corners:
                        if corner in board.eggs_player or corner in board.eggs_enemy:
                            continue
                        if board.is_cell_blocked(corner):
                            continue
                        corner_dist_cur = self._manhattan(cur_loc, corner)
                        corner_dist_next = self._manhattan(next_loc, corner)
                        if corner_dist_next < corner_dist_cur:
                            score += 40.0  # Moving toward unclaimed corner
                            if corner_dist_next <= 2:
                                score += 30.0  # Very close to corner!
                    
                    # Bonus for unvisited tiles (more egg opportunities)
                    if next_loc not in self.visited_tiles:
                        score += 35.0
                    
                    # Bonus for parity tiles (can egg there)
                    if self._is_my_parity(next_loc):
                        score += 20.0
                    
                    # Penalty for risk
                    score -= 25.0 * self._risk_at(next_loc)
                    
                    # Penalty for backtracking
                    if next_loc == self.prev_pos:
                        score -= 40.0
                    
                    if score > best_hunt_score:
                        best_hunt_score = score
                        best_egg_hunt = mv
                
                if best_egg_hunt is not None:
                    choice = best_egg_hunt
        
        # MALDINI: STRATEGIC TURD DROPS - CUT OFF OPPONENT EXPANSION
        # Key insight: Turds are FREE when we can't lay an egg anyway!
        # 
        # Strategy:
        # - NO turds in early game (moves 0-9) - focus on expansion
        # - MIDGAME (moves 10-28): Use turds to CUT OFF opponent's expansion path
        # - Place turds between enemy and unexplored territory
        # - Never turd in endgame (need to focus on eggs)
        turds_remaining = 5 - len(board.turds_player)
        our_move_count = board.turn_count // 2
        
        # Check if turd moves are available (requires distance >= 2 from enemy)
        turd_moves = [mv for mv in legal_moves if mv[1] == MoveType.TURD]
        
        # Can we lay an egg at our current position?
        can_egg_here = board.can_lay_egg()
        
        # Turd window: MIDGAME ONLY (moves 10-28)
        # This is when we're actively contesting territory with opponent
        in_turd_window = 10 <= our_move_count <= 28 and board.turns_left_player > 12
        
        center_col = self.size // 2
        enemy_loc = board.chicken_enemy.get_location()
        
        if turd_moves and turds_remaining > 0 and in_turd_window:
            # KEY: If we CAN'T egg here, turd is FREE - always consider it!
            should_consider_turd = (not can_egg_here) or (choice[1] != MoveType.EGG)
            
            if should_consider_turd:
                best_turd = None
                best_turd_score = -1e9
                
                spawn_on_left = self.spawn[0] < center_col
                
                # Calculate enemy's likely expansion direction
                # Enemy wants to go away from their spawn toward center/our side
                enemy_expansion_dir_x = 1 if self.enemy_spawn[0] == 0 else -1
                
                for mv in turd_moves:
                    turd_loc = loc_after_direction(cur_loc, mv[0])
                    if not board.is_valid_cell(turd_loc) or board.is_cell_blocked(turd_loc):
                        continue
                    
                    # Base score - higher if we can't egg (turd is free!)
                    score = 120.0 if not can_egg_here else 40.0
                    
                    dist_from_center = abs(cur_loc[0] - center_col)
                    dist_to_enemy = self._manhattan(cur_loc, enemy_loc)
                    
                    # ===== CUTOFF VALUE - This is the key metric =====
                    # A turd is valuable if it's BETWEEN the enemy and open space
                    
                    # Is this turd blocking enemy's path toward our side?
                    on_enemy_half = (cur_loc[0] >= center_col) if spawn_on_left else (cur_loc[0] < center_col)
                    
                    # HUGE bonus for turds that cut off enemy expansion
                    # Best position: near center, between enemy and unexplored territory
                    if dist_from_center <= 2 and dist_to_enemy <= 5:
                        score += 100.0  # Prime cutoff position!
                    elif dist_from_center <= 3 and dist_to_enemy <= 6:
                        score += 70.0  # Good cutoff position
                    
                    # Bonus for being on enemy's expansion path
                    # If we're ahead of the enemy (closer to their destination)
                    if on_enemy_half:
                        score += 60.0  # We're in their territory
                        
                        # Extra bonus if we're blocking their path to center
                        if dist_to_enemy <= 4:
                            score += 40.0  # Directly blocking them!
                    
                    # Bonus for same row as enemy (horizontal cutoff)
                    if abs(cur_loc[1] - enemy_loc[1]) <= 1:
                        score += 35.0  # Same row - blocks horizontal movement
                    
                    # Bonus for being between enemy and unclaimed corners
                    corners = [(0, 0), (0, self.size - 1), (self.size - 1, 0), (self.size - 1, self.size - 1)]
                    for corner in corners:
                        if corner in board.eggs_player or corner in board.eggs_enemy:
                            continue
                        enemy_to_corner = self._manhattan(enemy_loc, corner)
                        turd_to_corner = self._manhattan(cur_loc, corner)
                        # If we're between enemy and corner
                        if turd_to_corner < enemy_to_corner and enemy_to_corner <= 5:
                            score += 30.0  # Blocking their corner access
                    
                    # Moderate bonus for protecting our egg clusters
                    eggs_nearby = sum(1 for e in board.eggs_player if self._manhattan(cur_loc, e) <= 2)
                    if eggs_nearby >= 2:
                        score += 20.0
                    
                    # Bonus for new territory
                    if turd_loc not in self.visited_tiles:
                        score += 15.0
                    
                    # PENALTY: Don't waste turds far from the action
                    if dist_to_enemy > 8:
                        score -= 50.0  # Too far from enemy to matter
                    
                    # PENALTY: Don't waste turds in our own corner
                    dist_to_our_spawn = self._manhattan(cur_loc, self.spawn)
                    if dist_to_our_spawn <= 3 and dist_from_center > 3:
                        score -= 60.0  # Wasted in our corner
                    
                    # Risk penalty
                    score -= 15.0 * self._risk_at(turd_loc)
                    
                    if score > best_turd_score:
                        best_turd_score = score
                        best_turd = mv
                
                # Turd decision
                if best_turd is not None:
                    if not can_egg_here:
                        # Turd is FREE - do it if it has cutoff value
                        if best_turd_score > 80.0:
                            choice = best_turd
                    elif best_turd_score > 180.0 and choice[1] != MoveType.EGG:
                        # Only override for really high-value cutoff positions
                        choice = best_turd
        
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
        # Compute current local territory once per call (for mode selection and deltas)
        territory_cur_now = self._territory_diff_bfs(board, cur, radius=5)
        mode = self._select_mode(board, territory_cur_now)
        nearest_egg_cur = 1
        if self.phase != "opening":
            nearest_egg_cur = self._nearest_egg_distance(board, cur, limit=6)
        eggs_self = board.chicken_player.get_eggs_laid()
        eggs_opp = board.chicken_enemy.get_eggs_laid()
        egg_diff = eggs_self - eggs_opp
        dist_center = self._distance_to_center(next_loc)
        # No special blotch repulsion here; handled via risk grid

        if mt == MoveType.EGG:
            base = 60.0
            
            # MALDINI: ENDGAME EGG PRIORITY - in endgame, eggs are EVERYTHING
            if self.phase == "endgame":
                base = 100.0  # Much higher base score for eggs in endgame
                # Extra bonus based on turns remaining - more urgent as game ends
                if board.turns_left_player <= 10:
                    base += 40.0  # Critical endgame - MUST lay eggs
                elif board.turns_left_player <= 20:
                    base += 20.0  # Late endgame
            
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
            # Favor parity-correct eggs and novelty of current tile
            if self._is_my_parity(cur):
                base += 6.0
            base += 2.5 * self._novelty_bonus(cur)
            
            # MALDINI: Early game chain bonus - reward eggs that set up more eggs
            if self.phase == "opening" or board.turn_count <= 25:
                # Bonus for eggs adjacent to our existing eggs (building chains)
                adjacent_eggs = self._count_adjacent_eggs(board, cur)
                if adjacent_eggs > 0:
                    base += 12.0 * adjacent_eggs  # Strong bonus for extending chains
                
                # Bonus for eggs that lead to more egg opportunities
                chain_potential = self._egg_chain_score(board, next_loc, max_depth=3)
                base += 15.0 * chain_potential
                
                # Corner bonus in early game
                if self._is_corner(cur):
                    base += 20.0  # Extra corner bonus early
            
            # Opening throttling: REDUCED - we want more eggs early now
            if self.phase == "opening":
                base -= 15.0  # Reduced from 32.0
                # Only penalize back-to-back eggs, not spacing
                if self.moves_since_last_egg < 1:
                    base -= 10.0  # Reduced penalty
                # Only slight penalty if novel moves exist
                if self._safe_novel_exists and chain < 1.0:  # Only if chain is weak
                    base -= 8.0  # Reduced from 18.0
            # Mode adjustments: simplify priorities
            if mode == "expand":
                base -= 20.0
            elif mode == "egg":
                base += 12.0
            return base

        if mt == MoveType.TURD:
            # MALDINI: SIMPLIFIED TURD SCORING
            # The forced turd logic in play() handles strategic placement
            # This scoring is just a fallback for when turds compete normally
            base = 20.0
            
            # Basic bonus if we can't egg here (turd is "free")
            if not board.can_lay_egg():
                base += 30.0
            
            # Risk penalty
            base -= 15.0 * self._risk_at(cur)
            
            return base

        lane_progress = self._lane_progress(cur, next_loc)
        future_egg = self._future_egg_turns(board, next_loc)
        choke = self._enemy_choke_bonus(board, next_loc)
        path_pull = self._path_progress(board, next_loc)
        parity_bonus = 12.0 if self._is_my_parity(next_loc) else 0.0
        # Outward expansion from spawn
        sx, sy = self.spawn
        cur_out = abs(cur[0] - sx) + abs(cur[1] - sy)
        nxt_out = abs(next_loc[0] - sx) + abs(next_loc[1] - sy)
        outward_gain_multiplier = 1.9 if self.phase == "opening" else 1.0
        outward_gain = self.outward_weight * outward_gain_multiplier * (nxt_out - cur_out)
        # Exploration incentives
        novelty = self._novelty_bonus(next_loc)
        frontier_step = self._frontier_step_bonus(cur, next_loc)
        coverage = self._coverage_penalty(next_loc)
        branching = self._branching_factor(board, next_loc)
        # MALDINI: Direct bonus for unvisited tiles
        unvisited_bonus = 8.0 if next_loc not in self.visited_tiles else 0.0
        if self.trapdoor_recovery_mode and next_loc not in self.visited_tiles:
            unvisited_bonus = 15.0  # Extra strong during recovery
        # MALDINI: Opponent pressure - reward moves that limit enemy options
        opponent_pressure = self._opponent_pressure_score(board, next_loc)
        
        # MALDINI: Global cutoff drive - TRACK AND INTERCEPT the opponent
        # In midgame, we MUST be aware of opponent and contest territory
        cutoff_drive = 0.0
        if self.phase == "midgame":  # Only in midgame, not opening or endgame
            enemy_loc = board.chicken_enemy.get_location()
            dist_to_enemy = self._manhattan(next_loc, enemy_loc)
            curr_dist_to_enemy = self._manhattan(cur, enemy_loc)
            
            # Always reward closing distance in midgame (unless very risky)
            if risk_next < 0.7:
                if dist_to_enemy < curr_dist_to_enemy:
                    # Stronger bonus when enemy is far (we need to catch up)
                    if curr_dist_to_enemy > 5:
                        cutoff_drive = 8.0  # Strong chase when far
                    elif curr_dist_to_enemy > 3:
                        cutoff_drive = 5.0  # Medium chase
                    else:
                        cutoff_drive = 3.0  # Maintain pressure when close
            
            # Bonus for moving towards center - center control is key
            nxt_center_d = self._distance_to_center(next_loc)
            cur_center_d = self._distance_to_center(cur)
            if nxt_center_d < cur_center_d:
                cutoff_drive += 3.0  # Center control bonus
            
            # Extra bonus if enemy is dominating center and we're reclaiming
            enemy_center_dist = self._distance_to_center(enemy_loc)
            if enemy_center_dist < 2.5 and nxt_center_d < cur_center_d:
                cutoff_drive += 4.0  # Contest their center control
            
            # Bonus for moves that put us between enemy and unclaimed corners
            corners = [(0, 0), (0, self.size - 1), (self.size - 1, 0), (self.size - 1, self.size - 1)]
            for corner in corners:
                if corner in board.eggs_player or corner in board.eggs_enemy:
                    continue
                enemy_to_corner = self._manhattan(enemy_loc, corner)
                our_to_corner = self._manhattan(next_loc, corner)
                cur_to_corner = self._manhattan(cur, corner)
                # If we're getting between enemy and an unclaimed corner
                if our_to_corner < enemy_to_corner and our_to_corner < cur_to_corner:
                    cutoff_drive += 4.0  # Intercepting corner access

        # Enemy turd proximity (radius-2)
        enemy_turds_near = self._count_enemy_turds_within(board, next_loc, radius=2)
        # Local open space (radius-2)
        open_space = self._local_open_space(board, next_loc)
        # Lightweight cut-in trigger: move toward interior when we can egg soon
        next_is_eggable = board.can_lay_egg_at_loc(next_loc)
        egg_in_two = self._egg_in_two_steps(board, next_loc)
        nearest_egg_steps = 1
        nearest_egg_cur = 1
        if self.phase != "opening":
            nearest_egg_steps = self._nearest_egg_distance(board, next_loc, limit=6)
            nearest_egg_cur = self._nearest_egg_distance(board, cur, limit=6)
        # Egg opportunity seeking: in mid/endgame, reward moving closer to egg tiles
        # MALDINI: Much stronger in endgame - we NEED to find and lay eggs
        egg_seek_bonus = 0.0
        if self.phase != "opening" and self.moves_since_last_egg >= 2:
            # Reward getting closer to eggable tiles
            egg_improvement = nearest_egg_cur - nearest_egg_steps
            if egg_improvement > 0:
                # Much stronger bonus in endgame
                multiplier = 8.0 if self.phase == "endgame" else 3.0
                egg_seek_bonus = multiplier * egg_improvement
            # Extra bonus if we're far from any egg opportunity and moving toward one
            if nearest_egg_cur >= 4 and egg_improvement > 0:
                egg_seek_bonus += 10.0 if self.phase == "endgame" else 4.0
            # Bonus for moving to an immediately eggable tile
            if next_is_eggable and nearest_egg_cur >= 2:
                egg_seek_bonus += 15.0 if self.phase == "endgame" else 5.0
        # Edge escape: encourage moving away from borders when cramped
        cur_edge = self._distance_to_border(cur)
        nxt_edge = self._distance_to_border(next_loc)
        edge_escape = max(0, nxt_edge - cur_edge)
        cur_center_d = self._distance_to_center(cur)
        nxt_center_d = self._distance_to_center(next_loc)
        cut_in_bonus = 0.0
        if nxt_center_d < cur_center_d and (next_is_eggable or egg_in_two):
            cut_in_bonus = 2.3 + (0.7 if (egg_in_two and not next_is_eggable) else 0.0)
        # Removed heavy global region and immediate mobility shaping to restore prior strong behavior
        # Opening diagonal sweep: aim toward best corner away from enemy to sweep and flank
        diag_bonus = 0.0
        sep_bonus_raw = 0.0
        vor_bonus_raw = 0.0
        if self.phase == "opening":
            corner = self._best_diagonal_corner(board)
            cur_d_corner = self._manhattan(cur, corner)
            nxt_d_corner = self._manhattan(next_loc, corner)
            diag_bonus = 3.0 * float(cur_d_corner - nxt_d_corner)
            enemy_loc = board.chicken_enemy.get_location()
            sep_now = self._manhattan(cur, enemy_loc)
            sep_next = self._manhattan(next_loc, enemy_loc)
            sep_bonus_raw = 0.6 * float(sep_next - sep_now)
            vor_bonus_raw = 0.2 * float(self._local_voronoi_advantage(board, next_loc, radius=5))
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
        territory_contrib = risk_scale * (territory_weight * t_adv + delta_weight * max(0.0, t_delta))
        territory_contrib = self._clamp(territory_contrib, -6.0, 6.0)
        # Harmonize: if BFS territory is already strong, damp local Voronoi bonus to avoid double counting
        vor_bonus = 0.0 if abs(t_adv) >= 4.0 else 0.5 * vor_bonus_raw
        
        # MALDINI: MIDGAME AGGRESSION - DOMINATE THE BOARD
        # This is where we WIN or LOSE the game. Be AGGRESSIVE.
        midgame_aggression = 0.0
        if self.phase == "midgame":
            center_col = self.size // 2
            spawn_on_left = self.spawn[0] < center_col
            enemy_loc = board.chicken_enemy.get_location()
            enemy_spawn = board.chicken_enemy.get_spawn()
            dist_to_enemy = self._manhattan(next_loc, enemy_loc)
            cur_dist_to_enemy = self._manhattan(cur, enemy_loc)
            
            # Which half of the board
            on_enemy_half = (next_loc[0] >= center_col) if spawn_on_left else (next_loc[0] < center_col)
            on_our_half = not on_enemy_half
            
            # Predict enemy movement direction
            enemy_heading = self._predict_enemy_direction()
            
            # === TRACK THE OPPONENT ===
            # In midgame, we MUST know where the opponent is and respond
            # If they're expanding into open space, we need to cut them off
            
            # Strong reward for closing distance when opponent is far
            if cur_dist_to_enemy > 4:
                # Opponent is far - we need to get closer to contest
                if dist_to_enemy < cur_dist_to_enemy:
                    midgame_aggression += 20.0  # Strong chase bonus
            elif cur_dist_to_enemy > 2:
                # Medium distance - pressure them
                if dist_to_enemy < cur_dist_to_enemy:
                    midgame_aggression += 15.0
                    if dist_to_enemy <= 3:
                        midgame_aggression += 10.0  # Getting close
            else:
                # Very close - maintain pressure but don't collide
                if dist_to_enemy >= 2:
                    midgame_aggression += 8.0  # Safe distance to keep pressure
            
            # === INTERCEPT OPPONENT'S PATH ===
            # If we know where opponent is heading, get in front of them
            if enemy_heading is not None:
                # Calculate where enemy is likely going
                enemy_target = loc_after_direction(enemy_loc, enemy_heading)
                if self._in_bounds(enemy_target):
                    # Reward moves that get us closer to their destination
                    our_dist_to_target = self._manhattan(next_loc, enemy_target)
                    enemy_dist_to_target = self._manhattan(enemy_loc, enemy_target)
                    if our_dist_to_target < enemy_dist_to_target:
                        midgame_aggression += 25.0  # We can intercept!
                    elif our_dist_to_target == enemy_dist_to_target:
                        midgame_aggression += 15.0  # Race condition
            
            # === TERRITORY INVASION ===
            # HUGE reward for being on enemy's half
            if on_enemy_half:
                midgame_aggression += 25.0  # Base invasion bonus
                
                # Extra for unvisited tiles on enemy side
                if next_loc not in self.visited_tiles:
                    midgame_aggression += 20.0  # Claiming new enemy territory!
                
                # Bonus for being deeper into enemy territory
                if spawn_on_left:
                    depth = next_loc[0] - center_col
                else:
                    depth = center_col - next_loc[0]
                midgame_aggression += 4.0 * max(0, depth)  # Deeper = better
            
            # Reward moving toward enemy spawn (ultimate domination)
            cur_spawn_dist = self._manhattan(cur, enemy_spawn)
            next_spawn_dist = self._manhattan(next_loc, enemy_spawn)
            if next_spawn_dist < cur_spawn_dist:
                midgame_aggression += 15.0
            
            # === SPACE DOMINATION ===
            # Count unexplored tiles on enemy side reachable from this position
            enemy_side_unexplored = 0
            our_side_unexplored = 0
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if abs(dx) + abs(dy) > 4:
                        continue
                    tile = (next_loc[0] + dx, next_loc[1] + dy)
                    if not self._in_bounds(tile):
                        continue
                    if tile in self.visited_tiles or board.is_cell_blocked(tile):
                        continue
                    tile_on_enemy_side = (tile[0] >= center_col) if spawn_on_left else (tile[0] < center_col)
                    if tile_on_enemy_side:
                        enemy_side_unexplored += 1
                    else:
                        our_side_unexplored += 1
            
            # Strong preference for moves that access enemy territory
            midgame_aggression += 3.0 * enemy_side_unexplored
            # But also don't abandon our side completely
            midgame_aggression += 1.0 * our_side_unexplored
            
            # === CUTTING OFF OPPONENT ===
            # Reward positioning that blocks enemy from unexplored territory
            if dist_to_enemy <= 6:
                blocking_value = 0
                high_value_blocks = 0
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if abs(dx) + abs(dy) > 3:
                            continue
                        tile = (next_loc[0] + dx, next_loc[1] + dy)
                        if not self._in_bounds(tile):
                            continue
                        if tile in self.visited_tiles or board.is_cell_blocked(tile):
                            continue
                        our_dist = self._manhattan(next_loc, tile)
                        enemy_dist_to_tile = self._manhattan(enemy_loc, tile)
                        if our_dist < enemy_dist_to_tile:
                            blocking_value += 1
                            # High value if blocking parity tiles or corners
                            if self._is_my_parity(tile) or self._is_corner(tile):
                                high_value_blocks += 1
                midgame_aggression += 3.0 * blocking_value
                midgame_aggression += 5.0 * high_value_blocks
            
            # === RESPOND TO OPPONENT EXPANSION ===
            # If opponent has more eggs than us in midgame, be MORE aggressive
            # But if we're way behind, we need to focus on eggs not just chasing
            egg_diff = board.chicken_player.get_eggs_laid() - board.chicken_enemy.get_eggs_laid()
            if egg_diff < -3:
                # We're way behind - need to balance aggression with egg laying
                midgame_aggression *= 0.8  # Reduce aggression, focus on eggs
            elif egg_diff < 0:
                # We're slightly behind - be more aggressive
                midgame_aggression *= 1.15
            elif egg_diff > 3:
                # We're ahead - can be slightly less aggressive
                midgame_aggression *= 0.85
            
            # === TERRITORY GAINS ===
            if territory_delta > 0:
                midgame_aggression += 8.0 * territory_delta
                if on_enemy_half:
                    midgame_aggression += 5.0 * territory_delta  # Extra for enemy territory
        
        escape_bonus = 0.0
        escape_pressure = 0.0
        escape_egg_delta = 0.0
        # Detect oscillation early to boost escape
        oscillating = False
        if len(self.recent_positions) >= 4:
            recent_unique = set(list(self.recent_positions)[-6:])
            oscillating = len(recent_unique) <= 2
        if self.phase != "opening":
            stagnating = self.stagnation_count >= 3
            egg_starved = (self.moves_since_last_egg >= 3) and not board.can_lay_egg()
            egg_far = nearest_egg_steps > 3
            cramped = open_space < 0.55 and branching <= 1
            # Detect saturated area: no eggs nearby, need to explore
            saturated_area = nearest_egg_cur >= 4 and not board.can_lay_egg()
            if stagnating and nearest_egg_cur > 2:
                escape_pressure += 0.6
            if egg_starved:
                escape_pressure += 0.6
            if egg_far:
                escape_pressure += 0.5
            if cramped:
                escape_pressure += 0.4
            if oscillating:
                escape_pressure += 1.0  # Strong push to escape oscillation
            if saturated_area:
                escape_pressure += 0.8  # Push to find new egg areas
            escape_pressure = min(2.0, escape_pressure)
            if escape_pressure > 0.0:
                inward = 1.0 if nxt_center_d < cur_center_d else 0.0
                away_from_edge = max(0.0, float(nxt_edge - cur_edge))
                egg_delta = max(0.0, float(nearest_egg_cur - nearest_egg_steps))
                escape_bonus += escape_pressure * (
                    3.0 * frontier_step + 1.8 * novelty + 1.2 * away_from_edge + 1.5 * inward
                )
                if egg_delta > 0.0:
                    escape_bonus += 2.4 * escape_pressure * egg_delta
                    escape_egg_delta = egg_delta
                if coverage >= 1.0 and frontier_step == 0.0 and novelty < 0.5:
                    escape_bonus -= 1.6 * coverage * escape_pressure
                if coverage >= 2.0:
                    escape_bonus -= 1.0 * escape_pressure
                # When oscillating, strongly prefer any tile that breaks out
                if oscillating:
                    if novelty > 0.5:
                        escape_bonus += 15.0
                    # Even non-novel tiles that aren't in the oscillation set are good
                    osc_tiles = set(list(self.recent_positions)[-8:])
                    if next_loc not in osc_tiles:
                        escape_bonus += 20.0
                    # Reward moving toward center/open areas when stuck on edge
                    if cur_edge <= 1 and nxt_edge > cur_edge:
                        escape_bonus += 8.0

        base = 32.0 + parity_bonus
        base += 5.0 * lane_progress
        # Stronger choke to encourage cutoffs and separation
        base += 15.0 * choke
        # MALDINI: Extra bonus for cutting off enemy from open space
        cutoff_bonus = self._enemy_cutoff_bonus(board, next_loc)
        base += 2.0 * cutoff_bonus
        base += cutoff_drive # Added cutoff drive
        base += path_pull
        base += outward_gain
        base += novelty + 2.2
        # Stuck pressure increases frontier drive
        stuck_pressure = min(1.0, float(self.stagnation_count) / 5.0)
        frontier_wt = (7.2 if self.phase == "opening" else 6.2) + 2.0 * stuck_pressure
        # MALDINI: Extra frontier pressure in midgame to dominate space
        if self.phase == "midgame":
            frontier_wt += 3.0
        if mode == "expand":
            frontier_wt += 2.0
        elif mode == "egg":
            frontier_wt += 0.8
        base += frontier_wt * frontier_step
        base += 2.0 * open_space
        if mode == "expand":
            base += 0.8 * novelty + 0.8 * open_space
        base += cut_in_bonus
        base += diag_bonus + sep_bonus + vor_bonus
        base += escape_bonus
        base += territory_contrib
        base += egg_seek_bonus
        # MALDINI: Add midgame aggression bonus
        base += midgame_aggression
        # MALDINI: Direct unvisited tile bonus
        base += unvisited_bonus
        # Collision avoidance: penalize stepping into predicted enemy next tile
        if enemy_forecast is not None and next_loc == enemy_forecast:
            # Allow if it meaningfully improves territory under low risk
            if not (territory_delta > 2.0 and risk_next < 0.6):
                base -= 10.0
        base -= 18.0 * risk_next
        # Extra penalty for proximity to enemy turds in opening (mobility preservation)
        if self.phase == "opening" and enemy_turds_near > 0:
            base -= min(8.0, 2.5 * enemy_turds_near) * (1.0 if branching <= 2 else 0.6)
        base -= 2.5 * dist_center
        # Edge escape encouragement
        if self.phase != "endgame":
            base += 1.5 * float(edge_escape)
        # Strong edge-stuck escape: if we've been on border and oscillating, push hard inward
        if oscillating and cur_edge <= 1:
            if nxt_edge > cur_edge:
                base += 15.0  # Strong reward for moving away from edge
            elif nxt_edge <= cur_edge:
                base -= 8.0  # Penalty for staying on edge while oscillating
        # Delay plain moves that don't lead to near-term egg in the opening
        if self.phase == "opening":
            base -= 1.5 * future_egg
        else:
            base -= 1.8 * future_egg
        
        # MALDINI: Opponent pressure bonus - STRONG in mid-game but BALANCED
        # Mid-game: Track and pressure opponent, but don't forget eggs!
        # Endgame: forget opponent, just spam eggs in our territory
        if self.phase == "opening":
            base += 1.5 * opponent_pressure  # Light pressure in opening
        elif self.phase == "midgame":
            base += 7.0 * opponent_pressure  # Strong pressure in mid-game
        else:
            base += 0.5 * opponent_pressure  # MINIMAL in endgame - focus on eggs, not opponent
        
        # Mode-specific shaping
        # MALDINI: In endgame, cutoff mode should be much weaker - we want eggs, not cutoffs
        if mode == "cutoff":
            if self.phase == "endgame":
                base += 0.5 * choke + 0.3 * path_pull  # Minimal cutoff bonus in endgame
            else:
                base += 3.0 * choke + 1.5 * path_pull + 2.0 * opponent_pressure
        elif mode == "egg":
            base += 0.8 * future_egg  # soften anti-egg penalty when we want to egg
            if self.phase == "endgame":
                base += 5.0  # Extra bonus for being in egg mode during endgame
        # Corridor/dead-end awareness: discourage moving into dead-ends early unless it cuts off space or enables quick egg
        if self.phase == "opening" and branching <= 1:
            if not (territory_delta > 0.0 or egg_in_two):
                base -= 6.0
        # Corner approach planning: encourage stepping into a safe, novel corner we can soon egg
        if (
            mt == MoveType.PLAIN
            and self.phase == "opening"
            and self._is_corner(next_loc)
            and risk_next < 0.75
        ):
            if next_loc not in board.eggs_player and next_loc not in board.turds_player and next_loc not in board.turds_enemy:
                corner_plan = 1.5 + 1.0 * self._novelty_bonus(next_loc)
                if egg_in_two:
                    corner_plan += 1.2
                base += corner_plan
        # Discourage revisits, especially if a safe novel option exists; stronger when stagnating
        # MALDINI: Increased penalties for revisiting - we need to explore!
        base -= 2.5 * coverage * (1.0 + 0.5 * stuck_pressure)
        if self._safe_novel_exists and coverage >= 2.0:
            base -= 8.0  # Increased from 5.0
        if self.phase != "opening" and self.stagnation_count >= 4 and coverage >= 2.0:
            base -= 4.0 + 0.8 * float(self.stagnation_count - 3)  # Increased
        # MALDINI: Extra penalty for revisiting in trapdoor recovery mode
        if self.trapdoor_recovery_mode and next_loc in self.visited_tiles:
            base -= 15.0  # Strong penalty - we MUST explore new areas after reset
        if escape_pressure > 0.0 and frontier_step == 0.0 and novelty < 0.5 and escape_egg_delta <= 0.0:
            base -= 1.8 * escape_pressure
        # Avoid immediate backtracks to the previous position
        if next_loc == self.prev_pos:
            base -= 16.0
        # Cycle avoidance: penalize re-entering short cycles (A-B-A, A-B-C-A) more when stagnating
        cycle_pen = self._cycle_penalty(next_loc)
        base -= min(12.0, cycle_pen * (1.0 + 0.5 * stuck_pressure))
        # Strong oscillation penalty: detect A↔B bouncing and break it
        osc_pen = self._oscillation_penalty(cur, next_loc)
        base -= osc_pen
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
        """
        Main evaluation function.

        Core idea:
        - Primary: who will win the egg race if both sides play reasonably?
          => current eggs + Voronoi "future eggs" over the remaining turns.
        - Secondary: mobility / blocking so we don't walk into dead-ends.
        - Small bonuses/penalties for risk / choke to keep good habits.
        """
        # --- Basic material: eggs laid so far ---
        eggs_self = board.chicken_player.get_eggs_laid()
        eggs_opp = board.chicken_enemy.get_eggs_laid()
        egg_diff = eggs_self - eggs_opp

        # --- Future eggs via Voronoi territory estimation ---
        future_self, future_opp = self._voronoi_future_eggs(board)
        future_diff = future_self - future_opp

        # --- Mobility / terminal-ish situations ---
        mobility_self = len(board.get_valid_moves())
        mobility_opp = len(board.get_valid_moves(enemy=True))
        mobility_diff = mobility_self - mobility_opp

        my_loc = board.chicken_player.get_location()
        risk_here = self._risk_at(my_loc)
        choke = self._enemy_choke_bonus(board, my_loc)

        turns_left = board.turns_left_player

        # --- Base scoring: eggs now + future eggs are king ---
        # Future eggs should matter almost as much as current eggs.
        score = 120.0 * float(egg_diff)
        score += 10.0 * float(future_diff)

        # --- Mobility: don't let yourself get boxed in ---
        score += 4.0 * float(mobility_diff)

        # Hard punish true zero-mobility (usually game-ending)
        if mobility_self == 0:
            score -= 2000.0
        if mobility_opp == 0:
            score += 2000.0

        # --- Light positional seasoning ---
        score -= 40.0 * risk_here      # avoid standing on obvious traps
        score += 15.0 * choke          # reward creating choke points

        # --- Phase awareness ---
        # Near the very end, crank current eggs even harder and
        # slightly de-emphasize future territory (not much time left).
        if turns_left <= 12:
            # Extra egg urgency: from 1x at 12 turns left to 4x near 0.
            urgency = max(1, 4 - (turns_left // 4))  # rough 1..4
            score += 40.0 * urgency * float(egg_diff)

            # In the last few turns, it's mostly about immediate eggs
            # and not about long-range reachable territory.
            if turns_left <= 4:
                score += 2.0 * float(mobility_diff)
        else:
            # Small bias to turning advantages into wins earlier
            score -= 0.1 * float(turns_left)

        return score
    
    def _voronoi_future_eggs(self, board: Board) -> Tuple[int, int]:
        """
        Estimate future egg potential for both players using a simple Voronoi
        territory approach and the remaining number of turns.

        We:
          - Run a multi-source BFS from both chickens.
          - Assign each non-blocked cell to whichever chicken reaches it sooner.
          - Ignore cells that are exactly tied (contested).
          - Only count cells that can realistically be reached and laid on
            within the remaining turns.
        Returns:
            (future_self, future_opp)
        """
        from collections import deque

        start_self = board.chicken_player.get_location()
        start_opp = board.chicken_enemy.get_location()

        # If somehow we're on an invalid tile, bail gracefully.
        if not self._in_bounds(start_self) or not self._in_bounds(start_opp):
            return (0, 0)

        owner: Dict[Tuple[int, int], int] = {}
        dist: Dict[Tuple[int, int], int] = {}

        q: deque[Tuple[Tuple[int, int], int, int]] = deque()

        # Initialize BFS with both players
        if not board.is_cell_blocked(start_self):
            owner[start_self] = 0
            dist[start_self] = 0
            q.append((start_self, 0, 0))
        if not board.is_cell_blocked(start_opp):
            if start_opp in owner:
                owner[start_opp] = -1
            else:
                owner[start_opp] = 1
                dist[start_opp] = 0
            q.append((start_opp, 1, 0))

        # Multi-source BFS
        while q:
            loc, pid, d = q.popleft()
            for direction in Direction:
                nxt = loc_after_direction(loc, direction)
                if not self._in_bounds(nxt):
                    continue
                if board.is_cell_blocked(nxt):
                    continue

                if nxt not in dist:
                    dist[nxt] = d + 1
                    owner[nxt] = pid
                    q.append((nxt, pid, d + 1))
                else:
                    if dist[nxt] == d + 1 and owner[nxt] != pid:
                        owner[nxt] = -1  # contested

        turns_left = board.turns_left_player
        future = [0, 0]  # [self, opp]

        for cell, pid in owner.items():
            if pid not in (0, 1):
                continue

            d = dist[cell]

            # need d moves to get there + ~1 to lay
            if d + 1 > turns_left:
                continue

            future[pid] += 1

        return future[0], future[1]

    # ------------------------------------------------------------------ #
    # Opening guidance
    # ------------------------------------------------------------------ #

    def _opening_move(
        self, board: Board, legal_moves: Sequence[Tuple[Direction, MoveType]]
    ) -> Optional[Tuple[Direction, MoveType]]:
        """
        MALDINI Aggressive Opening Strategy:
        1. FAST vertical expansion - maximize distance covered
        2. Lay eggs opportunistically but NEVER slow down expansion
        3. Watch opponent - if they're expanding faster, we need to match
        4. NEVER step backward toward spawn - only forward or sideways
        """
        if self.phase != "opening":
            return None
        
        cur = board.chicken_player.get_location()
        enemy_loc = board.chicken_enemy.get_location()
        
        # Calculate expansion metrics
        our_dist_from_spawn = self._manhattan(cur, self.spawn)
        enemy_dist_from_spawn = self._manhattan(enemy_loc, self.enemy_spawn)
        
        # Are we falling behind in expansion?
        expansion_deficit = enemy_dist_from_spawn - our_dist_from_spawn
        need_to_catch_up = expansion_deficit >= 2
        
        # Check if we've reached the vertical edge
        at_top = cur[1] == 0
        at_bottom = cur[1] == self.size - 1
        at_vertical_edge = (self._sweep_vertical == Direction.UP and at_top) or \
                          (self._sweep_vertical == Direction.DOWN and at_bottom)
        
        # Define what "backward" means - toward our spawn
        center_col = self.size // 2
        backward_horizontal = Direction.LEFT if self.spawn[0] == 0 else Direction.RIGHT
        backward_vertical = Direction.DOWN if self.spawn[1] == 0 else Direction.UP
        
        # Helper to check if a direction is "backward" (toward spawn)
        def is_retreat(direction: Direction) -> bool:
            if direction == backward_horizontal and abs(cur[0] - self.spawn[0]) <= 2:
                return True
            if direction == backward_vertical and abs(cur[1] - self.spawn[1]) <= 2:
                return True
            return False
        
        # Phase 1: Vertical sweep - GO FAST
        if self._perimeter_phase == "vertical":
            # Switch to cut-in if we hit edge OR we've done enough vertical
            if at_vertical_edge:
                self._perimeter_phase = "cut_in"
                self._cut_in_turn = board.turn_count
            else:
                # PRIORITY 1: Keep moving vertically - speed is key!
                desired_dir = self._sweep_vertical
                candidates = [mv for mv in legal_moves if mv[0] == desired_dir]
                
                if candidates:
                    # If we're behind on expansion, ALWAYS move (don't stop to egg)
                    if need_to_catch_up:
                        plain_moves = [mv for mv in candidates if mv[1] == MoveType.PLAIN]
                        if plain_moves:
                            return plain_moves[0]
                    
                    # Otherwise, egg if we can (but prefer plain to keep moving)
                    if board.can_lay_egg() and self._risk_at(cur) < 0.7:
                        egg_moves = [mv for mv in candidates if mv[1] == MoveType.EGG]
                        if egg_moves:
                            return egg_moves[0]
                    
                    plain_moves = [mv for mv in candidates if mv[1] == MoveType.PLAIN]
                    if plain_moves:
                        return plain_moves[0]
                
                # Can't go vertical - try horizontal toward center (never backward!)
                horiz_dir = self._sweep_horizontal
                horiz_candidates = [mv for mv in legal_moves if mv[0] == horiz_dir]
                if horiz_candidates:
                    plain_moves = [mv for mv in horiz_candidates if mv[1] == MoveType.PLAIN]
                    if plain_moves:
                        # Still in vertical phase, just stepping around obstacle
                        return plain_moves[0]
                
                # If blocked everywhere forward, switch to cut-in
                self._perimeter_phase = "cut_in"
                self._cut_in_turn = board.turn_count
        
        # Phase 2: Cut inward toward center/enemy territory
        if self._perimeter_phase == "cut_in":
            # Primary: horizontal toward center
            desired_dir = self._sweep_horizontal
            candidates = [mv for mv in legal_moves if mv[0] == desired_dir]
            
            if candidates:
                # If behind, prioritize movement over eggs
                if need_to_catch_up:
                    plain_moves = [mv for mv in candidates if mv[1] == MoveType.PLAIN]
                    if plain_moves:
                        return plain_moves[0]
                
                if board.can_lay_egg() and self._risk_at(cur) < 0.7:
                    egg_moves = [mv for mv in candidates if mv[1] == MoveType.EGG]
                    if egg_moves:
                        return egg_moves[0]
                
                plain_moves = [mv for mv in candidates if mv[1] == MoveType.PLAIN]
                if plain_moves:
                    return plain_moves[0]
            
            # Secondary: continue vertical or opposite vertical (but NEVER retreat!)
            for alt_dir in [self._sweep_vertical, self._opposite_dir(self._sweep_vertical)]:
                if is_retreat(alt_dir):
                    continue  # NEVER retreat toward spawn
                
                alt_candidates = [mv for mv in legal_moves if mv[0] == alt_dir]
                if alt_candidates:
                    if need_to_catch_up:
                        plain_moves = [mv for mv in alt_candidates if mv[1] == MoveType.PLAIN]
                        if plain_moves:
                            return plain_moves[0]
                    
                    if board.can_lay_egg() and self._risk_at(cur) < 0.7:
                        egg_moves = [mv for mv in alt_candidates if mv[1] == MoveType.EGG]
                        if egg_moves:
                            return egg_moves[0]
                    
                    plain_moves = [mv for mv in alt_candidates if mv[1] == MoveType.PLAIN]
                    if plain_moves:
                        return plain_moves[0]
            
            # End opening phase after enough cut-in moves
            if board.turn_count >= self._cut_in_turn + 6:
                self._perimeter_phase = "done"
        
        # FALLBACK: If we get here, pick the best non-retreating move
        best_fallback = None
        best_fallback_score = -1e9
        
        for mv in legal_moves:
            if is_retreat(mv[0]):
                continue
            
            next_loc = loc_after_direction(cur, mv[0])
            if not board.is_valid_cell(next_loc) or board.is_cell_blocked(next_loc):
                continue
            
            score = 0.0
            # Distance from spawn (more = better)
            score += 20.0 * self._manhattan(next_loc, self.spawn)
            # Unvisited bonus
            if next_loc not in self.visited_tiles:
                score += 50.0
            # Egg bonus
            if mv[1] == MoveType.EGG:
                score += 30.0
            # Risk penalty
            score -= 30.0 * self._risk_at(next_loc)
            
            if score > best_fallback_score:
                best_fallback_score = score
                best_fallback = mv
        
        return best_fallback
    
    def _opposite_dir(self, d: Direction) -> Direction:
        """Return the opposite direction."""
        if d == Direction.UP:
            return Direction.DOWN
        elif d == Direction.DOWN:
            return Direction.UP
        elif d == Direction.LEFT:
            return Direction.RIGHT
        else:
            return Direction.LEFT

    def _build_opening_script(self) -> List[Direction]:
        # Legacy script - not used anymore but kept for compatibility
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
        # Refresh if no target, already reached, stale, or we're stuck mid/endgame
        stuck_refresh = (
            self.phase != "opening"
            and (
                self.stagnation_count >= 3
                or self.moves_since_last_egg >= 4
            )
        )
        if (
            self.frontier_target is None
            or self.frontier_target in self.visited_tiles
            or board.turn_count >= self.frontier_refresh_turn
            or stuck_refresh
        ):
            self.frontier_target = self._compute_frontier_target(board)
            self.frontier_refresh_turn = board.turn_count + (4 if stuck_refresh else 6)

    def _compute_frontier_target(self, board: Board) -> Optional[Tuple[int, int]]:
        start = board.chicken_player.get_location()
        best_target: Optional[Tuple[int, int]] = None
        best_score = -1e9
        seen = set([start])
        q: deque[Tuple[Tuple[int, int], int]] = deque([(start, 0)])
        max_depth = 18
        center = (self.size - 1) / 2.0
        sx, sy = self.spawn
        stuck_mode = self.phase != "opening" and (
            self.stagnation_count >= 3 or self.moves_since_last_egg >= 4
        )
        # In mid/endgame, prioritize areas where we can lay eggs AND cut off opponent
        egg_seeking = self.phase != "opening" and self.moves_since_last_egg >= 2
        cutoff_seeking = self.phase != "opening"
        dist_penalty = 0.18 if stuck_mode else 0.3
        
        # Opponent location for cutoff targeting
        enemy_loc = board.chicken_enemy.get_location()
        
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
            score = 6.0 * novelty - dist_penalty * dist - 0.6 * risk
            
            # Opening bias: nudge toward centerline and outward expansion
            if self.phase == "opening":
                score += -0.12 * center_dist + 0.05 * outward
            elif stuck_mode:
                score += 0.35 * outward - 0.08 * center_dist + 0.2 * dist
            
            # Mid/endgame: strongly prefer tiles where we can lay eggs
            if egg_seeking and board.can_lay_egg_at_loc(loc):
                score += 4.0
                # Extra bonus for egg tiles far from current position (explore new areas)
                if dist >= 4:
                    score += 2.0
            
            # Mid/endgame: Prefer tiles that are close to the enemy (to cut them off)
            if cutoff_seeking:
                dist_to_enemy = abs(x - enemy_loc[0]) + abs(y - enemy_loc[1])
                # Bonus for being somewhat close to enemy (within interaction range)
                # but not suicide (risk check handles safety)
                if dist_to_enemy <= 5:
                    score += 3.0 * (6.0 - dist_to_enemy) / 6.0
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

    def _find_unexplored_regions(self, board: Board) -> List[Tuple[Set[Tuple[int, int]], Tuple[int, int]]]:
        """
        Find all contiguous unexplored regions and return them with their closest entry point.
        Returns list of (region_tiles, closest_entry_point) sorted by region size (largest first).
        """
        unexplored = set()
        for x in range(self.size):
            for y in range(self.size):
                loc = (x, y)
                if loc not in self.visited_tiles and board.is_valid_cell(loc) and not board.is_cell_blocked(loc):
                    unexplored.add(loc)
        
        if not unexplored:
            return []
        
        # Find connected components of unexplored tiles
        regions = []
        while unexplored:
            # BFS to find one connected region
            start = next(iter(unexplored))
            region = set()
            q = deque([start])
            while q:
                loc = q.popleft()
                if loc in region or loc not in unexplored:
                    continue
                region.add(loc)
                unexplored.discard(loc)
                for d in Direction:
                    nxt = loc_after_direction(loc, d)
                    if nxt in unexplored and nxt not in region:
                        q.append(nxt)
            if region:
                regions.append(region)
        
        # For each region, find the closest entry point (visited tile adjacent to region)
        cur_loc = board.chicken_player.get_location()
        result = []
        for region in regions:
            # Find tiles on the boundary (visited tiles adjacent to this region)
            best_entry = None
            best_dist = 999
            for tile in region:
                for d in Direction:
                    adj = loc_after_direction(tile, d)
                    if adj in self.visited_tiles or adj == cur_loc:
                        dist = self._manhattan(cur_loc, adj)
                        if dist < best_dist:
                            best_dist = dist
                            best_entry = tile  # The unexplored tile we want to reach
            if best_entry is not None:
                result.append((region, best_entry))
        
        # Sort by region size (largest first)
        result.sort(key=lambda x: len(x[0]), reverse=True)
        return result

    def _get_largest_unexplored_target(self, board: Board) -> Optional[Tuple[int, int]]:
        """Get a target tile in the largest unexplored region."""
        regions = self._find_unexplored_regions(board)
        if not regions:
            return None
        
        # Get the largest region
        largest_region, entry_point = regions[0]
        
        # If region is too small (< 4 tiles), not worth special targeting
        if len(largest_region) < 4:
            return None
        
        return entry_point

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

    def _egg_in_k_steps(self, board: Board, loc: Tuple[int, int], k: int) -> bool:
        """Return True if we can reach an eggable tile in k or fewer steps."""
        if k <= 0:
            return board.can_lay_egg_at_loc(loc)
        if board.can_lay_egg_at_loc(loc):
            return True
        for d in Direction:
            nxt = loc_after_direction(loc, d)
            if not board.is_valid_cell(nxt) or board.is_cell_blocked(nxt):
                continue
            if self._egg_in_k_steps(board, nxt, k - 1):
                return True
        return False

    def _egg_chain_score(self, board: Board, next_loc: Tuple[int, int], max_depth: int = 3) -> float:
        """
        Score how well a move sets up future egg-laying opportunities.
        Higher score = better chain potential.
        """
        # If we can egg immediately at next_loc, perfect chain
        if board.can_lay_egg_at_loc(next_loc):
            return 1.0
        
        score = 0.0
        # Can we egg in 1 step from next_loc?
        if self._egg_in_k_steps(board, next_loc, k=1):
            score = 0.8
        # Can we egg in 2 steps?
        elif self._egg_in_k_steps(board, next_loc, k=2):
            score = 0.5
        # Can we egg in 3 steps?
        elif max_depth >= 3 and self._egg_in_k_steps(board, next_loc, k=3):
            score = 0.3
        
        return score

    def _count_adjacent_eggs(self, board: Board, loc: Tuple[int, int]) -> int:
        """Count how many of our eggs are adjacent to this location."""
        count = 0
        for d in Direction:
            adj = loc_after_direction(loc, d)
            if adj in board.eggs_player:
                count += 1
        return count

    def _nearest_egg_distance(self, board: Board, start: Tuple[int, int], limit: int = 5) -> int:
        """Return BFS distance to the nearest eggable tile up to a limit (else limit+1)."""
        visited = set([start])
        q: deque[Tuple[Tuple[int, int], int]] = deque([(start, 0)])
        while q:
            loc, dist = q.popleft()
            if dist > limit:
                break
            if board.can_lay_egg_at_loc(loc):
                return dist
            for dir_ in Direction:
                nxt = loc_after_direction(loc, dir_)
                if nxt in visited or not self._in_bounds(nxt):
                    continue
                if board.is_cell_blocked(nxt):
                    continue
                visited.add(nxt)
                q.append((nxt, dist + 1))
        return limit + 1

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
    
    def _enemy_cutoff_bonus(self, board: Board, loc: Tuple[int, int]) -> float:
        """
        MALDINI: Reward moves that cut off opponent from open space.
        Returns bonus for positioning between enemy and unexplored territory.
        """
        enemy_loc = board.chicken_enemy.get_location()
        my_loc = board.chicken_player.get_location()
        
        # Calculate how much open space enemy can reach vs us
        # Simple heuristic: count reachable unvisited tiles within radius 4
        def count_reachable_open(start: Tuple[int, int], blocked_by: Optional[Tuple[int, int]] = None) -> int:
            visited = {start}
            if blocked_by:
                visited.add(blocked_by)
            queue = [start]
            count = 0
            depth = 0
            max_depth = 4
            next_queue = []
            while queue and depth < max_depth:
                for pos in queue:
                    for d in Direction:
                        nxt = loc_after_direction(pos, d)
                        if nxt in visited:
                            continue
                        if not self._in_bounds(nxt):
                            continue
                        if board.is_cell_blocked(nxt):
                            continue
                        visited.add(nxt)
                        next_queue.append(nxt)
                        # Count tiles not yet visited by us
                        if nxt not in self.visited_tiles:
                            count += 1
                queue = next_queue
                next_queue = []
                depth += 1
            return count
        
        # Enemy's open space if we move to loc vs if we don't
        enemy_open_with_us = count_reachable_open(enemy_loc, blocked_by=loc)
        enemy_open_without = count_reachable_open(enemy_loc, blocked_by=None)
        
        # Bonus for reducing enemy's access to open space
        reduction = enemy_open_without - enemy_open_with_us
        bonus = 0.0
        if reduction > 0:
            bonus = min(30.0, 6.0 * reduction)  # Increased significantly to prioritize cutting off
        
        # MALDINI: Additional bonus for intercepting enemy's predicted path
        predicted_dir = self._predict_enemy_direction()
        if predicted_dir is not None:
            # Where is enemy likely heading?
            predicted_enemy_loc = loc_after_direction(enemy_loc, predicted_dir)
            # Bonus if we're moving toward their predicted path
            dist_to_predicted = self._manhattan(loc, predicted_enemy_loc)
            dist_from_cur = self._manhattan(my_loc, predicted_enemy_loc)
            if dist_to_predicted < dist_from_cur:
                bonus += 3.0  # Intercepting their path
        
        return bonus
    
    def _opponent_pressure_score(self, board: Board, loc: Tuple[int, int]) -> float:
        """
        MALDINI: Score how much pressure we're putting on the opponent.
        Higher = we're limiting their options more.
        """
        enemy_loc = board.chicken_enemy.get_location()
        dist = self._manhattan(loc, enemy_loc)
        
        # Base pressure from proximity (closer = more pressure)
        # STRONGER bonuses for being close in midgame
        if dist <= 1:
            pressure = 20.0  # In their face
        elif dist <= 2:
            pressure = 12.0  # Very close
        elif dist <= 3:
            pressure = 6.0   # Pressuring
        elif dist <= 4:
            pressure = 3.0   # Within striking distance
        else:
            pressure = 0.0
        
        # Bonus if we're between enemy and open space
        enemy_to_us = (loc[0] - enemy_loc[0], loc[1] - enemy_loc[1])
        
        # Count unexplored tiles in each quadrant relative to enemy
        unexplored_toward_us = 0
        unexplored_away = 0
        corners_toward_us = 0
        parity_toward_us = 0
        
        corners = [(0, 0), (0, self.size - 1), (self.size - 1, 0), (self.size - 1, self.size - 1)]
        
        for x in range(self.size):
            for y in range(self.size):
                tile = (x, y)
                if tile in self.visited_tiles or tile in board.eggs_enemy:
                    continue
                if board.is_cell_blocked(tile):
                    continue
                # Is this tile in our direction from enemy or away?
                tile_dir = (x - enemy_loc[0], y - enemy_loc[1])
                # Dot product to see if same direction
                dot = tile_dir[0] * enemy_to_us[0] + tile_dir[1] * enemy_to_us[1]
                if dot > 0:
                    unexplored_toward_us += 1
                    if tile in corners:
                        corners_toward_us += 1
                    if self._is_my_parity(tile):
                        parity_toward_us += 1
                else:
                    unexplored_away += 1
        
        # Bonus if we're blocking access to more unexplored tiles
        if unexplored_toward_us > unexplored_away:
            pressure += 4.0  # We're blocking their expansion
            if unexplored_toward_us > 8:
                pressure += 3.0  # Blocking a lot of tiles
        
        # Extra bonus for blocking corners and parity tiles
        pressure += 2.0 * corners_toward_us
        pressure += 0.5 * parity_toward_us
        
        # Bonus for being on same row/column as enemy (limits their movement)
        if loc[0] == enemy_loc[0] or loc[1] == enemy_loc[1]:
            if dist <= 3:
                pressure += 4.0  # Same axis = limiting their options
        
        return pressure
    
    def _predict_enemy_direction(self) -> Optional[Direction]:
        """Predict which direction enemy is likely heading based on recent moves."""
        if len(self.enemy_positions) < 2:
            return None
        
        # Look at last 3 positions to find trend
        positions = list(self.enemy_positions)[-3:]
        if len(positions) < 2:
            return None
        
        # Calculate net movement
        dx = positions[-1][0] - positions[0][0]
        dy = positions[-1][1] - positions[0][1]
        
        # Determine dominant direction
        if abs(dx) > abs(dy):
            return Direction.RIGHT if dx > 0 else Direction.LEFT
        elif abs(dy) > 0:
            return Direction.DOWN if dy > 0 else Direction.UP
        return None

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
    
    def _reachable_parity_territory(
        self,
        board: Board,
        friendly: bool,
        max_depth: int = 6,
    ) -> int:
        """
        Approximate how many parity-correct tiles this side can still realistically convert
        into eggs in the near future.

        We BFS from the current chicken location up to max_depth steps (or turns_left),
        counting passable tiles with the correct parity that are not already eggs/turds.
        """
        if friendly:
            start = board.chicken_player.get_location()
            parity = self.my_parity
            eggs = board.eggs_player
            my_turds = board.turds_player
            opp_turds = board.turds_enemy
        else:
            start = board.chicken_enemy.get_location()
            parity = self.enemy_parity
            eggs = board.eggs_enemy
            my_turds = board.turds_enemy
            opp_turds = board.turds_player

        # Cap by remaining turns so we don't overestimate in late game
        max_depth = min(max_depth, max(1, board.turns_left_player))

        visited: Set[Tuple[int, int]] = {start}
        q: deque[Tuple[Tuple[int, int], int]] = deque([(start, 0)])
        dist = {start: 0}
        count = 0

        while q:
            loc, d = q.popleft()
            if d >= max_depth:
                continue

            for dir_ in Direction:
                nxt = loc_after_direction(loc, dir_)
                if nxt in visited:
                    continue
                if not self._in_bounds(nxt):
                    continue
                if board.is_cell_blocked(nxt):
                    continue

                visited.add(nxt)
                nd = d + 1
                dist[nxt] = nd
                q.append((nxt, nd))

                x, y = nxt
                if (x + y) % 2 != parity:
                    continue
                if nxt in eggs or nxt in my_turds or nxt in opp_turds:
                    continue

                count += 1

        return count


    def _reachable_region_size(self, board: Board, start: Tuple[int, int]) -> int:
        """
        Flood-fill from `start` over passable cells and return how many cells are
        in this connected region. This approximates how big an 'island' a side
        is currently stuck on (used to avoid being happily boxed in).
        """
        if not self._in_bounds(start) or board.is_cell_blocked(start):
            return 0

        visited: Set[Tuple[int, int]] = {start}
        q: deque[Tuple[Tuple[int, int], int]] = deque([(start, 0)])

        while q:
            loc, _ = q.popleft()
            for d in Direction:
                nxt = loc_after_direction(loc, d)
                if nxt in visited:
                    continue
                if not self._in_bounds(nxt):
                    continue
                if board.is_cell_blocked(nxt):
                    continue
                visited.add(nxt)
                q.append((nxt, 0))

        return len(visited)

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

    def _oscillation_penalty(self, cur: Tuple[int, int], next_loc: Tuple[int, int]) -> float:
        """Detect and heavily penalize A↔B oscillation patterns."""
        n = len(self.recent_positions)
        if n < 4:
            return 0.0
        recent = list(self.recent_positions)[-8:]
        
        # Check for A-B-A-B pattern (oscillating between two tiles)
        unique_recent = set(recent)
        # Only trigger if we've been stuck for at least 4 moves
        if len(unique_recent) <= 2 and len(recent) >= 4:
            # We've been bouncing between at most 2 tiles
            if next_loc in unique_recent:
                # Staying in oscillation - penalty scales with duration
                duration = len(recent) - 3  # How many turns beyond minimum
                return 30.0 + 8.0 * duration
            else:
                # Breaking out! Give a BONUS (negative penalty)
                return -15.0
        
        # Check for A-B-A pattern in last 4 positions
        if n >= 4:
            if recent[-1] == recent[-3] and recent[-2] == recent[-4] and next_loc == recent[-2]:
                return 20.0
        
        return 0.0
    
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

