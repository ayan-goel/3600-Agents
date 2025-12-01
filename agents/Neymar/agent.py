from __future__ import annotations

from collections import deque
from typing import Callable, Dict, List, Set, Tuple

import numpy as np

from game.board import Board
from game.enums import Direction, MoveType, loc_after_direction


class SearchTimeout(Exception):
    pass


def prob_hear(dx: int, dy: int) -> float:
    if dx > 2 or dy > 2:
        return 0.0
    if dx == 2 and dy == 2:
        return 0.0
    if dx == 2 or dy == 2:
        return 0.1
    if dx == 1 and dy == 1:
        return 0.25
    if dx == 1 or dy == 1:
        return 0.5
    return 0.0


def prob_feel(dx: int, dy: int) -> float:
    if dx > 1 or dy > 1:
        return 0.0
    if dx == 1 and dy == 1:
        return 0.15
    if dx == 1 or dy == 1:
        return 0.3
    return 0.0


class TrapdoorBelief:
    def __init__(self, size: int):
        self.size = size
        self.belief_even = np.zeros((size, size), dtype=np.float64)
        self.belief_odd = np.zeros((size, size), dtype=np.float64)
        self._safe_tiles: Set[Tuple[int, int]] = set()
        self._init_center_prior()

    def _init_center_prior(self):
        for y in range(self.size):
            for x in range(self.size):
                dist_to_edge = min(x, y, self.size - 1 - x, self.size - 1 - y)
                if dist_to_edge <= 1:
                    weight = 0.0
                elif dist_to_edge == 2:
                    weight = 1.0
                else:
                    weight = 2.0
                
                if weight > 0:
                    if (x + y) % 2 == 0:
                        self.belief_even[y, x] = weight
                    else:
                        self.belief_odd[y, x] = weight
        
        s_even = self.belief_even.sum()
        s_odd = self.belief_odd.sum()
        if s_even > 0:
            self.belief_even /= s_even
        if s_odd > 0:
            self.belief_odd /= s_odd

    def update(self, chicken_loc: Tuple[int, int], sensor_data: List[Tuple[bool, bool]]):
        if len(sensor_data) < 2:
            return
        
        self.mark_safe(chicken_loc)
        
        even_hear, even_feel = sensor_data[0]
        odd_hear, odd_feel = sensor_data[1]
        
        self._update_grid(self.belief_even, chicken_loc, even_hear, even_feel, 0)
        self._update_grid(self.belief_odd, chicken_loc, odd_hear, odd_feel, 1)

    def _update_grid(self, belief: np.ndarray, loc: Tuple[int, int], 
                     did_hear: bool, did_feel: bool, parity: int):
        cx, cy = loc
        
        for y in range(self.size):
            for x in range(self.size):
                if (x + y) % 2 != parity:
                    continue
                if (x, y) in self._safe_tiles:
                    belief[y, x] = 0.0
                    continue
                if belief[y, x] <= 0:
                    continue
                
                dx, dy = abs(x - cx), abs(y - cy)
                p_h = prob_hear(dx, dy)
                p_f = prob_feel(dx, dy)
                
                if did_hear:
                    lh = p_h if p_h > 0 else 0.001
                else:
                    lh = 1.0 - p_h
                
                if did_feel:
                    lf = p_f if p_f > 0 else 0.001
                else:
                    lf = 1.0 - p_f
                
                likelihood = max(lh * lf, 1e-9)
                belief[y, x] *= likelihood
        
        total = belief.sum()
        if total > 0:
            belief /= total
        else:
            self._init_single_grid(belief, parity)

    def _init_single_grid(self, belief: np.ndarray, parity: int):
        for y in range(self.size):
            for x in range(self.size):
                if (x + y) % 2 != parity:
                    continue
                if (x, y) in self._safe_tiles:
                    continue
                dist_to_edge = min(x, y, self.size - 1 - x, self.size - 1 - y)
                if dist_to_edge <= 1:
                    belief[y, x] = 0.0
                elif dist_to_edge == 2:
                    belief[y, x] = 1.0
                else:
                    belief[y, x] = 2.0
        total = belief.sum()
        if total > 0:
            belief /= total

    def mark_safe(self, loc: Tuple[int, int]):
        x, y = loc
        if 0 <= x < self.size and 0 <= y < self.size:
            self._safe_tiles.add(loc)
            self.belief_even[y, x] = 0.0
            self.belief_odd[y, x] = 0.0
            
            s_even = self.belief_even.sum()
            s_odd = self.belief_odd.sum()
            if s_even > 0:
                self.belief_even /= s_even
            if s_odd > 0:
                self.belief_odd /= s_odd

    def get_risk(self, loc: Tuple[int, int]) -> float:
        x, y = loc
        if not (0 <= x < self.size and 0 <= y < self.size):
            return 1.0
        if loc in self._safe_tiles:
            return 0.0
        if (x + y) % 2 == 0:
            return float(self.belief_even[y, x])
        return float(self.belief_odd[y, x])
    
    def get_total_risk(self, loc: Tuple[int, int]) -> float:
        x, y = loc
        if not (0 <= x < self.size and 0 <= y < self.size):
            return 1.0
        if loc in self._safe_tiles:
            return 0.0
        return float(self.belief_even[y, x]) + float(self.belief_odd[y, x])

    def get_high_risk_tiles(self, threshold: float = 0.1) -> Set[Tuple[int, int]]:
        danger = set()
        for y in range(self.size):
            for x in range(self.size):
                if (x, y) in self._safe_tiles:
                    continue
                risk = self.belief_even[y, x] + self.belief_odd[y, x]
                if risk >= threshold:
                    danger.add((x, y))
        return danger


class PlayerAgent:
    DEPTH = 9

    def __init__(self, board: Board, time_left: Callable[[], float]):
        self.size = board.game_map.MAP_SIZE
        self.my_parity = board.chicken_player.even_chicken
        self.enemy_parity = board.chicken_enemy.even_chicken
        self.trap_belief = TrapdoorBelief(self.size)
        self.visit_counts: Dict[Tuple[int, int], int] = {}
        self.tt: Dict[int, Tuple[float, int]] = {}
        self._deadline = 0.0
        self._cached_potential = 0.0
        self._danger_zone: Set[Tuple[int, int]] = set()
        self._is_endgame = False

    def play(
        self,
        board: Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable[[], float],
    ) -> Tuple[Direction, MoveType]:
        my_loc = board.chicken_player.get_location()
        enemy_loc = board.chicken_enemy.get_location()
        
        self.trap_belief.update(my_loc, sensor_data)
        self.trap_belief.mark_safe(enemy_loc)
        self.visit_counts[my_loc] = self.visit_counts.get(my_loc, 0) + 1
        
        self._danger_zone = self.trap_belief.get_high_risk_tiles(0.08)
        
        # Endgame detection: last ~20 moves (turn_count >= 60 out of ~80)
        self._is_endgame = board.turn_count >= 60
        
        self.tt.clear()
        
        my_pot = self._bfs_potential(board, my_loc, self.my_parity, False)
        en_pot = self._bfs_potential(board, enemy_loc, self.enemy_parity, True)
        self._cached_potential = (my_pot - en_pot) * 2.0
        
        legal = board.get_valid_moves()
        if not legal:
            return Direction.UP, MoveType.PLAIN

        filtered = self._filter(board, legal)
        if len(filtered) == 1:
            return filtered[0]

        moves_left = max(1, 40 - board.turn_count // 2)
        budget = min(time_left() / moves_left * 0.6, 4.0)
        self._deadline = time_left() - budget

        best = filtered[0]
        
        try:
            for d in range(1, self.DEPTH + 1):
                if time_left() < self._deadline:
                    break
                scores = []
                for m in filtered:
                    if time_left() < self._deadline:
                        raise SearchTimeout()
                    nb = board.forecast_move(m[0], m[1], check_ok=False)
                    if nb:
                        s = self._ab(nb, d - 1, -1e9, 1e9, False, time_left)
                        scores.append((m, s))
                if scores:
                    scores.sort(key=lambda x: x[1], reverse=True)
                    best = scores[0][0]
                    filtered = [x[0] for x in scores]
        except SearchTimeout:
            pass

        return best

    def _filter(self, board: Board, moves: List[Tuple[Direction, MoveType]]) -> List[Tuple[Direction, MoveType]]:
        cur = board.chicken_player.get_location()
        
        if board.can_lay_egg():
            eggs = [m for m in moves if m[1] == MoveType.EGG]
            if eggs:
                # In endgame, prioritize eggs but still avoid high-risk trapdoor tiles
                if self._is_endgame:
                    safe_eggs = []
                    moderate_eggs = []
                    risky_eggs = []
                    for m in eggs:
                        nl = loc_after_direction(cur, m[0])
                        risk = self.trap_belief.get_total_risk(nl)
                        if risk < 0.20:  # Slightly more permissive than normal
                            safe_eggs.append((m, risk))
                        elif risk < 0.35:  # Moderate risk - acceptable in endgame
                            moderate_eggs.append((m, risk))
                        else:
                            risky_eggs.append((m, risk))
                    
                    # Prefer safe eggs, then moderate, avoid truly risky ones
                    if safe_eggs:
                        safe_eggs.sort(key=lambda x: x[1])
                        return [x[0] for x in safe_eggs]
                    if moderate_eggs:
                        moderate_eggs.sort(key=lambda x: x[1])
                        return [x[0] for x in moderate_eggs]
                    # Only take risky eggs if no other option
                    if risky_eggs:
                        risky_eggs.sort(key=lambda x: x[1])
                        return [x[0] for x in risky_eggs]
                
                safe_eggs = []
                risky_eggs = []
                for m in eggs:
                    nl = loc_after_direction(cur, m[0])
                    risk = self.trap_belief.get_total_risk(nl)
                    if risk < 0.15:
                        safe_eggs.append((m, risk))
                    else:
                        risky_eggs.append((m, risk))
                if safe_eggs:
                    safe_eggs.sort(key=lambda x: x[1])
                    return [x[0] for x in safe_eggs]
                risky_eggs.sort(key=lambda x: x[1])
                return [x[0] for x in risky_eggs]
        
        safe_moves = []
        risky_moves = []
        
        for m in moves:
            nl = loc_after_direction(cur, m[0])
            if m[1] == MoveType.TURD:
                if nl[0] in (0, self.size-1) or nl[1] in (0, self.size-1):
                    continue
            
            risk = self.trap_belief.get_total_risk(nl)
            if risk < 0.15:
                safe_moves.append((m, risk))
            elif risk < 0.4:
                risky_moves.append((m, risk))
        
        if safe_moves:
            safe_moves.sort(key=lambda x: x[1])
            return [x[0] for x in safe_moves]
        
        if risky_moves:
            risky_moves.sort(key=lambda x: x[1])
            return [x[0] for x in risky_moves]
        
        result = []
        for m in moves:
            nl = loc_after_direction(cur, m[0])
            if m[1] == MoveType.TURD:
                if nl[0] in (0, self.size-1) or nl[1] in (0, self.size-1):
                    continue
            result.append(m)
        
        return result if result else moves

    def _ab(self, board: Board, depth: int, alpha: float, beta: float, 
            maximizing: bool, tl: Callable[[], float]) -> float:
        if tl() < self._deadline:
            raise SearchTimeout()

        if board.is_game_over():
            me = board.chicken_player.get_eggs_laid()
            en = board.chicken_enemy.get_eggs_laid()
            return (me - en) * 1000.0

        if depth == 0:
            return self._eval(board)

        h = hash((board.chicken_player.get_location(), board.chicken_enemy.get_location(),
                  board.chicken_player.get_eggs_laid(), board.chicken_enemy.get_eggs_laid(),
                  board.turn_count, maximizing))
        if h in self.tt:
            cs, cd = self.tt[h]
            if cd >= depth:
                return cs

        if maximizing:
            moves = board.get_valid_moves(enemy=False)
            if not moves:
                return self._eval(board)
            if board.can_lay_egg():
                moves = [m for m in moves if m[1] == MoveType.EGG] or moves
            val = -1e9
            for m in moves:
                nb = board.forecast_move(m[0], m[1], check_ok=False)
                if nb:
                    val = max(val, self._ab(nb, depth-1, alpha, beta, False, tl))
                    alpha = max(alpha, val)
                    if beta <= alpha:
                        break
            self.tt[h] = (val, depth)
            return val
        else:
            board.reverse_perspective()
            moves = board.get_valid_moves(enemy=False)
            if not moves:
                board.reverse_perspective()
                return self._eval(board)
            if board.can_lay_egg():
                moves = [m for m in moves if m[1] == MoveType.EGG] or moves
            val = 1e9
            for m in moves:
                nb = board.forecast_move(m[0], m[1], check_ok=False)
                if nb:
                    nb.reverse_perspective()
                    val = min(val, self._ab(nb, depth-1, alpha, beta, True, tl))
                    beta = min(beta, val)
                    if beta <= alpha:
                        break
            board.reverse_perspective()
            self.tt[h] = (val, depth)
            return val

    def _eval(self, board: Board) -> float:
        me = board.chicken_player.get_eggs_laid()
        en = board.chicken_enemy.get_eggs_laid()
        
        # In endgame, eggs matter more but still respect trapdoor risk
        egg_weight = 30.0 if self._is_endgame else 15.0
        score = (me - en) * egg_weight

        my_loc = board.chicken_player.get_location()
        en_loc = board.chicken_enemy.get_location()

        risk = self.trap_belief.get_total_risk(my_loc)
        # Keep meaningful risk penalty even in endgame - trapdoors still kill
        risk_penalty = 100.0 if self._is_endgame else 150.0
        score -= risk * risk_penalty
        
        if my_loc in self._danger_zone:
            danger_penalty = 35.0 if self._is_endgame else 50.0
            score -= danger_penalty

        vc = self.visit_counts.get(my_loc, 0)
        score -= 3.0 * max(0, vc - 1)

        cx, cy = self.size // 2, self.size // 2
        for t in board.turds_player:
            if abs(t[0] - cx) <= 2 and abs(t[1] - cy) <= 2:
                score += 5.0

        score += self._cached_potential

        corners = [(0,0), (0,self.size-1), (self.size-1,0), (self.size-1,self.size-1)]
        for c in corners:
            if c not in board.eggs_player and c not in board.eggs_enemy:
                if (c[0] + c[1]) % 2 == self.my_parity:
                    md = abs(my_loc[0]-c[0]) + abs(my_loc[1]-c[1])
                    ed = abs(en_loc[0]-c[0]) + abs(en_loc[1]-c[1])
                    if md < ed:
                        score += 12.0 / max(md, 1)

        return score

    def _bfs_potential(self, board: Board, start: Tuple[int, int], parity: int, is_enemy: bool) -> int:
        blocked_eggs = board.eggs_player if is_enemy else board.eggs_enemy
        blocked_turds = board.turds_player if is_enemy else board.turds_enemy
        all_obs = board.eggs_player | board.eggs_enemy | board.turds_player | board.turds_enemy

        visited: Set[Tuple[int, int]] = {start}
        queue = deque([start])
        count = 0

        while queue:
            loc = queue.popleft()
            if (loc[0] + loc[1]) % 2 == parity and loc not in all_obs:
                count += 1
            for d in Direction:
                nl = loc_after_direction(loc, d)
                if not (0 <= nl[0] < self.size and 0 <= nl[1] < self.size):
                    continue
                if nl in visited or nl in blocked_eggs:
                    continue
                blocked = nl in blocked_turds
                if not blocked:
                    for dd in Direction:
                        if loc_after_direction(nl, dd) in blocked_turds:
                            blocked = True
                            break
                if not blocked:
                    visited.add(nl)
                    queue.append(nl)
        return count
