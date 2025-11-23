from __future__ import annotations

from collections import deque
from typing import Callable, List, Optional, Sequence, Tuple

from game.board import Board
from game.enums import Direction, MoveType, loc_after_direction

from agents.Fluffy.trapdoor_belief import TrapdoorBelief


class PlayerAgent:
    """
    Kylian: multi-phase heuristic agent with aggressive space control, trap awareness,
    and strategic turd placement.
    """

    def __init__(self, board: Board, time_left: Callable[[], float]):
        del time_left
        self.game_map = board.game_map
        self.size = self.game_map.MAP_SIZE

        # Strategic phases (measured in turns left for the player)
        self.MIDGAME_TURNS = 25
        self.ENDGAME_TURNS = 12
        self.game_phase = "opening"

        # Trapdoor reasoning
        self.trap_belief = TrapdoorBelief(self.game_map)

        # Turd management
        self.turd_strategy = TurdStrategy()

        # History for pattern avoidance
        self.my_history: List[Tuple[int, Tuple[Direction, MoveType]]] = []

        # Cached weights
        self.corner_bonus = 200.0
        self.base_egg_value = 120.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def play(
        self,
        board: Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable[[], float],
    ) -> Tuple[Direction, MoveType]:
        del time_left

        self._update_phase(board)
        self._update_trapdoor_beliefs(board, sensor_data)

        valid_moves = board.get_valid_moves()
        if not valid_moves:
            return Direction.UP, MoveType.PLAIN

        best_move = valid_moves[0]
        best_score = float("-inf")

        for move in valid_moves:
            score = self._evaluate_move(board, move)
            if score > best_score:
                best_score = score
                best_move = move

        self.my_history.append((board.turn_count, best_move))
        return best_move

    # ------------------------------------------------------------------
    # Move evaluation
    # ------------------------------------------------------------------
    def _evaluate_move(self, board: Board, move: Tuple[Direction, MoveType]) -> float:
        future_board = self._forecast(board, move)
        if future_board is None:
            return float("-inf")

        dir, move_type = move
        score = 0.0

        my_loc = future_board.chicken_player.get_location()
        enemy_loc = future_board.chicken_enemy.get_location()

        # 1. Egg laying priority
        if move_type == MoveType.EGG:
            score += self._egg_priority(board, future_board, my_loc)

        # 2. Turd evaluation
        elif move_type == MoveType.TURD:
            turd_score = self.turd_strategy.evaluate(board, future_board, my_loc, enemy_loc)
            if not self.turd_strategy.should_place(
                self.game_phase, turd_score, board.chicken_player.get_turds_left()
            ):
                turd_score -= 120.0
            score += turd_score

        # 3. Position quality
        score += 30.0 * self._position_quality(my_loc, enemy_loc)

        # 4. Trapdoor risk
        trap_risk = self._trapdoor_risk(board, my_loc)
        if trap_risk >= 0.7:
            score -= 250.0 * trap_risk
        elif trap_risk >= 0.3:
            score -= 90.0 * trap_risk

        # 5. Mobility & space control
        score += 12.0 * self._mobility_diff(future_board)

        # 6. Blocking potential (can we pressure traps?)
        score += self._blocking_potential(future_board)

        # 7. Territory / target pull
        score += 8.0 * self._territory_pull(board, my_loc)

        # 8. Endgame considerations
        score += 40.0 * self._endgame_pressure(board, future_board)

        # 9. Pattern avoidance
        score -= self._pattern_penalty(move)

        return score

    # ------------------------------------------------------------------
    # Heuristic components
    # ------------------------------------------------------------------
    def _egg_priority(self, board: Board, future_board: Board, loc: Tuple[int, int]) -> float:
        score = self.base_egg_value
        if self._is_corner(loc):
            score += self.corner_bonus

        score += 15.0 * self._territory_control(future_board, loc)
        if self.game_phase == "opening":
            score += 10.0  # faster expansion
        return score

    def _position_quality(self, my_loc: Tuple[int, int], enemy_loc: Tuple[int, int]) -> float:
        center = (self.size - 1) / 2.0
        dist_center = abs(my_loc[0] - center) + abs(my_loc[1] - center)
        score = 0.0

        if 2 <= dist_center <= 3:
            score += 1.2
        elif dist_center < 2:
            score += 0.4

        if self._is_edge(my_loc):
            score += 0.3

        # Maintain pressure distance
        dist_enemy = abs(my_loc[0] - enemy_loc[0]) + abs(my_loc[1] - enemy_loc[1])
        if dist_enemy == 3:
            score += 1.0
        elif dist_enemy == 2:
            score += 0.5
        elif dist_enemy <= 1:
            score -= 0.6
        elif dist_enemy >= 6:
            score -= 0.3

        return score

    def _mobility_diff(self, future_board: Board) -> float:
        my_moves = len(future_board.get_valid_moves())
        enemy_moves = len(future_board.get_valid_moves(enemy=True))
        return float(my_moves - enemy_moves)

    def _blocking_potential(self, future_board: Board) -> float:
        enemy_moves = len(future_board.get_valid_moves(enemy=True))
        if enemy_moves <= 1:
            return 25.0
        if enemy_moves == 2:
            return 12.0
        return 0.0

    def _territory_pull(self, board: Board, loc: Tuple[int, int]) -> float:
        pull = 0.0
        for target in self._high_value_cells(board):
            dist = abs(loc[0] - target[0]) + abs(loc[1] - target[1])
            pull += 1.0 / (dist + 1)
        return pull

    def _high_value_cells(self, board: Board) -> List[Tuple[int, int]]:
        cells: List[Tuple[int, int]] = []
        for x in range(self.size):
            for y in range(self.size):
                cell = (x, y)
                if board.can_lay_egg_at_loc(cell) and cell not in board.eggs_player:
                    weight = 1.0
                    if self._is_corner(cell):
                        weight = 3.0
                    if weight >= 1.0:
                        cells.extend([cell] * int(weight))
        if not cells:
            cells.append(board.chicken_player.get_location())
        return cells

    def _territory_control(self, board: Board, loc: Tuple[int, int]) -> float:
        control = 0.0
        for dir in Direction:
            nxt = loc_after_direction(loc, dir)
            if board.is_valid_cell(nxt) and board.can_lay_egg_at_loc(nxt):
                control += 1.0
                if self._is_corner(nxt):
                    control += 1.5

        # radius-2 influence
        visited = set([loc])
        q = deque([(loc, 0)])
        while q:
            cell, dist = q.popleft()
            if dist >= 2:
                continue
            for dir in Direction:
                nxt = loc_after_direction(cell, dir)
                if not board.is_valid_cell(nxt) or nxt in visited:
                    continue
                visited.add(nxt)
                if board.can_lay_egg_at_loc(nxt):
                    control += 0.3
                q.append((nxt, dist + 1))
        return control

    def _endgame_pressure(self, board: Board, future_board: Board) -> float:
        if self.game_phase != "endgame":
            return 0.0
        my_eggs = len(future_board.eggs_player)
        enemy_eggs = len(future_board.eggs_enemy)
        diff = my_eggs - enemy_eggs
        if diff > 0:
            return 1.5
        if diff < 0:
            return -1.0
        return 0.0

    def _pattern_penalty(self, move: Tuple[Direction, MoveType]) -> float:
        if len(self.my_history) < 3:
            return 0.0
        last_moves = [m for _, m in self.my_history[-3:]]
        last_moves.append(move)
        unique = set(last_moves)
        if len(unique) <= 2:
            return 15.0
        return 0.0

    # ------------------------------------------------------------------
    # Trapdoor handling
    # ------------------------------------------------------------------
    def _update_trapdoor_beliefs(self, board: Board, sensor_data: List[Tuple[bool, bool]]) -> None:
        if not sensor_data:
            return
        self.trap_belief.update(board.chicken_player, sensor_data)
        for loc in getattr(board, "found_trapdoors", []):
            self.trap_belief.register_known_trapdoor(loc)

    def _trapdoor_risk(self, board: Board, loc: Tuple[int, int]) -> float:
        if loc in getattr(board, "found_trapdoors", set()):
            return 1.0
        return float(self.trap_belief.trapdoor_prob_at(loc))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _update_phase(self, board: Board) -> None:
        turns_left = board.turns_left_player
        if turns_left <= self.ENDGAME_TURNS:
            self.game_phase = "endgame"
        elif turns_left <= self.MIDGAME_TURNS:
            self.game_phase = "midgame"
        else:
            self.game_phase = "opening"

    def _forecast(self, board: Board, move: Tuple[Direction, MoveType]) -> Optional[Board]:
        copy = board.get_copy()
        ok = copy.apply_move(move[0], move[1], check_ok=True)
        if not ok:
            return None
        return copy

    def _is_corner(self, loc: Tuple[int, int]) -> bool:
        x, y = loc
        return (x in (0, self.size - 1)) and (y in (0, self.size - 1))

    def _is_edge(self, loc: Tuple[int, int]) -> bool:
        x, y = loc
        return x in (0, self.size - 1) or y in (0, self.size - 1)


class TurdStrategy:
    """Evaluate whether a turd placement meaningfully blocks the opponent."""

    def __init__(self):
        self.thresholds = {
            "opening": 150.0,
            "midgame": 90.0,
            "endgame": 60.0,
        }

    def evaluate(
        self,
        board: Board,
        future_board: Board,
        loc: Tuple[int, int],
        enemy_loc: Tuple[int, int],
    ) -> float:
        score = 0.0

        blocked_egg_sites = 0
        for dir in Direction:
            adj = loc_after_direction(loc, dir)
            if board.is_valid_cell(adj) and board.chicken_enemy.can_lay_egg(adj):
                blocked_egg_sites += 1
                if (adj[0] in (0, board.game_map.MAP_SIZE - 1)) and (
                    adj[1] in (0, board.game_map.MAP_SIZE - 1)
                ):
                    blocked_egg_sites += 2  # corner pressure

        score += blocked_egg_sites * 40.0

        # Mobility reduction
        before_moves = len(board.get_valid_moves(enemy=True))
        after_moves = len(future_board.get_valid_moves(enemy=True))
        score += (before_moves - after_moves) * 25.0

        # Choke point detection
        if self._creates_choke(board, loc):
            score += 80.0

        # Don't place turd too close to enemy (risk of waste)
        if abs(loc[0] - enemy_loc[0]) + abs(loc[1] - enemy_loc[1]) <= 1:
            score -= 60.0

        return score

    def should_place(self, phase: str, value: float, turds_left: int) -> bool:
        if turds_left <= 0:
            return False
        threshold = self.thresholds.get(phase, 90.0)
        return value >= threshold

    def _creates_choke(self, board: Board, loc: Tuple[int, int]) -> bool:
        open_neighbors = 0
        for d in Direction:
            nxt = loc_after_direction(loc, d)
            if board.is_valid_cell(nxt) and nxt not in board.turds_player and nxt not in board.turds_enemy:
                open_neighbors += 1
        return open_neighbors <= 2


