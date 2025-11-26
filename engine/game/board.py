from typing import List, Tuple

from game.chicken import Chicken
from game.enums import (
    Direction,
    MoveType,
    Result,
    WinReason,
    loc_after_direction,
)
from game.game_map import GameMap
from game.history import History


def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


class Board:
    """
    Board is the representation of the state of the match. It contains
    only the state known to the player -- undiscovered trapdoors are not
    included in the Board (although found ones are).

    Any coordinates should be given to the board in the form of x, y.

    Check_validity is on by default for most functions, but slows
    down execution. If a player is confident their actions are valid,
    they can directly apply turns and moves with check_validity as false.

    Be wary that invalid actions/turns could lead to functions throwing
    errors, so make sure to handle them with a try/except in case so that
    your program doesn't crash. If an apply function throws an error,
    it is not guarenteed that the board state will be valid or that the state
    will be the same as when the function started.
    """

    def __init__(
        self,
        game_map: GameMap,
        time_to_play: float = 20,
        build_history: bool = False,
        copy: bool = False,
    ):
        """
        Initializes the board with the specified game map and configuration options.

        Parameters:
            game_map (game_map.GameMap): The map representing the game environment.
            time_to_play (float, optional): The time limit for the game in seconds. Defaults to 20.
            build_history (bool, optional): Whether to track the history of the game. Defaults to False.
            copy (bool, optional): Whether to initialize a copy of the game game_map. Defaults to False.
        """

        self.game_map = game_map

        if not copy:
            self.eggs_player = set()
            self.eggs_enemy = set()
            self.turds_player = set()
            self.turds_enemy = set()

            # These will get their initial positions and even-ness
            # from the gameplay
            self.chicken_player = Chicken(self.game_map.MAX_TURDS)
            self.chicken_enemy = Chicken(self.game_map.MAX_TURDS)

            self.found_trapdoors = set()

            # game metadata
            self.turn_count = 0

            self.MAX_TURNS = 40
            self.turns_left_player = self.MAX_TURNS
            self.turns_left_enemy = self.MAX_TURNS

            # more game metadata
            self.winner = None

            self.time_to_play = time_to_play
            self.player_time = time_to_play
            self.enemy_time = time_to_play
            self.win_reason = None

            self.chicken_blocked = False
            self.is_as_turn = True

            # history building
            self.build_history = build_history
            if build_history:
                self.history = History()

    def is_valid_cell(self, loc: Tuple[int, int]) -> bool:
        """
        Checks if the given coordinates are within the valid board boundaries.

        Parameters:
            (x, y): The location to check.

        Returns:
            (bool): True if the cell is valid, False otherwise.
        """
        return (
            loc[0] >= 0
            and loc[1] >= 0
            and loc[0] < self.game_map.MAP_SIZE
            and loc[1] < self.game_map.MAP_SIZE
        )

    def is_cell_in_enemy_turd_zone(self, loc: Tuple[int, int]) -> bool:
        """
        Checks if a specific cell is within the enemy's turd zone.

        Parameters:
            loc: The (x,y) location of the cell.

        Returns:
            (bool): True if the cell is within the enemy's turd zone, False otherwise.
        """
        for dir in Direction:
            new_loc = loc_after_direction(loc, dir)
            if new_loc in self.turds_enemy:
                return True
        if loc in self.turds_enemy:
            return True

        return False

    def is_cell_blocked(self, loc: Tuple[int, int]) -> bool:
        """
        Checks if a specific cell is blocked for the given player.

        Parameters:
            loc: The (x,y) location of the cell.

        Returns:
            (bool): True if the cell is blocked or occupied by the enemy chicken, False otherwise.
        """
        if not self.is_valid_cell(loc):
            return True

        enemy_loc = self.chicken_enemy.get_location()
        if enemy_loc == loc:
            return True

        if loc in self.eggs_enemy:
            return True

        if self.is_cell_in_enemy_turd_zone(loc):
            return True

        return False

    def is_valid_direction(self, dir: Direction | int):
        """
        Checks if a direction is valid for the given player.

        Parameters:
            dir (Direction | int): The direction to check.

        Returns:
            (bool): True if the direction is valid, False otherwise.
        """

        my_loc = self.chicken_player.get_location()
        next_loc = loc_after_direction(my_loc, dir)
        return not self.is_cell_blocked(next_loc)

    def is_valid_move(
        self, dir: Direction | int, move_type: MoveType | int, enemy: bool = False
    ):
        """
        Checks if a move is valid for the given player.

        Parameters:
            dir (Direction | int): The direction of the move.
            move_type (MoveType | int): The type of move (walk, egg, turd).

        Returns:
            (bool): True if the move is valid, False otherwise.
        """

        if enemy:
            if move_type == MoveType.TURD and self.chicken_enemy.get_turds_left() <= 0:
                return False
            test_loc = self.chicken_enemy.get_location()
            opposing_loc = self.chicken_player.get_location()
            even_chicken = self.chicken_enemy.even_chicken

        else:
            if move_type == MoveType.TURD and self.chicken_player.get_turds_left() <= 0:
                return False
            test_loc = self.chicken_player.get_location()
            opposing_loc = self.chicken_enemy.get_location()
            even_chicken = self.chicken_player.even_chicken

        # Wrong parity?
        if move_type == MoveType.EGG:
            if (test_loc[0] + test_loc[1]) % 2 != even_chicken:
                return False

        # Where would it end up?
        new_loc = loc_after_direction(test_loc, dir)

        # Would that put the chickens on top of each other?
        if new_loc == opposing_loc:
            return False

        # Is it off the board?
        if (
            new_loc[0] < 0
            or new_loc[1] < 0
            or new_loc[0] >= self.game_map.MAP_SIZE
            or new_loc[1] >= self.game_map.MAP_SIZE
        ):
            return False

        # Opposing eggs?
        if enemy:
            if new_loc in self.eggs_player:
                return False
        else:
            if new_loc in self.eggs_enemy:
                return False

        # Opposing turds?
        if enemy:
            problematic_turds = self.turds_player
        else:
            problematic_turds = self.turds_enemy

        for direction in Direction:
            adjacent_loc = loc_after_direction(new_loc, direction)
            if adjacent_loc in problematic_turds:
                return False

        # Plain move is fine
        if move_type == MoveType.PLAIN:
            return True

        # Is anything belonging to the same player already in the square?
        if enemy:
            if test_loc in self.eggs_enemy:
                return False
            if test_loc in self.turds_enemy:
                return False
        else:
            if test_loc in self.eggs_player:
                return False
            if test_loc in self.turds_player:
                return False

        # If it is an egg laying, we are good to go
        if move_type == MoveType.EGG:
            return True

        if manhattan_distance(test_loc, opposing_loc) < 2:
            return False

        return True

    def get_valid_moves(self, enemy: bool = False) -> List[Tuple[Direction, MoveType]]:
        """
        Returns a list of all valid moves for the player or enemy.

        Parameters:
            enemy (bool, optional): If True, returns valid moves for enemy; if False, returns for player.

        Returns:
            (List[Tuple[Direction, MoveType]]): List of tuples containing valid direction and move type combinations.
        """
        valid_moves = []
        for dir in Direction:
            for move_type in MoveType:
                if self.is_valid_move(dir, move_type, enemy=enemy):
                    valid_moves.append((dir, move_type))

        return valid_moves

    def can_lay_egg(self):
        """
        Checks if the player can lay an egg at their current location.

        Parameters:

        Returns:
            (bool): True if an egg can be laid, False otherwise.
        """

        my_loc = self.chicken_player.get_location()
        return self.can_lay_egg_at_loc(my_loc)

    def can_lay_egg_at_loc(self, loc: Tuple[int, int]):
        """
        Checks if the player can lay an egg at a specific location.

        Parameters:
            x (int): The x-coordinate of the location.
            y (int): The y-coordinate of the location.

        Returns:
            (bool): True if an egg can be laid at that location, False otherwise.
        """

        return (
            self.chicken_player.can_lay_egg(loc)
            and loc not in self.eggs_player
            and loc not in self.turds_player
            and loc not in self.turds_enemy
        )

    def can_lay_turd(self):
        """
        Checks if the player can lay a turd at their current location.


        Returns:
            (bool): True if a turd can be laid, False otherwise.
        """

        my_loc = self.chicken_player.get_location()
        return self.can_lay_turd_at_loc(my_loc)

    def can_lay_turd_at_loc(self, loc):
        """
        Checks if the player can lay a turd at a specific location.

        Parameters:
            x (int): The x-coordinate of the location.
            y (int): The y-coordinate of the location.

        Returns:
            (bool): True if a turd can be laid at that location, False otherwise.
        """
        enemy_loc = self.chicken_enemy.get_location()
        return (
            self.chicken_player.has_turds_left()
            and manhattan_distance(loc, enemy_loc) > 1
            and loc not in self.turds_player
            and loc not in self.turds_enemy
            and loc not in self.eggs_player
            and loc not in self.eggs_enemy
        )

    def apply_move(
        self,
        dir: Direction | int,
        move_type: MoveType | int,
        timer: float = 0,
        check_ok: bool = True,
    ):
        """
        Applies a move to the board for the specified player.

        Parameters:
            dir (Direction | int): The direction of the move.
            move_type (MoveType | int): The type of move (walk, egg, turd).
            timer (float, optional): Time taken for the move in seconds. Defaults to 0.
            check_ok (bool, optional): If True, validates the move before applying; if False, skips validation. Defaults to True.

        Returns:
            (bool): True if the move was successfully applied, False otherwise.
        """
        # TODO: put in try/except
        # try:

        if check_ok:
            if not self.is_valid_move(dir, move_type):
                return False

        my_loc = self.chicken_player.get_location()
        offset = 0 if self.is_as_turn else 2

        if move_type == MoveType.EGG:
            if (my_loc[0] == 0 or my_loc[0] == self.game_map.MAP_SIZE - 1) and (
                my_loc[1] == 0 or my_loc[1] == self.game_map.MAP_SIZE - 1
            ):
                self.chicken_player.increment_eggs_laid(self.game_map.CORNER_REWARD)
            else:
                self.chicken_player.increment_eggs_laid()
            self.eggs_player.add(my_loc)

        elif move_type == MoveType.TURD:
            self.chicken_player.decrement_turds()
            self.turds_player.add(my_loc)

        new_loc = self.chicken_player.apply_dir(dir)

        self.end_turn(move_type, timer)

        return True
        # except:
        #     return False

    def end_turn(self, move_type: MoveType, timer=0):
        """
        Ends the current turn and updates game state.

        Parameters:
            move_type (MoveType): The type of move that was made.
            timer (float, optional): Time taken for the turn in seconds. Defaults to 0.
        """
        self.turn_count += 1
        self.turns_left_player -= 1
        
        self.player_time -= timer

        if not self.has_moves_left(enemy=True):
            if self.turns_left_enemy > 0:
                self.chicken_player.increment_eggs_laid(5)
                self.chicken_blocked = True

        self.check_win()

        if self.build_history:
            self.history.record_round_update(
                self.chicken_player.get_location(),
                move_type,
                self.chicken_player.get_eggs_laid(),
                self.chicken_enemy.get_eggs_laid(),
                self.chicken_player.get_turds_left(),
                self.chicken_enemy.get_turds_left(),
                self.player_time,
                self.enemy_time,
                self.turns_left_player,
                self.turns_left_enemy,
                self.is_as_turn,
            )
            
        self.is_as_turn = not self.is_as_turn

    def check_win(self, timeout_bounds: float = 0.5):
        """
        Checks if the game has been won and sets the winner accordingly.

        Parameters:
            timeout_bounds (float, optional): The time threshold in seconds for determining timeout ties. Defaults to 0.5.
        """
        if self.player_time <= 0:
            if self.enemy_time <= timeout_bounds:
                self.set_winner(Result.TIE, WinReason.TIMEOUT)
            else:
                self.set_winner(Result.ENEMY, WinReason.TIMEOUT)
        elif self.enemy_time <= 0:
            if self.player_time <= timeout_bounds:
                self.set_winner(Result.TIE, WinReason.TIMEOUT)
            else:
                self.set_winner(Result.PLAYER, WinReason.TIMEOUT)
        elif (self.turns_left_player == 0 and self.turns_left_enemy == 0) or self.turn_count >= 2 * self.MAX_TURNS:
            if self.chicken_player.get_eggs_laid() < self.chicken_enemy.get_eggs_laid():
                self.set_winner(Result.ENEMY, WinReason.EGGS_LAID)
            elif (
                self.chicken_player.get_eggs_laid() > self.chicken_enemy.get_eggs_laid()
            ):
                self.set_winner(Result.PLAYER, WinReason.EGGS_LAID)
            else:
                self.set_winner(Result.TIE, WinReason.EGGS_LAID)
        elif self.chicken_blocked:
            # seperate condition in case we want to change outcome of blocking win
            if self.chicken_player.get_eggs_laid() < self.chicken_enemy.get_eggs_laid():
                self.set_winner(Result.ENEMY, WinReason.BLOCKING_END)
            elif (
                self.chicken_player.get_eggs_laid() > self.chicken_enemy.get_eggs_laid()
            ):
                self.set_winner(Result.PLAYER, WinReason.BLOCKING_END)
            else:
                self.set_winner(Result.TIE, WinReason.BLOCKING_END)

    def is_game_over(self):
        """
        Checks if the game is over.

        Returns:
            (bool): True if the game is over, False otherwise.
        """
        return self.winner is not None

    def forecast_move(
        self,
        dir: Direction | int,
        move_type: MoveType | int,
        check_ok: bool = True,
    ):
        """
        Creates a copy of the board with a forecasted move applied.

        Parameters:
            dir (Direction | int): The direction of the move.
            move_type (MoveType | int): The type of move (walk, egg, turd).
            check_ok (bool, optional): If True, validates the move before applying; if False, skips validation. Defaults to True.

        Returns:
            (Board): A new Board object with the move applied, or None if the move is invalid.
        """
        board_copy = self.get_copy()
        ok = board_copy.apply_move(dir, move_type, check_ok)
        return board_copy if ok else None

    def set_found_trapdoors(self, found_trapdoors):
        """
        Sets a trapdoor at the specified location and updates sampling masks if randomization is enabled.

        Parameters:
            x (int): The x-coordinate for the trapdoor.
            y (int): The y-coordinate for the trapdoor.
        """
        self.found_trapdoors = set(found_trapdoors)

    def get_copy(
        self,
        build_history: bool = False,
        asymmetric: bool = False,
    ):
        """
        Creates a deep copy of the current board.

        Parameters:
            build_history (bool, optional): Whether the copy should track history. Defaults to False.

        Returns:
            (Board): A new Board object with the same state as the current one.
        """
        board_copy = Board(self.game_map, build_history=build_history, copy=True)

        board_copy.eggs_player = self.eggs_player.copy()
        board_copy.eggs_enemy = self.eggs_enemy.copy()
        board_copy.turds_player = self.turds_player.copy()
        board_copy.turds_enemy = self.turds_enemy.copy()
        board_copy.found_trapdoors = self.found_trapdoors.copy()
        board_copy.is_as_turn = self.is_as_turn

        board_copy.chicken_player = self.chicken_player.get_copy()
        board_copy.chicken_enemy = self.chicken_enemy.get_copy()

        # game metadata
        board_copy.turn_count = self.turn_count

        board_copy.MAX_TURNS = self.MAX_TURNS
        board_copy.turns_left_player = self.turns_left_player
        board_copy.turns_left_enemy = self.turns_left_enemy

        # more game metadata
        board_copy.winner = self.winner

        board_copy.time_to_play = self.time_to_play
        board_copy.player_time = self.player_time
        board_copy.enemy_time = self.enemy_time
        board_copy.win_reason = self.win_reason
        board_copy.chicken_blocked = self.chicken_blocked

        # history building
        board_copy.build_history = build_history
        if build_history:
            board_copy.history = self.history

        return board_copy

    def has_moves_left(self, enemy=False):
        """
        Checks if the player has any valid moves remaining.

        Returns:
            (bool): True if there are valid moves remaining, False otherwise.
        """

        return len(self.get_valid_moves(enemy=enemy)) > 0

    def set_build_history(self, build_history: bool):
        """
        Sets whether the history of the game should be recorded.

        Parameters:
            build_history (bool): Whether to track the game history. True to record, False to not record.
        """

        self.build_history = build_history

    def set_winner(self, result: Result, reason: WinReason):
        """
        Sets the winner and the reason for the game's outcome.

        Parameters:
            result (Result): The winner of the game.
            reason (WinReason): The reason for the outcome.
        """

        self.winner = result
        self.win_reason = reason

    def get_winner(self) -> Result:
        """
        Returns the winner of the game.

        Returns:
            (Result): The winner of the game.
        """

        return self.winner

    def get_win_reason(self) -> str:
        """
        Returns the string explaining the reason why the game was won.

        Returns:
            (str): The reason for the game's outcome.
        """
        return self.win_reason

    def get_history(self) -> dict:
        """
        Get a dictionary representation for the renderer.

        Returns:
            (dict): A dictionary representing the game history.
        """
        return self.history

    def reverse_perspective(self):
        """
        Reverses the perspective from player to enemy or vice versa.
        This swaps all player and enemy references internally.
        """

        self.eggs_player, self.eggs_enemy = (
            self.eggs_enemy,
            self.eggs_player,
        )
        self.turds_player, self.turds_enemy = (
            self.turds_enemy,
            self.turds_player,
        )

        self.chicken_player, self.chicken_enemy = (
            self.chicken_enemy,
            self.chicken_player,
        )

        self.player_time, self.enemy_time = self.enemy_time, self.player_time
        self.turns_left_player, self.turns_left_enemy = self.turns_left_enemy, self.turns_left_player
