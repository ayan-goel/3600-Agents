from typing import Tuple

from game.enums import Direction
from game.game_map import prob_feel, prob_hear


class Chicken:
    """
    This class represents a chicken.
    """

    def __init__(self, max_turds: int = 0, copy: bool = False):
        """
        Initializes the Chicken object with copy flag.

        Parameters:
            max_turds (int, optional): Maximum number of turds the chicken can place. Defaults to 0.
            copy (bool, optional): If True, initializes as an empty copy; if False, initializes with default values. Defaults to False.
        """

        if not copy:
            self.turds_left = max_turds
            self.eggs_laid = 0

    def start(self, start_loc: Tuple[int, int], even_chicken: int):
        """
        Initializes the chicken's starting position and parity.

        Parameters:
            start_loc (Tuple[int, int]): The (x, y) coordinates of the starting location.
            even_chicken (int): The parity indicator (0 or 1) that determines which cells the chicken can lay eggs on.
        """
        self.spawn = start_loc
        self.loc = start_loc
        self.even_chicken = even_chicken

    def is_player_a(self) -> bool:
        return self.even_chicken == 0

    def get_spawn(self) -> Tuple[int, int]:
        """
        Returns a copy of the chicken's spawn location.

        Returns:
            Tuple[int, int]: containing the (x, y) coordinates of the spawn location.
        """
        return self.spawn

    def get_location(self) -> Tuple[int, int]:
        """
        Returns a copy of the chicken's current location.

        Returns:
            (Tuple[int, int]): tuple containing the (x, y) coordinates of the current location.
        """
        return self.loc

    def increment_eggs_laid(self, eggs=1):
        """
        Increments the count of eggs laid by the chicken.

        Parameters:
            eggs (int, optional): Number of eggs to add to the count. Defaults to 1.
        """
        self.eggs_laid += eggs

    def reset_location(self):
        """
        Resets the chicken's location back to its spawn point.
        """
        self.loc = self.spawn

    def can_lay_egg(self, loc: Tuple[int, int]):
        """
        Checks if the chicken can lay an egg at the specified location based on parity.

        Parameters:
            x (int): The x-coordinate of the location.
            y (int): The y-coordinate of the location.

        Returns:
            (bool): True if the chicken can lay an egg at that location, False otherwise.
        """
        return (loc[0] + loc[1]) % 2 == self.even_chicken

    def decrement_turds(self):
        """
        Decrements the count of turds remaining.
        """
        self.turds_left -= 1

    def get_turds_left(self) -> int:
        """
        Returns the number of turds remaining.

        Returns:
            (int): Number of turds left.
        """
        return self.turds_left

    def get_turds_placed(self) -> int:
        """
        Returns the number of turds that have been placed.

        Returns:
            (int): Number of turds placed.
        """
        return 5 - self.turds_left

    def get_eggs_laid(self):
        """
        Returns the number of eggs laid by the chicken.

        Returns:
            (int): Number of eggs laid.
        """
        return self.eggs_laid

    def get_next_loc(self, dir=Direction | int, loc=None) -> Tuple[int, int]:
        """
        Returns the next location if the chicken moves in the specified direction. Does not apply the move.

        Parameters:
            dir (Direction | int): The direction to move.
            loc (Tuple[int, int]): The starting location. If None, uses current location. Defaults to None.

        Returns:
            (np.ndarray): Array containing the (x, y) coordinates of the next location, or None if direction is invalid.
        """
        loc = self.loc if (loc is None) else loc

        match dir:
            case Direction.UP:
                loc = (loc[0], loc[1] - 1)
            case Direction.RIGHT:
                loc = (loc[0] + 1, loc[1])
            case Direction.DOWN:
                loc = (loc[0], loc[1] + 1)
            case Direction.LEFT:
                loc = (loc[0] - 1, loc[1])

            case _:
                return None

        return loc

    def apply_dir(self, dir=Direction | int) -> Tuple[int, int]:
        """
        Applies a direction to move the chicken and updates its location.

        Parameters:
            dir (Direction | int): The direction to move.

        Returns:
            (Tuple[int, int]): Array containing the new (x, y) coordinates, or None if direction is invalid.
        """
        self.loc = self.get_next_loc(dir)
        return self.loc

    def lay_egg(self):
        """
        Lays an egg at the chicken's current location and increments the egg count.

        Returns:
            (Tuple[int, int]): Array containing the (x, y) coordinates where the egg was laid.
        """
        self.eggs_laid += 1
        return self.loc

    def drop_turd(self):
        """
        Drops a turd at the chicken's current location and decrements the turd count.

        Returns:
            (Tuple[int, int]): Array containing the (x, y) coordinates where the turd was dropped.
        """
        self.turds_left -= 1
        return self.loc

    def has_turds_left(self):
        """
        Checks if the chicken has turds remaining.

        Returns:
            (bool): True if there are turds remaining, False otherwise.
        """
        return self.turds_left > 0

    def prob_senses_if_trapdoor_were_at(
        self, did_hear: bool, did_feel: bool, x: int, y: int
    ):
        """
        Returns a tuple representing the probability of the player (hearing, feeling) the trapdoor if it were at x,y.
        """
        delta_x = abs(x - self.loc[0])
        delta_y = abs(y - self.loc[1])
        if did_hear:
            hear_likelihood = prob_hear(delta_x, delta_y)
        else:
            hear_likelihood = 1.0 - prob_hear(delta_x, delta_y)
        if did_feel:
            feel_likelihood = prob_feel(delta_x, delta_y)
        else:
            feel_likelihood = 1.0 - prob_feel(delta_x, delta_y)
        return (hear_likelihood, feel_likelihood)

    def get_copy(self) -> "Chicken":
        """
        Return a deep copy of the chicken.

        Returns:
            (Chicken): A deep copy of the current chicken object.
        """

        new_chicken = Chicken(copy=True)

        new_chicken.even_chicken = self.even_chicken
        new_chicken.eggs_laid = self.eggs_laid
        new_chicken.turds_left = self.turds_left
        new_chicken.loc = self.loc
        new_chicken.spawn = self.spawn

        return new_chicken
