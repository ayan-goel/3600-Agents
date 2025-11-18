import numpy as np

"""
For calculating the probability of hearing a sound based on the distance from the source.
delta_x and delta_y are non-negative
"""


def prob_hear(delta_x: int, delta_y: int):
    if delta_x > 2 or delta_y > 2:
        return 0.0
    if delta_x == 2 and delta_y == 2:
        return 0.0
    if delta_x == 2 or delta_y == 2:
        return 0.1
    if delta_x == 1 and delta_y == 1:
        return 0.25
    if delta_x == 1 or delta_y == 1:
        return 0.5
    return 0.0


def prob_feel(delta_x: int, delta_y: int):
    if delta_x > 1 or delta_y > 1:
        return 0.0
    if delta_x == 1 and delta_y == 1:
        return 0.15
    if delta_x == 1 or delta_y == 1:
        return 0.3
    return 0.0


class GameMap:
    """
    GameMap is an internal utility class used by board to initialize
    constants and store immutable map data for the board. Also used
    to generate trapdoors. Do not use the functions in this class, they
    will not be helpful to you. If you want to use the class variables,
    you may by accessing them through
    the game_board.game_map
    """

    def __init__(self):
        self.MAP_SIZE = 8
        self.MAX_TRAPDOOR_DIST_FROM_CENTER = 2
        self.NUM_TRAPDOORS_PER_TEAM = 1
        self.MAX_TURDS = 5
        self.CORNER_REWARD = 3
        self.TRAPDOOR_PENALTY = -4

    def reflect(self, coords, symmetry):
        """
        Reflects coordinates across the map given a type of symmetry.
        """
        x, y = coords
        match symmetry:
            case 0:  # horizontal
                return (self.MAP_SIZE - 1 - x, y)
            case 1:  # vertical
                return (x, self.MAP_SIZE - 1 - y)
            case 2:  # origin
                return (self.MAP_SIZE - 1 - x, self.MAP_SIZE - 1 - y)
