from collections.abc import Callable
from time import sleep
from typing import List, Set, Tuple

import numpy as np
from game import *

"""
Melvin is the dumbest agent of all. He randomly selects a move from the list of valid moves.
"""


class PlayerAgent:
    """
    /you may add functions, however, __init__ and play are the entry points for
    your program and should not be changed.
    """

    def __init__(self, board: board.Board, time_left: Callable):
        pass

    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        location = board.chicken_player.get_location()
        print(f"I'm at {location}.")
        print(f"Trapdoor A: heard? {sensor_data[0][0]}, felt? {sensor_data[0][1]}")
        print(f"Trapdoor B: heard? {sensor_data[1][0]}, felt? {sensor_data[1][1]}")
        print(f"Starting to think with {time_left()} seconds left.")
        # Not really thinking; Yolanda is not a deep thinker
        sleep(1.5)
        moves = board.get_valid_moves()
        result = moves[np.random.randint(len(moves))]
        print(f"I have {time_left()} seconds left. Playing {result}.")
        return result
