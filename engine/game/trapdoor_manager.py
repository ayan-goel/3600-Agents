from typing import List, Tuple

import numpy as np

from .chicken import Chicken
from .enums import Cell, Direction, MoveType, Result, WinReason
from .game_map import GameMap, prob_feel, prob_hear
from .history import History


def delta_locs(loc_a: Tuple[int, int], loc_b: Tuple[int, int]) -> Tuple[int, int]:
    ax, ay = loc_a
    bx, by = loc_b
    return (abs(ax - bx), abs(ay - by))


def choose_trapdoor(weights, parity) -> Tuple[int, int]:
    dim = weights.shape[0]
    flattened = weights.flatten()
    i = 1
    j = parity
    while (i + j) % 2 != parity:
        indx = np.random.choice(len(flattened), p=flattened)
        i = indx // dim
        j = indx % dim
    return (i, j)


class TrapdoorManager:
    def __init__(self, game_map: GameMap):
        self.game_map = game_map
        self.spawns = []
        self.trapdoors = []

    # choose_spawns should be called before choose trapdoor_weights
    def choose_spawns(self):
        if len(self.spawns) > 0:
            print("ERROR: choose_spawns called twice")
        dim = self.game_map.MAP_SIZE

        # Left edge or right
        if np.random.random() < 0.5:
            x1 = 0
            x2 = dim - 1
        else:
            x1 = dim - 1
            x2 = 0

        # Not a corner
        y1 = np.random.randint(1, dim - 1)
        y2 = y1

        self.spawns = [(x1, y1), (x2, y2)]

        return self.spawns.copy()

    def choose_trapdoors(self):
        if len(self.trapdoors) > 0:
            print("ERROR: choose_trapdoors called twice")

        dim = self.game_map.MAP_SIZE
        unnormalized = np.zeros((dim, dim))
        unnormalized[2 : dim - 2, 2 : dim - 2] = 1.0
        unnormalized[3 : dim - 3, 3 : dim - 3] = 2.0
        normalized = unnormalized / np.sum(unnormalized)
        even_trapdoor = choose_trapdoor(normalized, 0)
        odd_trapdoor = choose_trapdoor(normalized, 1)
        self.trapdoors = [even_trapdoor, odd_trapdoor]
        return self.trapdoors.copy()

    def sample_trapdoors(self, loc: Tuple[int, int]) -> List[Tuple[bool, bool]]:
        result = []
        for door_indx in range(2):
            door = self.trapdoors[door_indx]
            delta = delta_locs(loc, door)
            hear_p = prob_hear(delta[0], delta[1])
            did_hear = np.random.rand() < hear_p
            feel_p = prob_feel(delta[0], delta[1])
            did_feel = np.random.rand() < feel_p
            result.append((did_hear, did_feel))
        return result

    def is_trapdoor(self, loc: Tuple[int, int]) -> bool:
        return loc in self.trapdoors

    def get_trapdoors(self):
        return self.trapdoors.copy()

    def get_spawns(self):
        return self.spawns.copy()
