from enum import IntEnum, auto
from typing import List, Tuple


class Result(IntEnum):
    PLAYER = 0
    ENEMY = 1
    TIE = 2
    ERROR = 3


class ResultArbiter(IntEnum):
    PLAYER_A = 0
    PLAYER_B = 1
    TIE = 2
    ERROR = 3


class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


def loc_after_direction(loc: Tuple[int, int], dir: Direction) -> Tuple[int, int]:
    x, y = loc
    if dir == Direction.UP:
        return (x, y - 1)
    elif dir == Direction.DOWN:
        return (x, y + 1)
    elif dir == Direction.LEFT:
        return (x - 1, y)
    elif dir == Direction.RIGHT:
        return (x + 1, y)
    else:
        raise ValueError("Invalid direction")
        return (0, 0)


class MoveType(IntEnum):
    PLAIN = 0
    EGG = 1
    TURD = 2


class Cell(IntEnum):
    SPACE = 0
    PLAYER_A_EGG = 1
    PLAYER_A_TURD = 2
    PLAYER_B_EGG = 3
    PLAYER_B_TURD = 4
    TRAPDOOR = 5


class WinReason(IntEnum):
    EGGS_LAID = 0
    BLOCKING_END = 1
    TIMEOUT = 2
    INVALID_TURN = 3
    CODE_CRASH = 4
    MEMORY_ERROR = 5
    FAILED_INIT = 6
