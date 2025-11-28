from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Callable, List, Tuple

if TYPE_CHECKING:
    from game.board import Board
    from game.enums import Direction, MoveType

_IMPL_MODULE = None


def _load_impl():
    global _IMPL_MODULE
    if _IMPL_MODULE is None:
        _IMPL_MODULE = importlib.import_module("._agent_impl", package=__package__)
    return _IMPL_MODULE


def __getattr__(name: str) -> Any:
    impl = _load_impl()
    value = getattr(impl, name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    impl = _load_impl()
    return sorted(set(globals().keys()) | set(dir(impl)))


class PlayerAgent:
    """Thin proxy that lazily instantiates the heavy Lamine implementation."""

    def __init__(self, board: "Board", time_left: Callable):
        impl = _load_impl()
        self._delegate = impl.PlayerAgent(board, time_left)

    def play(
        self,
        board: "Board",
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable[[], float],
    ) -> Tuple["Direction", "MoveType"]:
        return self._delegate.play(board, sensor_data, time_left)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._delegate, item)


__all__ = ["PlayerAgent"]






