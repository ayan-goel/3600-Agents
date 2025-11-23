from __future__ import annotations

import importlib
from typing import Any, List

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


__all__ = ["PlayerAgent"]



