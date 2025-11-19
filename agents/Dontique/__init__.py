from __future__ import annotations

from .trapdoor_belief import TrapdoorBelief

__all__ = ["PlayerAgent", "TrapdoorBelief"]


def __getattr__(name: str):
    if name == "PlayerAgent":
        from .agent import PlayerAgent as _PlayerAgent

        globals()["PlayerAgent"] = _PlayerAgent
        return _PlayerAgent
    raise AttributeError(f"module 'Dontique' has no attribute {name!r}")
