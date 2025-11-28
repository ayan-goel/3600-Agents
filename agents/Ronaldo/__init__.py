try:
    from .agent import PlayerAgent as Agent
    __all__ = ["Agent"]
except ImportError:
    # Allow importing submodules without game package
    __all__ = []


