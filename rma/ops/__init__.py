"""Locally scoped research operations (Algorithm 1's u)."""
from .base import (  # noqa: F401
    FakeBackend,
    Operation,
    OpContext,
    REGISTRY,
    get_operation,
    register,
)
from . import critic as _critic  # noqa: F401  (registers "critic")
from . import solver as _solver  # noqa: F401  (registers "solver")
from . import literature as _literature  # noqa: F401  (registers "literature")
from . import meeting as _meeting  # noqa: F401  (registers "meeting")
from . import revise as _revise  # noqa: F401  (registers "revise")
from . import evaluator as _evaluator  # noqa: F401  (registers "evaluator")
from . import concepts as _concepts  # noqa: F401  (registers "concepts")

__all__ = ["Operation", "OpContext", "FakeBackend", "REGISTRY", "register", "get_operation"]
