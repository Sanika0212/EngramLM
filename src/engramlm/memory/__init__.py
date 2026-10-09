"""External memory: state container, vectorized writes, eviction, consolidation."""

from engramlm.memory.bank import WriteResult, write
from engramlm.memory.policies import build_policy, fifo_retention, priority_retention
from engramlm.memory.state import MemoryState

__all__ = [
    "MemoryState",
    "WriteResult",
    "build_policy",
    "fifo_retention",
    "priority_retention",
    "write",
]
