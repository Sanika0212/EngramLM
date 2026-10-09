"""External memory: state container, vectorized writes, eviction, consolidation."""

from engramlm.memory.bank import WriteResult, write
from engramlm.memory.consolidation import ConsolidationStats, consolidate
from engramlm.memory.policies import build_policy, fifo_retention, priority_retention
from engramlm.memory.state import MemoryState

__all__ = [
    "ConsolidationStats",
    "MemoryState",
    "WriteResult",
    "build_policy",
    "consolidate",
    "fifo_retention",
    "priority_retention",
    "write",
]
