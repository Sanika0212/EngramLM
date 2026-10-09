"""The memory state carried between segments.

``MemoryState`` is an explicit value, not hidden module state: a model call takes a
state and returns a new one. That keeps evaluation deterministic and independent
of call order, and it lets every batch row (stream) own a separate memory bank.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

_TENSOR_FIELDS = ("vectors", "priority", "age", "valid", "origin_segment", "origin_position")


@dataclass(frozen=True)
class MemoryState:
    """Per-stream memory bank.

    Shapes use ``B`` for streams (batch rows), ``C`` for capacity, ``D`` for width.

    Attributes:
        vectors: ``(B, C, D)`` stored hidden states.
        priority: ``(B, C)`` write priority in ``(0, 1]``. May carry gradient so a
            learned salience scorer can be trained through later reads.
        age: ``(B, C)`` number of segments since the slot was written.
        valid: ``(B, C)`` boolean occupancy mask.
        origin_segment: ``(B, C)`` index of the segment that wrote the slot.
        origin_position: ``(B, C)`` absolute token position of the written token.
        segments_seen: number of segments that have been written so far.
    """

    vectors: Tensor
    priority: Tensor
    age: Tensor
    valid: Tensor
    origin_segment: Tensor
    origin_position: Tensor
    segments_seen: int = 0

    @classmethod
    def empty(
        cls,
        batch_size: int,
        capacity: int,
        dim: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> MemoryState:
        zeros = torch.zeros(batch_size, capacity, device=device)
        return cls(
            vectors=torch.zeros(batch_size, capacity, dim, device=device, dtype=dtype),
            priority=zeros.clone(),
            age=zeros.clone(),
            valid=torch.zeros(batch_size, capacity, dtype=torch.bool, device=device),
            origin_segment=torch.full((batch_size, capacity), -1, dtype=torch.long, device=device),
            origin_position=torch.full((batch_size, capacity), -1, dtype=torch.long, device=device),
        )

    @property
    def batch_size(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def capacity(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[2])

    @property
    def device(self) -> torch.device:
        return self.vectors.device

    def num_valid(self) -> Tensor:
        """``(B,)`` number of occupied slots per stream."""
        return self.valid.sum(dim=1)

    def is_empty(self) -> bool:
        return self.capacity == 0 or not bool(self.valid.any())

    def detach(self) -> MemoryState:
        """Cut the autograd graph (used at truncated-BPTT window boundaries)."""
        return replace(self, vectors=self.vectors.detach(), priority=self.priority.detach())

    def to(self, device: torch.device | str) -> MemoryState:
        return replace(self, **{name: getattr(self, name).to(device) for name in _TENSOR_FIELDS})

    def select_streams(self, rows: Tensor) -> MemoryState:
        """Keep only the given batch rows (for example, finished vs. running streams)."""
        return replace(self, **{name: getattr(self, name)[rows] for name in _TENSOR_FIELDS})

    def save(self, path: str | Path) -> None:
        tensors = {name: getattr(self, name).detach().contiguous().cpu() for name in _TENSOR_FIELDS}
        save_file(tensors, str(path), metadata={"segments_seen": str(self.segments_seen)})

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str | None = None) -> MemoryState:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as handle:
            metadata = handle.metadata() or {}
        tensors = load_file(str(path), device=str(device) if device is not None else "cpu")
        return cls(**tensors, segments_seen=int(metadata.get("segments_seen", 0)))
