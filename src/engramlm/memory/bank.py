"""Vectorized writes into a :class:`MemoryState`.

A write is two top-k operations and a gather, with no Python loops over tokens
and no host synchronisation:

1. per stream, pick the ``budget`` highest-priority candidate tokens;
2. age the existing slots by one segment and append the new candidates;
3. keep the ``capacity`` entries with the highest retention score.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from engramlm.memory.policies import RetentionFn
from engramlm.memory.state import MemoryState

_PRIORITY_FLOOR = 1e-6


@dataclass(frozen=True)
class WriteResult:
    """New memory plus which candidate positions were written (for inspection)."""

    memory: MemoryState
    written: Tensor  # (B, T) bool: candidate tokens that entered the bank this step


def write(
    memory: MemoryState,
    candidates: Tensor,
    priority: Tensor,
    *,
    budget: int,
    retention: RetentionFn,
    mask: Tensor | None = None,
    positions: Tensor | None = None,
) -> WriteResult:
    """Write up to ``budget`` tokens per stream and evict down to capacity.

    Args:
        memory: current state, ``B`` streams with capacity ``C``.
        candidates: ``(B, T, D)`` hidden states of the segment just processed.
        priority: ``(B, T)`` salience in ``(0, 1]``; may require grad.
        budget: maximum tokens written per stream.
        retention: eviction policy (see :mod:`engramlm.memory.policies`).
        mask: ``(B, T)`` bool, ``False`` for padding tokens that must never be written.
        positions: ``(B, T)`` absolute token positions, stored for inspection.
    """
    batch, seq_len, dim = candidates.shape
    if memory.batch_size != batch or memory.dim != dim:
        raise ValueError(
            f"memory is ({memory.batch_size}, {memory.capacity}, {memory.dim}) "
            f"but candidates are {tuple(candidates.shape)}"
        )
    device = candidates.device
    if mask is None:
        mask = torch.ones(batch, seq_len, dtype=torch.bool, device=device)
    if positions is None:
        positions = torch.arange(seq_len, device=device).expand(batch, seq_len)

    aged = memory.age + memory.valid.to(memory.age.dtype)
    k = min(budget, seq_len)
    if memory.capacity == 0 or k == 0:
        nothing = torch.zeros(batch, seq_len, dtype=torch.bool, device=device)
        state = MemoryState(
            vectors=memory.vectors,
            priority=memory.priority,
            age=aged,
            valid=memory.valid,
            origin_segment=memory.origin_segment,
            origin_position=memory.origin_position,
            segments_seen=memory.segments_seen + 1,
        )
        return WriteResult(memory=state, written=nothing)

    priority = priority.clamp_min(_PRIORITY_FLOOR)
    selection_score = priority.detach().masked_fill(~mask, float("-inf"))
    top_score, top_idx = selection_score.topk(k, dim=1)
    new_valid = torch.isfinite(top_score)

    new_vectors = candidates.gather(1, top_idx.unsqueeze(-1).expand(batch, k, dim))
    new_priority = priority.gather(1, top_idx)
    new_position = positions.gather(1, top_idx)

    all_vectors = torch.cat([memory.vectors.to(candidates.dtype), new_vectors], dim=1)
    all_priority = torch.cat([memory.priority.to(new_priority.dtype), new_priority], dim=1)
    all_age = torch.cat([aged, torch.zeros(batch, k, device=device, dtype=aged.dtype)], dim=1)
    all_valid = torch.cat([memory.valid, new_valid], dim=1)
    all_segment = torch.cat(
        [memory.origin_segment, torch.full_like(top_idx, memory.segments_seen)], dim=1
    )
    all_position = torch.cat([memory.origin_position, new_position], dim=1)

    score = retention(all_priority, all_age).masked_fill(~all_valid, float("-inf"))
    # Break exact ties in favour of newer entries so FIFO and equal priorities are stable.
    score = score + 1e-9 * torch.arange(score.shape[1], device=device, dtype=score.dtype)
    keep_score, keep = score.topk(memory.capacity, dim=1)

    state = MemoryState(
        vectors=all_vectors.gather(1, keep.unsqueeze(-1).expand(batch, memory.capacity, dim)),
        priority=all_priority.gather(1, keep),
        age=all_age.gather(1, keep),
        valid=all_valid.gather(1, keep) & torch.isfinite(keep_score),
        origin_segment=all_segment.gather(1, keep),
        origin_position=all_position.gather(1, keep),
        segments_seen=memory.segments_seen + 1,
    )

    # Map kept slots that came from this write back to their candidate token index.
    kept_new = (keep >= memory.capacity) & state.valid
    source = top_idx.gather(1, (keep - memory.capacity).clamp_min(0))
    hits = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
    hits.scatter_add_(1, source, kept_new.long())
    return WriteResult(memory=state, written=hits > 0)
