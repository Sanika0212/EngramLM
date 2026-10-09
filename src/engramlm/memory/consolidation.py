"""Sleep-phase consolidation: merge near-duplicate memories.

Between segments the bank can be compressed by greedily merging the most similar
pair of slots while their cosine similarity exceeds a threshold. A merge keeps a
priority-weighted average vector, sums the priorities (so total priority mass is
conserved), keeps the younger age, and inherits provenance from the stronger
slot. Freed slots become available for new writes at a fixed memory budget.

Consolidation is a discrete, offline step and runs without gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from engramlm.memory.state import MemoryState


@dataclass(frozen=True)
class ConsolidationStats:
    merges: Tensor  # (B,) number of merges per stream


@torch.no_grad()
def consolidate(
    memory: MemoryState, threshold: float, max_merges: int | None = None
) -> tuple[MemoryState, ConsolidationStats]:
    """Merge slots whose cosine similarity is at least ``threshold``."""
    vectors = memory.vectors.detach().clone()
    priority = memory.priority.detach().clone()
    age = memory.age.clone()
    valid = memory.valid.clone()
    origin_segment = memory.origin_segment.clone()
    origin_position = memory.origin_position.clone()
    merges = torch.zeros(memory.batch_size, dtype=torch.long, device=memory.device)

    for b in range(memory.batch_size):
        slots = valid[b].nonzero().squeeze(1)
        n = slots.numel()
        if n < 2:
            continue
        vec = vectors[b, slots]
        pri = priority[b, slots]
        unit = F.normalize(vec, dim=-1)
        sim = unit @ unit.T
        sim.fill_diagonal_(float("-inf"))
        alive = torch.ones(n, dtype=torch.bool, device=memory.device)
        budget = n - 1 if max_merges is None else min(max_merges, n - 1)

        for _ in range(budget):
            flat = int(sim.argmax())
            i, j = divmod(flat, n)
            if float(sim[i, j]) < threshold:
                break
            if pri[j] > pri[i]:  # keep provenance of the stronger slot in i
                i, j = j, i
            weight = (pri[i] + pri[j]).clamp_min(1e-12)
            vec[i] = (pri[i] * vec[i] + pri[j] * vec[j]) / weight
            pri[i] = pri[i] + pri[j]
            si, sj = slots[i], slots[j]
            age[b, si] = torch.minimum(age[b, si], age[b, sj])
            valid[b, sj] = False
            alive[j] = False
            unit[i] = F.normalize(vec[i], dim=-1)
            row = unit @ unit[i]
            row[~alive] = float("-inf")
            row[i] = float("-inf")
            sim[i, :] = row
            sim[:, i] = row
            sim[j, :] = float("-inf")
            sim[:, j] = float("-inf")
            merges[b] += 1

        vectors[b, slots] = vec
        priority[b, slots] = pri
        dead = slots[~alive]
        vectors[b, dead] = 0.0
        priority[b, dead] = 0.0
        age[b, dead] = 0.0
        origin_segment[b, dead] = -1
        origin_position[b, dead] = -1

    state = MemoryState(
        vectors=vectors.to(memory.vectors.dtype),
        priority=priority,
        age=age,
        valid=valid,
        origin_segment=origin_segment,
        origin_position=origin_position,
        segments_seen=memory.segments_seen,
    )
    return state, ConsolidationStats(merges=merges)
