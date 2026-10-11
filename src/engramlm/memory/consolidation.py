"""Sleep-phase consolidation: merge near-duplicate memories.

Between segments the bank can be compressed by greedily merging the most similar
pair of slots while their cosine similarity exceeds a threshold. A merged group keeps
the priority-weighted average of its vectors, the sum of its priorities (so total
priority mass is conserved), its youngest age, and the provenance of its strongest
slot. Freed slots become available for new writes at a fixed memory budget.

Which slots merge is decided without gradient (it is a discrete choice), but the merge
itself is applied as a differentiable weighted average of the original vectors and
priorities. Later reads therefore still train the write path and the salience scorer
through consolidated memories; detaching them would cut that signal every segment.
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
def _plan(
    memory: MemoryState, threshold: float, max_merges: int | None
) -> tuple[Tensor, Tensor, Tensor]:
    """Greedy merge plan: weights ``(B, S, S)``, membership ``(B, S, S)``, merges ``(B,)``.

    Row ``i`` of ``weights`` holds the normalised priority weights of the slots merged
    into slot ``i``; ``membership`` holds ones instead of weights. Rows of slots merged
    away are zero; untouched slots keep an identity row.
    """
    batch, slots_n = memory.priority.shape
    device = memory.device
    eye = torch.eye(slots_n, device=device).expand(batch, slots_n, slots_n)
    weights = eye.clone()
    membership = eye.clone()
    merges = torch.zeros(batch, dtype=torch.long, device=device)
    for b in range(batch):
        slots = memory.valid[b].nonzero().squeeze(1)
        n = slots.numel()
        if n < 2:
            continue
        vec = memory.vectors[b, slots].float().clone()
        pri = memory.priority[b, slots].float().clone()
        groups = [[k] for k in range(n)]
        unit = F.normalize(vec, dim=-1)
        sim = unit @ unit.T
        sim.fill_diagonal_(float("-inf"))
        alive = torch.ones(n, dtype=torch.bool, device=device)
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
            groups[i] += groups[j]
            groups[j] = []
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
        base = memory.priority[b, slots].float()
        for i, group in enumerate(groups):
            si = slots[i]
            weights[b, si] = 0.0
            membership[b, si] = 0.0
            if not group:
                continue
            members = slots[torch.tensor(group, device=device)]
            share = base[group].clamp_min(0.0)
            total = share.sum()
            share = share / total if float(total) > 0 else torch.full_like(share, 1 / len(group))
            weights[b, si, members] = share
            membership[b, si, members] = 1.0
    return weights, membership, merges


def consolidate(
    memory: MemoryState, threshold: float, max_merges: int | None = None
) -> tuple[MemoryState, ConsolidationStats]:
    """Merge slots whose cosine similarity is at least ``threshold``."""
    weights, membership, merges = _plan(memory, threshold, max_merges)
    vectors = torch.einsum("bij,bjd->bid", weights.to(memory.vectors.dtype), memory.vectors)
    priority = torch.einsum("bij,bj->bi", membership.to(memory.priority.dtype), memory.priority)
    kept = membership.sum(-1) > 0  # slots that still hold a memory (merged ones are zero rows)
    valid = memory.valid & kept
    age = memory.age.clone()
    origin_segment = memory.origin_segment.clone()
    origin_position = memory.origin_position.clone()
    with torch.no_grad():
        member_age = torch.where(membership > 0, memory.age.unsqueeze(1), torch.inf)
        youngest = member_age.min(-1).values
        age = torch.where(valid, youngest.to(age.dtype), age)
        dead = memory.valid & ~kept
        age[dead] = 0.0
        origin_segment[dead] = -1
        origin_position[dead] = -1
    vectors = torch.where(dead.unsqueeze(-1), torch.zeros_like(vectors), vectors)
    priority = torch.where(dead, torch.zeros_like(priority), priority)
    state = MemoryState(
        vectors=vectors,
        priority=priority,
        age=age,
        valid=valid,
        origin_segment=origin_segment,
        origin_position=origin_position,
        segments_seen=memory.segments_seen,
    )
    return state, ConsolidationStats(merges=merges)
