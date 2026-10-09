"""Run long documents through an EngramLM one segment at a time."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from engramlm.memory import MemoryState
from engramlm.model import EngramLM


def segment_bounds(length: int, segment_len: int) -> list[tuple[int, int]]:
    """``[start, end)`` bounds that tile ``length`` tokens in ``segment_len`` chunks."""
    if segment_len < 2:
        raise ValueError("segment_len must be at least 2")
    return [(start, min(start + segment_len, length)) for start in range(0, length, segment_len)]


def token_nll_sum(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Summed next-token NLL (nats) and the number of scored tokens."""
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    nll = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return nll, (shift_labels != -100).sum()


@dataclass
class DocumentResult:
    nll_sum: float
    n_tokens: int
    memory: MemoryState
    segment_nll: list[float] = field(default_factory=list)
    logits: list[Tensor] = field(default_factory=list)

    @property
    def mean_nll(self) -> float:
        return self.nll_sum / max(self.n_tokens, 1)


@torch.no_grad()
def evaluate_document(
    model: EngramLM,
    input_ids: Tensor,
    *,
    segment_len: int,
    memory: MemoryState | None = None,
    score_mask: Tensor | None = None,
    keep_logits: bool = False,
    generator: torch.Generator | None = None,
) -> DocumentResult:
    """Score a batch of documents segment by segment, carrying memory forward.

    Args:
        input_ids: ``(B, L)`` token ids.
        segment_len: tokens per segment (each segment is one base-model forward).
        memory: starting memory (a fresh one if ``None``).
        score_mask: ``(B, L)`` bool; only these target tokens count towards the NLL.
        keep_logits: also return each segment's logits (on CPU).
    """
    batch = input_ids.shape[0]
    memory = memory if memory is not None else model.new_memory(batch, input_ids.device)
    nll_total = torch.zeros((), device=input_ids.device)
    count_total = torch.zeros((), dtype=torch.long, device=input_ids.device)
    result = DocumentResult(nll_sum=0.0, n_tokens=0, memory=memory)

    for start, end in segment_bounds(input_ids.shape[1], segment_len):
        segment = input_ids[:, start:end]
        labels = segment.clone()
        if score_mask is not None:
            labels[~score_mask[:, start:end]] = -100
        out = model(
            segment, memory=memory, position_offset=start, generator=generator, write_memory=True
        )
        nll, count = token_nll_sum(out.logits, labels)
        nll_total += nll
        count_total += count
        result.segment_nll.append(float(nll / count.clamp_min(1)))
        if keep_logits:
            result.logits.append(out.logits.detach().cpu())
        memory = out.memory

    result.nll_sum = float(nll_total)
    result.n_tokens = int(count_total)
    result.memory = memory
    return result
