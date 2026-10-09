"""Numerical causality audit for language models and memory wrappers.

A causal LM's logits at position ``t`` may depend only on tokens ``<= t``. The audit
replaces every token after a random cut point with different tokens and measures
how much the logits at or before the cut move. Any non-zero change means the model
can see the future, which silently inflates perplexity.

``audit_causality`` works on any ``input_ids -> logits`` callable;
``audit_engram`` fills an :class:`~engramlm.model.EngramLM` memory first, because a
leak through memory only shows up once memory holds something.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from engramlm.model import EngramLM

LogitsFn = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class AuditReport:
    max_abs_delta: float
    worst_cut: int
    trials: int
    tolerance: float

    @property
    def passed(self) -> bool:
        return self.max_abs_delta <= self.tolerance

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"{verdict}: max |delta logits| at or before the cut = {self.max_abs_delta:.3e} "
            f"(worst cut {self.worst_cut}, {self.trials} trials, tolerance {self.tolerance:g})"
        )


@torch.no_grad()
def audit_causality(
    logits_fn: LogitsFn,
    input_ids: Tensor,
    *,
    vocab_size: int,
    trials: int = 8,
    tolerance: float = 0.0,
    generator: torch.Generator | None = None,
) -> AuditReport:
    """Perturb tokens after random cut points and compare earlier logits."""
    seq_len = input_ids.shape[1]
    if seq_len < 2:
        raise ValueError("need at least two tokens to audit causality")
    gen = generator or torch.Generator().manual_seed(0)
    reference = logits_fn(input_ids)
    worst, worst_cut = 0.0, 0
    for _ in range(trials):
        cut = int(torch.randint(0, seq_len - 1, (1,), generator=gen))
        shift = torch.randint(1, vocab_size, input_ids[:, cut + 1 :].shape, generator=gen)
        perturbed = input_ids.clone()
        perturbed[:, cut + 1 :] = (
            input_ids[:, cut + 1 :] + shift.to(input_ids.device)
        ) % vocab_size
        delta = (logits_fn(perturbed)[:, : cut + 1] - reference[:, : cut + 1]).abs().max().item()
        if delta > worst:
            worst, worst_cut = delta, cut
    return AuditReport(max_abs_delta=worst, worst_cut=worst_cut, trials=trials, tolerance=tolerance)


@contextmanager
def _gate_value(model: EngramLM, value: float | None) -> Iterator[None]:
    if value is None:
        yield
        return
    saved = model.reader.gate.detach().clone()
    with torch.no_grad():
        model.reader.gate.fill_(value)
    try:
        yield
    finally:
        with torch.no_grad():
            model.reader.gate.copy_(saved)


def audit_engram(
    model: EngramLM,
    *,
    vocab_size: int,
    segment_len: int = 32,
    warmup_segments: int = 2,
    batch_size: int = 2,
    trials: int = 8,
    tolerance: float = 0.0,
    open_gate: float | None = 1.0,
    seed: int = 0,
) -> AuditReport:
    """Fill memory with ``warmup_segments`` random segments, then audit the next one.

    ``open_gate`` temporarily sets the reader gate so that memory reads actually
    affect the logits; an untrained model's zero gate would otherwise hide a leak.
    """
    gen = torch.Generator().manual_seed(seed)
    device = model.device

    def sample() -> Tensor:
        return torch.randint(0, vocab_size, (batch_size, segment_len), generator=gen).to(device)

    was_training = model.training
    model.eval()
    try:
        with _gate_value(model, open_gate), torch.no_grad():
            memory = model.new_memory(batch_size, device)
            for _ in range(warmup_segments):
                memory = model(sample(), memory=memory).memory

            def logits_fn(ids: Tensor) -> Tensor:
                logits: Tensor = model(ids, memory=memory, write_memory=False).logits
                return logits

            return audit_causality(
                logits_fn,
                sample(),
                vocab_size=vocab_size,
                trials=trials,
                tolerance=tolerance,
                generator=gen,
            )
    finally:
        model.train(was_training)
