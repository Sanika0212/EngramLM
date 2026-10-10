"""Salience scorers decide which tokens are written to memory, and how strongly.

Every scorer maps the segment just processed to a ``(B, T)`` priority in ``(0, 1)``.
Writes happen *after* a segment's logits are computed, and memory written from a
segment is only read by later segments, so scorers may look at the whole segment
without breaking causality.

Built-in scorers (select with ``EngramConfig.scorer``):

- ``surprise``: standardized next-token NLL under the model itself. Parameter-free.
- ``learned``: a small MLP over the hidden state plus the surprise feature,
  initialised to reproduce ``surprise`` exactly and trained through the priority
  bias that later reads apply.
- ``random``: uniform random priorities (a control).
- ``recency``: later tokens in a segment get higher priority (a Transformer-XL-like
  control that carries the tail of each segment forward).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class SalienceContext:
    """Everything a scorer may look at for one segment."""

    hidden: Tensor  # (B, T, D) memory-layer hidden states, before any memory read
    logits: Tensor  # (B, T, V) next-token logits for the segment
    input_ids: Tensor  # (B, T)
    mask: Tensor  # (B, T) bool, False for padding
    generator: torch.Generator | None = None


def token_surprise(logits: Tensor, input_ids: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Per-token negative log-likelihood and the mask of positions where it is defined.

    Position ``t`` is scored by the prediction made at ``t - 1``; the first token of a
    segment (and any token after padding) has no in-segment prediction.
    """
    batch, seq_len = input_ids.shape
    nll = torch.zeros(batch, seq_len, device=logits.device, dtype=torch.float32)
    defined = torch.zeros(batch, seq_len, dtype=torch.bool, device=logits.device)
    if seq_len > 1:
        log_probs = F.log_softmax(logits[:, :-1].detach().float(), dim=-1)
        nll[:, 1:] = -log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        defined[:, 1:] = mask[:, 1:] & mask[:, :-1]
    return nll, defined


def standardize(values: Tensor, defined: Tensor) -> Tensor:
    """Per-row z-score over ``defined`` positions; undefined positions become 0."""
    weight = defined.float()
    count = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (values * weight).sum(dim=1, keepdim=True) / count
    var = ((values - mean) ** 2 * weight).sum(dim=1, keepdim=True) / count
    z = (values - mean) / var.sqrt().clamp_min(1e-6)
    return z * weight


class SalienceScorer(nn.Module, ABC):
    """Base class: ``forward(ctx)`` returns ``(B, T)`` priorities in ``(0, 1)``."""

    @abstractmethod
    def forward(self, ctx: SalienceContext) -> Tensor: ...


class SurpriseScorer(SalienceScorer):
    def forward(self, ctx: SalienceContext) -> Tensor:
        nll, defined = token_surprise(ctx.logits, ctx.input_ids, ctx.mask)
        return torch.sigmoid(standardize(nll, defined))


class LearnedScorer(SalienceScorer):
    """MLP over ``[proj(norm(h)), z_surprise]``; starts out identical to ``surprise``."""

    def __init__(self, hidden_dim: int, proj_dim: int = 64) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, proj_dim)
        self.head = nn.Linear(proj_dim + 1, 1)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.weight[0, -1] = 1.0
            self.head.bias.zero_()

    def forward(self, ctx: SalienceContext) -> Tensor:
        nll, defined = token_surprise(ctx.logits, ctx.input_ids, ctx.mask)
        z = standardize(nll, defined).unsqueeze(-1)
        features = F.gelu(self.proj(self.norm(ctx.hidden.detach().float())))
        logit = self.head(torch.cat([features, z], dim=-1)).squeeze(-1)
        return torch.sigmoid(logit)


class RandomScorer(SalienceScorer):
    def forward(self, ctx: SalienceContext) -> Tensor:
        shape = ctx.input_ids.shape
        if ctx.generator is None:
            return torch.rand(shape, device=ctx.input_ids.device)
        draws = torch.rand(shape, generator=ctx.generator, device=ctx.generator.device)
        return draws.to(ctx.input_ids.device)


class RecencyScorer(SalienceScorer):
    def forward(self, ctx: SalienceContext) -> Tensor:
        seq_len = ctx.input_ids.shape[1]
        ramp = torch.arange(1, seq_len + 1, device=ctx.input_ids.device, dtype=torch.float32)
        return (ramp / seq_len).expand(ctx.input_ids.shape[0], seq_len).clone()


def build_scorer(name: str, hidden_dim: int) -> SalienceScorer:
    if name == "surprise":
        return SurpriseScorer()
    if name == "learned":
        return LearnedScorer(hidden_dim)
    if name == "random":
        return RandomScorer()
    if name == "recency":
        return RecencyScorer()
    raise ValueError(f"Unknown salience scorer {name!r}")
