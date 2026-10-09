"""Per-token memory reads injected into a decoder layer's residual stream.

Each token issues its own query (no pooling over the sequence) against the slots
of its stream's memory bank, attends to its top-k matches, and adds the result
through an output projection that is initialised to zero (the same trick LoRA
uses). At initialisation the reader is therefore an exact identity, so a wrapped
model reproduces its base model, yet the output projection receives gradient from
the first step. A zero-initialised *gate* would not work here: it would block the
gradient to every reader weight behind it, and the reader would never start
learning.

Retrieval logits get a ``log(priority)`` bias with a learnable strength. This is
the path through which a learned salience scorer receives gradient: a slot whose
reads lower the loss pushes its own priority up.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from engramlm.memory.state import MemoryState


@dataclass(frozen=True)
class ReadInfo:
    """Which slots each token read, for inspection. Detached from the graph."""

    slots: Tensor  # (B, T, k) slot indices into the memory bank that was read
    weights: Tensor  # (B, T, k) attention weights (0 where the slot was invalid)


class StaleMemoryAdapter(nn.Module):
    """Zero-initialised low-rank residual applied to stored vectors at read time.

    Memories written several segments ago were produced by older adapter weights;
    this lets the reader learn a cheap correction instead of rewriting the bank.
    """

    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, vectors: Tensor) -> Tensor:
        delta: Tensor = self.up(self.down(vectors))
        return vectors + delta


class MemoryReader(nn.Module):
    def __init__(self, hidden_dim: int, key_dim: int, top_k: int, drift_rank: int = 0) -> None:
        super().__init__()
        self.top_k = top_k
        self.key_dim = key_dim
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, key_dim, bias=False)
        self.key = nn.Linear(hidden_dim, key_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.out.weight)
        self.gate = nn.Parameter(torch.ones(1))
        self.priority_strength = nn.Parameter(torch.ones(1))
        self.adapter = StaleMemoryAdapter(hidden_dim, drift_rank) if drift_rank > 0 else None

    def forward(
        self, hidden: Tensor, memory: MemoryState, *, return_reads: bool = False
    ) -> tuple[Tensor, ReadInfo | None]:
        if memory.is_empty():
            return hidden, None

        dtype = hidden.dtype
        h = hidden.float()
        stored = memory.vectors.float()
        if self.adapter is not None:
            stored = self.adapter(stored)
        mem = self.memory_norm(stored)

        queries = self.query(self.query_norm(h))  # (B, T, Dk)
        keys = self.key(mem)  # (B, C, Dk)
        scores = queries @ keys.transpose(1, 2) / math.sqrt(self.key_dim)  # (B, T, C)
        bias = self.priority_strength * torch.log(memory.priority.float().clamp_min(1e-6))
        scores = scores + bias.unsqueeze(1)
        scores = scores.masked_fill(~memory.valid.unsqueeze(1), float("-inf"))

        k = min(self.top_k, memory.capacity)
        top_scores, top_slots = scores.topk(k, dim=-1)
        weights = torch.nan_to_num(F.softmax(top_scores, dim=-1), nan=0.0)
        dense = torch.zeros_like(scores).scatter(-1, top_slots, weights)  # (B, T, C)
        read = dense @ self.value(mem)  # (B, T, D)

        fused = h + torch.tanh(self.gate) * self.out(read)
        info = (
            ReadInfo(slots=top_slots.detach(), weights=weights.detach()) if return_reads else None
        )
        return fused.to(dtype), info


@contextmanager
def probe_reads(reader: MemoryReader, *, std: float = 0.02, seed: int = 0) -> Iterator[None]:
    """Temporarily give the reader a random output projection and an open gate.

    An untrained reader has a zero output projection, so memory cannot influence
    the logits and a leak through memory would be invisible. Audits and tests use
    this to make reads observable; the original weights are restored on exit.
    """
    saved_out = reader.out.weight.detach().clone()
    saved_gate = reader.gate.detach().clone()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(saved_out.shape, generator=gen) * std
    with torch.no_grad():
        reader.out.weight.copy_(noise.to(saved_out.device, saved_out.dtype))
        reader.gate.fill_(1.0)
    try:
        yield
    finally:
        with torch.no_grad():
            reader.out.weight.copy_(saved_out)
            reader.gate.copy_(saved_gate)
