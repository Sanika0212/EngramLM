"""Measure how far later tokens move earlier logits, for three models.

    uv run python scripts/leak_demo.py

* a plain causal LM (control),
* the same LM with a *sequence-pooled* memory: the mean hidden state of the whole
  sequence is added to every position, a common way memory modules end up
  seeing the future,
* EngramLM wrapping the same LM, with reads switched on (``probe_reads``) and a
  filled memory, for every scorer, eviction policy and consolidation setting.

Each number is the largest absolute change in logits at or before a random cut
point when every token after the cut is replaced (``engramlm.audit``).
"""

from __future__ import annotations

import itertools

import torch
from torch import Tensor, nn
from transformers import LlamaConfig, LlamaForCausalLM

from engramlm.audit import audit_causality, audit_engram
from engramlm.config import EngramConfig
from engramlm.model import EngramLM

VOCAB = 97


def tiny_llama(seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        bos_token_id=None,
        eos_token_id=None,
    )
    return LlamaForCausalLM(config).eval()  # type: ignore[no-untyped-call]


class SequencePooledMemory(nn.Module):
    """Adds the sequence mean of the final hidden states to every position."""

    def __init__(self, base: LlamaForCausalLM) -> None:
        super().__init__()
        self.base = base

    def forward(self, input_ids: Tensor) -> Tensor:
        hidden = self.base.model(input_ids=input_ids).last_hidden_state
        logits: Tensor = self.base.lm_head(hidden + hidden.mean(dim=1, keepdim=True))
        return logits


@torch.no_grad()
def measure_leaks(trials: int = 16, seed: int = 0) -> list[tuple[str, float]]:
    ids = torch.randint(0, VOCAB, (2, 32), generator=torch.Generator().manual_seed(seed))
    base = tiny_llama(seed)
    plain = audit_causality(
        lambda x: base(input_ids=x).logits, ids, vocab_size=VOCAB, trials=trials
    )
    pooled = audit_causality(SequencePooledMemory(base), ids, vocab_size=VOCAB, trials=trials)
    worst = 0.0
    for scorer, eviction, every in itertools.product(
        ["surprise", "learned", "random", "recency"], ["priority", "fifo"], [0, 1]
    ):
        config = EngramConfig(
            capacity=16,
            writes_per_segment=6,
            read_top_k=4,
            key_dim=8,
            scorer=scorer,
            eviction=eviction,
            consolidate_every=every,
            consolidate_threshold=0.0,
        )
        report = audit_engram(
            EngramLM(config, base_model=tiny_llama(seed)),
            vocab_size=VOCAB,
            segment_len=32,
            trials=trials,
        )
        worst = max(worst, report.max_abs_delta)
    return [
        ("Plain causal LM (control)", plain.max_abs_delta),
        ("Sequence-pooled memory", pooled.max_abs_delta),
        ("EngramLM, all 16 variants", worst),
    ]


if __name__ == "__main__":
    for label, value in measure_leaks():
        print(f"{label:30s} {value:.3e}")
