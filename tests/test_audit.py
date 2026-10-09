import pytest
import torch
from torch import Tensor, nn

from engramlm.audit import audit_causality, audit_engram
from engramlm.config import EngramConfig
from engramlm.model import EngramLM
from tests.conftest import VOCAB, make_tiny_llama


class SequenceMeanLeak(nn.Module):
    """A deliberately broken wrapper: adds the mean over the whole sequence to every
    position, the classic way a memory module ends up seeing the future."""

    def __init__(self) -> None:
        super().__init__()
        self.base = make_tiny_llama()

    def forward(self, input_ids: Tensor) -> Tensor:
        hidden = self.base.model(input_ids=input_ids).last_hidden_state
        hidden = hidden + hidden.mean(dim=1, keepdim=True)
        logits: Tensor = self.base.lm_head(hidden)
        return logits


def test_audit_passes_a_causal_model() -> None:
    base = make_tiny_llama()
    ids = torch.randint(0, VOCAB, (2, 16))
    report = audit_causality(lambda x: base(input_ids=x).logits, ids, vocab_size=VOCAB)
    assert report.passed
    assert report.max_abs_delta == 0.0
    assert "PASS" in report.summary()


def test_audit_catches_a_sequence_pooling_leak() -> None:
    leaky = SequenceMeanLeak().eval()
    ids = torch.randint(0, VOCAB, (2, 16))
    report = audit_causality(leaky, ids, vocab_size=VOCAB)
    assert not report.passed
    assert report.max_abs_delta > 1e-4
    assert "FAIL" in report.summary()


def test_audit_engram_restores_gate_and_mode() -> None:
    model = EngramLM(
        EngramConfig(capacity=16, writes_per_segment=4, read_top_k=4, key_dim=8),
        base_model=make_tiny_llama(),
    )
    model.train()
    report = audit_engram(model, vocab_size=VOCAB, segment_len=12, open_gate=0.7)
    assert report.passed
    assert float(model.reader.gate.detach()) == 0.0
    assert model.training


def test_audit_requires_two_tokens() -> None:
    with pytest.raises(ValueError, match="two tokens"):
        audit_causality(lambda x: x.float(), torch.zeros(1, 1, dtype=torch.long), vocab_size=4)
