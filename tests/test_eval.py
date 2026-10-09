import math

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from engramlm.config import EngramConfig
from engramlm.eval import (
    RecallTask,
    bits_per_byte,
    bits_per_token,
    bootstrap_ci,
    evaluate_recall,
    paired_bootstrap_diff,
    perplexity,
    strided_nll,
)
from engramlm.eval.recall import QUERY
from engramlm.model import EngramLM
from engramlm.segment import token_nll_sum
from tests.conftest import VOCAB, make_tiny_llama


def test_metric_conversions() -> None:
    nll = 10.0 * math.log(2)
    assert perplexity(nll, 10) == pytest.approx(2.0)
    assert bits_per_token(nll, 10) == pytest.approx(1.0)
    assert bits_per_byte(nll, 20) == pytest.approx(0.5)
    assert math.isnan(perplexity(1.0, 0))


def test_strided_nll_scores_every_token_once() -> None:
    base = make_tiny_llama()
    ids = torch.randint(3, VOCAB, (1, 50))

    def logits_fn(x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = base(input_ids=x).logits
        return logits

    nll, count = strided_nll(logits_fn, ids, max_length=16, stride=8)
    assert count == 49
    full_nll, full_count = strided_nll(logits_fn, ids, max_length=64, stride=64)
    with torch.no_grad():
        reference, ref_count = token_nll_sum(logits_fn(ids), ids)
    assert full_count == int(ref_count) == 49
    assert full_nll == pytest.approx(float(reference), rel=1e-5)
    assert nll != pytest.approx(full_nll)  # shorter context changes the score


def test_strided_nll_validates_stride() -> None:
    with pytest.raises(ValueError, match="stride"):
        strided_nll(lambda x: x, torch.zeros(1, 4, dtype=torch.long), max_length=4, stride=8)


def test_bootstrap_interval_brackets_the_mean() -> None:
    interval = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], n_resamples=2000)
    assert interval.low <= interval.mean <= interval.high
    assert interval.mean == pytest.approx(0.3)
    assert "[" in interval.format()
    single = bootstrap_ci([0.7])
    assert (single.low, single.high) == (0.7, 0.7)
    with pytest.raises(ValueError, match="at least one"):
        bootstrap_ci([])


def test_paired_difference() -> None:
    diff = paired_bootstrap_diff([1.0, 2.0, 3.0], [0.5, 1.5, 2.5], n_resamples=500)
    assert diff.mean == pytest.approx(0.5)
    assert diff.low == pytest.approx(0.5)
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_diff([1.0], [1.0, 2.0])


def test_recall_task_layout() -> None:
    task = RecallTask(pairs=4, queries=3, segment_len=16, gap_segments=1)
    ids, target = task.sample(5, torch.Generator().manual_seed(0))
    assert ids.shape == target.shape == (5, task.doc_len)
    assert int(ids.max()) < task.vocab_size
    assert target.sum(dim=1).tolist() == [3] * 5
    test_start = task.segment_len * (task.num_segments - 1)
    for b in range(5):
        study = dict(zip(ids[b, 0:8:2].tolist(), ids[b, 1:8:2].tolist(), strict=True))
        for pos in target[b].nonzero().squeeze(1).tolist():
            assert pos >= test_start
            assert int(ids[b, pos - 2]) == QUERY
            assert study[int(ids[b, pos - 1])] == int(ids[b, pos])


def test_recall_task_validation() -> None:
    with pytest.raises(ValueError, match="fit"):
        RecallTask(pairs=20, segment_len=16)


def test_evaluate_recall_returns_a_valid_accuracy() -> None:
    task = RecallTask(pairs=4, queries=4, segment_len=16, gap_segments=1)
    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=task.vocab_size,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            bos_token_id=None,
            eos_token_id=None,
            pad_token_id=0,
        )
    )
    model = EngramLM(EngramConfig(capacity=8, writes_per_segment=8, key_dim=8), base_model=base)
    result = evaluate_recall(model, task, n_docs=10, batch_size=4)
    assert 0.0 <= result.accuracy <= 1.0
    assert result.n_targets == 40
    assert math.isfinite(result.target_nll)
