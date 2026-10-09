import math

import pytest
import torch

from engramlm.config import EngramConfig
from engramlm.generation import generate
from engramlm.model import EngramLM
from engramlm.segment import evaluate_document, segment_bounds, token_nll_sum
from tests.conftest import VOCAB, make_tiny_llama


def _model(**overrides: object) -> EngramLM:
    settings: dict[str, object] = {
        "capacity": 32,
        "writes_per_segment": 8,
        "read_top_k": 4,
        "key_dim": 8,
    }
    settings.update(overrides)
    model = EngramLM(EngramConfig(**settings), base_model=make_tiny_llama()).eval()  # type: ignore[arg-type]
    with torch.no_grad():
        model.reader.gate.fill_(1.0)
    return model


def test_segment_bounds_tile_the_document() -> None:
    assert segment_bounds(10, 4) == [(0, 4), (4, 8), (8, 10)]
    with pytest.raises(ValueError, match="at least 2"):
        segment_bounds(10, 1)


def test_token_nll_sum_matches_mean_loss() -> None:
    logits = torch.randn(2, 5, VOCAB)
    labels = torch.randint(0, VOCAB, (2, 5))
    labels[0, 3] = -100
    total, count = token_nll_sum(logits, labels)
    mean = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, VOCAB), labels[:, 1:].reshape(-1), ignore_index=-100
    )
    assert int(count) == 7
    assert math.isclose(float(total / count), float(mean), rel_tol=1e-5)


def test_evaluate_document_counts_and_carries_memory() -> None:
    model = _model()
    ids = torch.randint(3, VOCAB, (2, 30))
    result = evaluate_document(model, ids, segment_len=10, keep_logits=True)
    assert len(result.segment_nll) == 3
    assert result.n_tokens == 2 * 3 * 9  # each segment scores T - 1 positions per row
    assert result.memory.segments_seen == 3
    assert len(result.logits) == 3
    assert math.isfinite(result.mean_nll)


def test_score_mask_restricts_scored_tokens() -> None:
    model = _model()
    ids = torch.randint(3, VOCAB, (1, 20))
    mask = torch.zeros(1, 20, dtype=torch.bool)
    mask[0, [5, 15]] = True
    result = evaluate_document(model, ids, segment_len=10, score_mask=mask)
    assert result.n_tokens == 2


def test_memory_changes_scores_of_later_segments() -> None:
    model = _model()
    ids = torch.randint(3, VOCAB, (1, 30))
    with_memory = evaluate_document(model, ids, segment_len=10)
    without = evaluate_document(_model(memory_enabled=False), ids, segment_len=10)
    assert with_memory.segment_nll[0] == pytest.approx(without.segment_nll[0])
    assert with_memory.segment_nll[2] != pytest.approx(without.segment_nll[2])


def test_cached_and_uncached_greedy_generation_agree() -> None:
    model = _model()
    prompt = torch.randint(3, VOCAB, (2, 6))
    memory = evaluate_document(model, torch.randint(3, VOCAB, (2, 24)), segment_len=8).memory
    cached = generate(model, prompt, memory=memory, max_new_tokens=8, use_cache=True)
    uncached = generate(model, prompt, memory=memory, max_new_tokens=8, use_cache=False)
    assert torch.equal(cached, uncached)
    assert cached.shape == (2, 14)


def test_generation_stops_at_eos() -> None:
    model = _model()
    prompt = torch.randint(3, VOCAB, (1, 4))
    first = generate(model, prompt, max_new_tokens=1)
    eos = int(first[0, -1])
    out = generate(model, prompt, max_new_tokens=10, eos_token_id=eos)
    assert out.shape[1] == 5


def test_sampling_is_reproducible_with_a_generator() -> None:
    model = _model()
    prompt = torch.randint(3, VOCAB, (1, 4))
    runs = [
        generate(
            model,
            prompt,
            max_new_tokens=6,
            temperature=0.8,
            top_p=0.9,
            generator=torch.Generator().manual_seed(3),
        )
        for _ in range(2)
    ]
    assert torch.equal(runs[0], runs[1])
