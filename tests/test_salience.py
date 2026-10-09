import pytest
import torch

from engramlm.salience import (
    LearnedScorer,
    SalienceContext,
    build_scorer,
    standardize,
    token_surprise,
)


def _ctx(batch: int = 2, seq_len: int = 6, vocab: int = 11, dim: int = 8) -> SalienceContext:
    torch.manual_seed(0)
    return SalienceContext(
        hidden=torch.randn(batch, seq_len, dim),
        logits=torch.randn(batch, seq_len, vocab),
        input_ids=torch.randint(0, vocab, (batch, seq_len)),
        mask=torch.ones(batch, seq_len, dtype=torch.bool),
    )


@pytest.mark.parametrize("name", ["surprise", "learned", "random", "recency"])
def test_scorers_return_priorities_in_unit_interval(name: str) -> None:
    ctx = _ctx()
    scores = build_scorer(name, hidden_dim=8)(ctx)
    assert scores.shape == (2, 6)
    assert torch.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()


def test_surprise_ranks_unexpected_token_highest() -> None:
    vocab = 5
    logits = torch.full((1, 4, vocab), -10.0)
    input_ids = torch.tensor([[0, 1, 2, 3]])
    # Predictions at t=0,1 put mass on the right next token; at t=2 they expect 0, not 3.
    logits[0, 0, 1] = 10.0
    logits[0, 1, 2] = 10.0
    logits[0, 2, 0] = 10.0
    ctx = SalienceContext(
        hidden=torch.zeros(1, 4, 3),
        logits=logits,
        input_ids=input_ids,
        mask=torch.ones(1, 4, dtype=torch.bool),
    )
    scores = build_scorer("surprise", 3)(ctx)
    assert int(scores[0].argmax()) == 3


def test_surprise_ignores_first_token_and_padding() -> None:
    ctx = _ctx()
    mask = ctx.mask.clone()
    mask[0, 3:] = False
    _, defined = token_surprise(ctx.logits, ctx.input_ids, mask)
    assert not defined[:, 0].any()
    assert defined[0].tolist() == [False, True, True, False, False, False]


def test_standardize_has_zero_mean_on_defined_positions() -> None:
    values = torch.tensor([[1.0, 2.0, 3.0, 100.0]])
    defined = torch.tensor([[True, True, True, False]])
    z = standardize(values, defined)
    assert float(z[0, :3].mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(z[0, 3]) == 0.0


def test_learned_scorer_starts_as_surprise_and_is_trainable() -> None:
    ctx = _ctx()
    learned = LearnedScorer(hidden_dim=8)
    surprise = build_scorer("surprise", 8)
    assert torch.allclose(learned(ctx), surprise(ctx), atol=1e-6)
    learned(ctx).sum().backward()
    assert learned.head.weight.grad is not None
    assert learned.head.weight.grad.abs().sum() > 0


def test_random_scorer_is_reproducible_with_generator() -> None:
    base = _ctx()
    a = build_scorer("random", 8)(
        SalienceContext(**{**base.__dict__, "generator": torch.Generator().manual_seed(7)})
    )
    b = build_scorer("random", 8)(
        SalienceContext(**{**base.__dict__, "generator": torch.Generator().manual_seed(7)})
    )
    assert torch.equal(a, b)


def test_recency_is_monotonic() -> None:
    scores = build_scorer("recency", 8)(_ctx())
    assert (scores[:, 1:] > scores[:, :-1]).all()


def test_unknown_scorer_raises() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        build_scorer("oracle", 8)
