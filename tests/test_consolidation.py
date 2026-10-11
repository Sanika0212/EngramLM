import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from engramlm.memory import MemoryState
from engramlm.memory.consolidation import consolidate


def _state(vectors: torch.Tensor, priority: torch.Tensor) -> MemoryState:
    batch, capacity, _ = vectors.shape
    return MemoryState(
        vectors=vectors,
        priority=priority,
        age=torch.arange(capacity, dtype=torch.float32).expand(batch, capacity).clone(),
        valid=torch.ones(batch, capacity, dtype=torch.bool),
        origin_segment=torch.arange(capacity).expand(batch, capacity).clone(),
        origin_position=torch.arange(capacity).expand(batch, capacity).clone(),
        segments_seen=capacity,
    )


def test_merges_duplicates_and_keeps_distinct() -> None:
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0])
    vectors = torch.stack([a, a * 2.0, b]).unsqueeze(0)
    priority = torch.tensor([[0.25, 0.75, 0.5]])
    merged, stats = consolidate(_state(vectors, priority), threshold=0.99)
    assert stats.merges.tolist() == [1]
    assert int(merged.num_valid()[0]) == 2
    kept = merged.valid[0]
    # The stronger slot (index 1) absorbs index 0, with the younger age (0).
    assert kept.tolist() == [False, True, True]
    assert float(merged.priority[0, 1]) == 1.0
    assert float(merged.age[0, 1]) == 0.0
    expected = (0.25 * a + 0.75 * 2.0 * a) / 1.0
    assert torch.allclose(merged.vectors[0, 1], expected)


def test_threshold_one_merges_only_parallel_vectors() -> None:
    vectors = torch.randn(1, 6, 8)
    merged, stats = consolidate(_state(vectors, torch.rand(1, 6)), threshold=1.0)
    assert stats.merges.tolist() == [0]
    assert torch.equal(merged.valid, torch.ones(1, 6, dtype=torch.bool))


def test_max_merges_is_respected() -> None:
    vectors = torch.ones(1, 5, 4)
    _, stats = consolidate(_state(vectors, torch.rand(1, 5)), threshold=0.5, max_merges=2)
    assert stats.merges.tolist() == [2]


def test_gradients_flow_through_merged_and_untouched_slots() -> None:
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0])
    vectors = torch.stack([a, a * 2.0, b]).unsqueeze(0).requires_grad_(True)
    priority = torch.tensor([[0.25, 0.75, 0.5]], requires_grad=True)
    merged, stats = consolidate(_state(vectors, priority), threshold=0.99)
    assert stats.merges.tolist() == [1]
    (merged.vectors.sum() + merged.priority.sum()).backward()
    assert vectors.grad is not None
    assert priority.grad is not None
    # Both merged slots and the untouched slot receive gradient; weights are 0.25/0.75.
    assert torch.allclose(vectors.grad[0, :, 0], torch.tensor([0.25, 0.75, 1.0]))
    assert torch.allclose(priority.grad, torch.ones(1, 3))


@settings(max_examples=50, deadline=None)
@given(
    capacity=st.integers(min_value=1, max_value=10),
    threshold=st.floats(min_value=-1.0, max_value=1.0),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_priority_mass_is_conserved(capacity: int, threshold: float, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    vectors = torch.randn(2, capacity, 4, generator=gen)
    priority = torch.rand(2, capacity, generator=gen) + 0.01
    merged, stats = consolidate(_state(vectors, priority), threshold=threshold)
    before = priority.sum(dim=1)
    after = (merged.priority * merged.valid).sum(dim=1)
    assert torch.allclose(before, after, atol=1e-5)
    assert (merged.num_valid() == capacity - stats.merges).all()
