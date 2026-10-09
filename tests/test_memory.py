from pathlib import Path

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from engramlm.memory import MemoryState, build_policy, fifo_retention, priority_retention, write


def _write(memory: MemoryState, priority: torch.Tensor, budget: int, policy: str = "priority"):
    batch, seq_len = priority.shape
    candidates = torch.randn(batch, seq_len, memory.dim)
    return write(
        memory, candidates, priority, budget=budget, retention=build_policy(policy, decay=0.9)
    ), candidates


def test_empty_state_shapes() -> None:
    mem = MemoryState.empty(3, 5, 4)
    assert (mem.batch_size, mem.capacity, mem.dim) == (3, 5, 4)
    assert mem.is_empty()
    assert mem.num_valid().tolist() == [0, 0, 0]


def test_write_picks_highest_priority_tokens() -> None:
    mem = MemoryState.empty(1, 4, 3)
    priority = torch.tensor([[0.1, 0.9, 0.2, 0.8, 0.3]])
    result, candidates = _write(mem, priority, budget=2)
    stored = result.memory.vectors[0, result.memory.valid[0]]
    expected = candidates[0, [1, 3]]
    assert torch.allclose(stored.sort(dim=0).values, expected.sort(dim=0).values)
    assert result.written[0].tolist() == [False, True, False, True, False]


def test_masked_tokens_are_never_written() -> None:
    mem = MemoryState.empty(1, 8, 2)
    priority = torch.tensor([[0.9, 0.8, 0.1]])
    mask = torch.tensor([[False, True, True]])
    result = write(
        mem,
        torch.randn(1, 3, 2),
        priority,
        budget=3,
        retention=priority_retention(0.9),
        mask=mask,
    )
    assert result.written[0].tolist() == [False, True, True]
    assert int(result.memory.num_valid()[0]) == 2


def test_priority_policy_evicts_lowest_retention() -> None:
    mem = MemoryState.empty(1, 2, 2)
    first, _ = _write(mem, torch.tensor([[0.2, 0.9]]), budget=2)
    second, _ = _write(first.memory, torch.tensor([[0.5]]), budget=1)
    kept = sorted(second.memory.priority[0, second.memory.valid[0]].tolist())
    # 0.9 decays to 0.81 but still beats the new 0.5; 0.2 * 0.9 = 0.18 is evicted.
    assert kept == pytest.approx([0.5, 0.9])


def test_fifo_policy_keeps_newest() -> None:
    mem = MemoryState.empty(1, 2, 2)
    retention = fifo_retention()
    a = write(mem, torch.randn(1, 2, 2), torch.tensor([[1.0, 1.0]]), budget=2, retention=retention)
    b = write(a.memory, torch.randn(1, 1, 2), torch.tensor([[0.01]]), budget=1, retention=retention)
    assert sorted(b.memory.origin_segment[0].tolist()) == [0, 1]
    assert 0.01 in [round(p, 4) for p in b.memory.priority[0].tolist()]


def test_ages_advance_and_segments_count() -> None:
    mem = MemoryState.empty(1, 4, 2)
    a, _ = _write(mem, torch.tensor([[0.5, 0.6]]), budget=2)
    b, _ = _write(a.memory, torch.tensor([[0.7]]), budget=1)
    ages = sorted(b.memory.age[0, b.memory.valid[0]].tolist())
    assert ages == [0.0, 1.0, 1.0]
    assert b.memory.segments_seen == 2


def test_streams_are_isolated() -> None:
    mem = MemoryState.empty(2, 3, 2)
    priority = torch.tensor([[0.9, 0.9], [0.1, 0.1]])
    mask = torch.tensor([[True, True], [False, False]])
    result = write(
        mem, torch.randn(2, 2, 2), priority, budget=2, retention=priority_retention(0.9), mask=mask
    )
    assert result.memory.num_valid().tolist() == [2, 0]


def test_priority_gradient_flows_through_write() -> None:
    mem = MemoryState.empty(1, 4, 2)
    priority = torch.rand(1, 5, requires_grad=True)
    result = write(mem, torch.randn(1, 5, 2), priority, budget=3, retention=priority_retention(0.9))
    result.memory.priority.sum().backward()
    assert priority.grad is not None
    assert int((priority.grad != 0).sum()) == 3


def test_zero_capacity_is_a_no_op_that_still_counts_segments() -> None:
    mem = MemoryState.empty(1, 0, 2)
    result, _ = _write(mem, torch.rand(1, 4), budget=4)
    assert result.memory.capacity == 0
    assert result.memory.segments_seen == 1
    assert not result.written.any()


def test_shape_mismatch_raises() -> None:
    mem = MemoryState.empty(1, 4, 3)
    with pytest.raises(ValueError, match="candidates"):
        write(mem, torch.randn(1, 2, 5), torch.rand(1, 2), budget=1, retention=fifo_retention())


def test_save_load_round_trip(tmp_path: Path) -> None:
    mem = MemoryState.empty(2, 4, 3)
    result, _ = _write(mem, torch.rand(2, 6), budget=3)
    path = tmp_path / "memory.safetensors"
    result.memory.save(path)
    loaded = MemoryState.load(path)
    assert loaded.segments_seen == result.memory.segments_seen
    for name in ("vectors", "priority", "age", "valid", "origin_segment", "origin_position"):
        assert torch.equal(getattr(loaded, name), getattr(result.memory, name).detach())


def test_detach_and_select_streams() -> None:
    mem = MemoryState.empty(3, 2, 2)
    result = write(
        mem,
        torch.randn(3, 2, 2),
        torch.rand(3, 2, requires_grad=True),
        budget=2,
        retention=priority_retention(0.9),
    )
    assert result.memory.priority.requires_grad
    detached = result.memory.detach()
    assert not detached.priority.requires_grad
    picked = detached.select_streams(torch.tensor([0, 2]))
    assert picked.batch_size == 2


@settings(max_examples=60, deadline=None)
@given(
    capacity=st.integers(min_value=1, max_value=12),
    budget=st.integers(min_value=0, max_value=10),
    segments=st.lists(st.integers(min_value=1, max_value=9), min_size=1, max_size=6),
    policy=st.sampled_from(["priority", "fifo"]),
)
def test_capacity_invariant_holds(
    capacity: int, budget: int, segments: list[int], policy: str
) -> None:
    torch.manual_seed(0)
    mem = MemoryState.empty(2, capacity, 4)
    total_written = 0
    for seg_len in segments:
        result, _ = _write(mem, torch.rand(2, seg_len), budget=budget, policy=policy)
        mem = result.memory
        total_written += min(budget, seg_len)
        assert int(mem.num_valid().max()) <= capacity
        assert result.written.sum(dim=1).le(min(budget, seg_len)).all()
    assert mem.num_valid().tolist() == [min(capacity, total_written)] * 2
    assert mem.segments_seen == len(segments)
