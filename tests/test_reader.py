import pytest
import torch
from torch import nn

from engramlm.hooks import find_decoder_layers, resolve_layer_index
from engramlm.memory import MemoryState, priority_retention, write
from engramlm.reader import MemoryReader, probe_reads
from tests.conftest import activate_reads


def _filled_memory(batch: int = 2, capacity: int = 6, dim: int = 16) -> MemoryState:
    torch.manual_seed(1)
    empty = MemoryState.empty(batch, capacity, dim)
    return write(
        empty,
        torch.randn(batch, 4, dim),
        torch.rand(batch, 4),
        budget=4,
        retention=priority_retention(0.9),
    ).memory


def test_reader_is_identity_at_init() -> None:
    reader = MemoryReader(hidden_dim=16, key_dim=8, top_k=3)
    hidden = torch.randn(2, 5, 16)
    fused, _ = reader(hidden, _filled_memory())
    assert torch.equal(fused, hidden)


def test_empty_memory_short_circuits() -> None:
    reader = MemoryReader(hidden_dim=16, key_dim=8, top_k=3)
    hidden = torch.randn(2, 5, 16)
    fused, info = reader(hidden, MemoryState.empty(2, 6, 16))
    assert fused is hidden
    assert info is None


def test_reads_change_output_once_output_projection_is_trained() -> None:
    reader = MemoryReader(hidden_dim=16, key_dim=8, top_k=3)
    activate_reads(reader)
    hidden = torch.randn(2, 5, 16)
    fused, info = reader(hidden, _filled_memory(), return_reads=True)
    assert not torch.allclose(fused, hidden)
    assert info is not None
    assert info.slots.shape == (2, 5, 3)
    assert torch.allclose(info.weights.sum(-1), torch.ones(2, 5))


def test_invalid_slots_get_zero_weight() -> None:
    reader = MemoryReader(hidden_dim=16, key_dim=8, top_k=6)
    memory = _filled_memory(capacity=6)  # only 4 of 6 slots are filled
    _, info = reader(torch.randn(2, 3, 16), memory, return_reads=True)
    assert info is not None
    picked_valid = memory.valid.gather(1, info.slots.flatten(1)).view_as(info.slots)
    assert (info.weights[~picked_valid] == 0).all()


def test_each_token_reads_independently() -> None:
    reader = MemoryReader(hidden_dim=16, key_dim=8, top_k=3)
    activate_reads(reader)
    memory = _filled_memory()
    hidden = torch.randn(2, 5, 16)
    changed = hidden.clone()
    changed[:, 3:] += 5.0
    a, _ = reader(hidden, memory)
    b, _ = reader(changed, memory)
    assert torch.equal(a[:, :3], b[:, :3])


def test_priority_receives_gradient_from_reads() -> None:
    reader = MemoryReader(hidden_dim=16, key_dim=8, top_k=4)
    activate_reads(reader)
    priority = torch.rand(2, 4, requires_grad=True)
    memory = write(
        MemoryState.empty(2, 4, 16),
        torch.randn(2, 4, 16),
        priority,
        budget=4,
        retention=priority_retention(0.9),
    ).memory
    fused, _ = reader(torch.randn(2, 5, 16), memory)
    fused.pow(2).sum().backward()
    assert priority.grad is not None
    assert priority.grad.abs().sum() > 0


def test_drift_adapter_starts_as_identity() -> None:
    plain = MemoryReader(hidden_dim=16, key_dim=8, top_k=3)
    activate_reads(plain)
    adapted = MemoryReader(hidden_dim=16, key_dim=8, top_k=3, drift_rank=2)
    adapted.load_state_dict(plain.state_dict(), strict=False)
    hidden = torch.randn(1, 4, 16)
    memory = _filled_memory(batch=1)
    assert torch.allclose(plain(hidden, memory)[0], adapted(hidden, memory)[0])


def test_probe_reads_restores_weights() -> None:
    reader = MemoryReader(hidden_dim=16, key_dim=8, top_k=3)
    hidden = torch.randn(1, 4, 16)
    memory = _filled_memory(batch=1)
    with probe_reads(reader):
        probed, _ = reader(hidden, memory)
        assert not torch.allclose(probed, hidden)
    assert torch.equal(reader.out.weight, torch.zeros_like(reader.out.weight))
    assert torch.equal(reader(hidden, memory)[0], hidden)


def test_find_decoder_layers(tiny_base: nn.Module) -> None:
    layers = find_decoder_layers(tiny_base)
    assert isinstance(layers, nn.ModuleList)
    assert len(layers) >= 2


def test_find_decoder_layers_rejects_unknown_models() -> None:
    with pytest.raises(ValueError, match="Could not find"):
        find_decoder_layers(nn.Linear(2, 2))


@pytest.mark.parametrize(("index", "expected"), [(0, 0), (-1, 3), (-4, 0)])
def test_resolve_layer_index(index: int, expected: int) -> None:
    assert resolve_layer_index(index, 4) == expected


def test_resolve_layer_index_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        resolve_layer_index(4, 4)
