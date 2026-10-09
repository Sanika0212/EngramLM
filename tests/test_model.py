import itertools

import pytest
import torch
from transformers import PreTrainedModel

from engramlm.config import EngramConfig
from engramlm.model import EngramLM, causal_lm_loss
from tests.conftest import VOCAB, activate_reads, make_tiny_llama


def _engram(base: PreTrainedModel, **overrides: object) -> EngramLM:
    settings: dict[str, object] = {
        "capacity": 16,
        "writes_per_segment": 6,
        "read_top_k": 4,
        "key_dim": 8,
        "memory_layer": -1,
    }
    settings.update(overrides)
    return EngramLM(EngramConfig(**settings), base_model=base).eval()  # type: ignore[arg-type]


def _open_gate(model: EngramLM) -> None:
    activate_reads(model.reader)


def _segments(n: int = 3, batch: int = 2, length: int = 12, seed: int = 0) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(3, VOCAB, (batch, length), generator=gen) for _ in range(n)]


def test_wrapper_matches_base_model_at_init(tiny_base: PreTrainedModel) -> None:
    reference = tiny_base
    model = _engram(reference)
    memory = model.new_memory(2)
    with torch.no_grad():
        for segment in _segments():
            expected = reference(input_ids=segment).logits
            out = model(segment, memory=memory)
            assert torch.equal(out.logits, expected)
            memory = out.memory
    assert int(memory.num_valid().min()) > 0


def test_memory_written_by_a_segment_is_read_only_later() -> None:
    model = _engram(make_tiny_llama())
    _open_gate(model)
    first, second = _segments(2)
    with torch.no_grad():
        out_first = model(first)
        assert out_first.reads is None  # memory was empty while reading
        out_second = model(second, memory=out_first.memory)
        no_memory = model(second)
    assert not torch.allclose(out_second.logits, no_memory.logits)


CASES = list(
    itertools.product(["surprise", "learned", "random", "recency"], ["priority", "fifo"], [0, 1])
)


@pytest.mark.parametrize(("scorer", "eviction", "consolidate_every"), CASES)
def test_wrapper_is_causal(scorer: str, eviction: str, consolidate_every: int) -> None:
    model = _engram(
        make_tiny_llama(),
        scorer=scorer,
        eviction=eviction,
        consolidate_every=consolidate_every,
        consolidate_threshold=0.0,
    )
    _open_gate(model)
    segments = _segments(3)
    memory = model.new_memory(2)
    with torch.no_grad():
        for segment in segments[:-1]:
            memory = model(
                segment, memory=memory, generator=torch.Generator().manual_seed(0)
            ).memory
        last = segments[-1]
        for cut in (3, 7):
            perturbed = last.clone()
            perturbed[:, cut + 1 :] = (perturbed[:, cut + 1 :] + 11) % VOCAB
            a = model(last, memory=memory, write_memory=False).logits
            b = model(perturbed, memory=memory, write_memory=False).logits
            delta = (a[:, : cut + 1] - b[:, : cut + 1]).abs().max().item()
            assert delta == 0.0, f"future tokens changed earlier logits by {delta:.3e}"


def test_memory_disabled_is_the_plain_base_model() -> None:
    base = make_tiny_llama()
    model = _engram(base, memory_enabled=False)
    _open_gate(model)
    segment = _segments(1)[0]
    with torch.no_grad():
        out = model(segment)
        assert torch.equal(out.logits, base(input_ids=segment).logits)
    assert out.memory.capacity == 0
    assert out.written is None


def test_write_budget_and_padding_are_respected() -> None:
    model = _engram(make_tiny_llama(), writes_per_segment=3)
    segment = _segments(1)[0]
    attention_mask = torch.ones_like(segment)
    attention_mask[1, 6:] = 0
    with torch.no_grad():
        out = model(segment, attention_mask=attention_mask)
    assert out.written is not None
    assert out.written.sum(dim=1).tolist() == [3, 3]
    assert not out.written[1, 6:].any()


def test_learned_scorer_trains_through_a_later_read() -> None:
    model = _engram(make_tiny_llama(), scorer="learned")
    model.train()
    _open_gate(model)
    first, second = _segments(2)
    memory = model(first).memory
    out = model(second, memory=memory, labels=second, write_memory=False)
    assert out.loss is not None
    out.loss.backward()
    grads = [p.grad for p in model.scorer.parameters() if p.grad is not None]
    assert grads
    assert sum(float(g.abs().sum()) for g in grads) > 0


def test_frozen_base_has_no_trainable_parameters() -> None:
    model = _engram(make_tiny_llama())
    assert all(not p.requires_grad for p in model.base.parameters())
    trainable = {id(p) for p in model.trainable_parameters()}
    assert trainable == {id(p) for p in model.reader.parameters()} | {
        id(p) for p in model.scorer.parameters()
    }


def test_lora_adds_only_adapter_parameters() -> None:
    pytest.importorskip("peft")
    model = _engram(make_tiny_llama(), lora_rank=2, lora_targets=("q_proj", "v_proj"))
    names = [n for n, p in model.base.named_parameters() if p.requires_grad]
    assert names
    assert all("lora_" in n for n in names)
    with torch.no_grad():
        out = model(_segments(1)[0])
    assert out.logits.shape[-1] == VOCAB


def test_loss_ignores_masked_labels() -> None:
    logits = torch.randn(1, 4, VOCAB)
    labels = torch.tensor([[5, 6, -100, -100]])
    expected = torch.nn.functional.cross_entropy(logits[0, :1], labels[0, 1:2])
    assert torch.allclose(causal_lm_loss(logits, labels), expected)


def test_cache_is_returned_when_requested() -> None:
    model = _engram(make_tiny_llama())
    with torch.no_grad():
        out = model(_segments(1)[0], use_cache=True, write_memory=False)
    assert out.past_key_values is not None
