from collections.abc import Iterator

import pytest
import torch

from engramlm.config import EngramConfig
from engramlm.eval import RecallTask
from engramlm.model import EngramLM
from engramlm.train import TrainConfig, train, warmup_cosine
from tests.conftest import make_tiny_llama


def _batches(
    task: RecallTask, batch_size: int, seed: int = 0
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    gen = torch.Generator().manual_seed(seed)
    while True:
        yield task.sample(batch_size, gen)


def test_schedule_warms_up_then_decays() -> None:
    factor = warmup_cosine(TrainConfig(steps=100, warmup_steps=10))
    assert factor(0) == pytest.approx(0.1)
    assert factor(9) == pytest.approx(1.0)
    assert factor(55) == pytest.approx(0.5)
    assert factor(100) == pytest.approx(0.0, abs=1e-9)


def test_train_config_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        TrainConfig(steps=0)
    with pytest.raises(ValueError, match="bptt"):
        TrainConfig(bptt_segments=-1)


@pytest.mark.parametrize("bptt_segments", [0, 1])
def test_training_updates_memory_parameters_and_logs(bptt_segments: int) -> None:
    task = RecallTask(pairs=4, queries=4, segment_len=16, gap_segments=1, n_filler=16)
    base = make_tiny_llama(hidden=32)
    base.resize_token_embeddings(task.vocab_size)
    model = EngramLM(
        EngramConfig(
            capacity=16, writes_per_segment=8, key_dim=8, scorer="learned", memory_layer=0
        ),
        base_model=base,
    )
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    config = TrainConfig(
        steps=4, batch_size=4, log_every=2, warmup_steps=1, bptt_segments=bptt_segments
    )
    history = train(model, _batches(task, 4), segment_len=task.segment_len, config=config)
    assert [h["step"] for h in history] == [2.0, 4.0]
    changed = [
        n for n, p in model.named_parameters() if p.requires_grad and not torch.equal(p, before[n])
    ]
    assert any(n.startswith("reader.") for n in changed)
    assert not model.training
