from pathlib import Path

import pytest
import torch

pytest.importorskip("matplotlib")

from engramlm.experiments.recall import RecallExperimentConfig, build_model
from engramlm.plots import (
    animate_memory,
    both_modes,
    parse_training_log,
    plot_causality,
    plot_lm_results,
    plot_recall_accuracy,
    plot_training_curves,
    recall_timeline,
)

LOG = """\
  [learned cap=16 seed=0] step 500 loss 2.400 gate 0.80
  [learned cap=16 seed=1] step 500 loss 2.600 gate 0.80
  [learned cap=16 seed=0] step 1000 loss 2.000 gate 0.80
  [no-memory cap=1 seed=0] step 500 loss 3.470 gate 0.76
"""


def test_parse_training_log_groups_seeds() -> None:
    curves = parse_training_log(LOG)
    assert curves[("learned", 16)][500] == [2.4, 2.6]
    assert ("no-memory", 0) in curves


def test_static_charts_render_in_both_modes(tmp_path: Path) -> None:
    rows = [
        {"variant": "no-memory", "capacity": 0, "accuracy": 0.03},
        {"variant": "learned", "capacity": 16, "accuracy": 0.3},
        {"variant": "learned", "capacity": 16, "accuracy": 0.35},
        {"variant": "recency", "capacity": 16, "accuracy": 0.03},
        {"variant": "learned", "capacity": 64, "accuracy": 0.4},
    ]
    paths = both_modes(plot_recall_accuracy, tmp_path / "acc", rows=rows, chance=1 / 32)
    paths += both_modes(
        plot_training_curves, tmp_path / "curves", log_text=LOG, capacity=16, chance_nll=3.47
    )
    paths += both_modes(plot_causality, tmp_path / "causal", entries=[("a", 0.0), ("b", 2e-3)])
    lm_rows = [
        {"variant": v, "bpb": b}
        for v, b in [("base", 1.2), ("lora", 1.1), ("lora+learned", 1.05), ("full-context", 0.95)]
    ]
    paths += both_modes(plot_lm_results, tmp_path / "lm", rows=lm_rows)
    assert len(paths) == 8
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 5_000
    assert {p.name for p in paths} >= {"acc-light.png", "acc-dark.png"}


def test_memory_animation_from_a_tiny_model(tmp_path: Path) -> None:
    config = RecallExperimentConfig.from_dict(
        {
            "task": {"pairs": 4, "queries": 4, "segment_len": 16, "gap_segments": 1},
            "hidden_size": 32,
            "num_layers": 2,
            "memory_layer": 0,
            "key_dim": 8,
            "writes_per_segment": 8,
        }
    )
    model = build_model(config, "learned", 8, 0)
    ids, _ = config.task.sample(1, torch.Generator().manual_seed(0))
    timeline = recall_timeline(model, config.task, ids)
    assert [f.kind for f in timeline] == ["study", "gap", "test"]
    assert len(timeline[-1].recalled) == 4
    assert all(len(f.bank_origin) == 8 for f in timeline)
    gif = animate_memory({"A": timeline, "B": timeline}, tmp_path / "m.gif", capacity=8)
    assert gif.read_bytes()[:6] in (b"GIF89a", b"GIF87a")
