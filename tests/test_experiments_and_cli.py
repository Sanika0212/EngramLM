import json
import math
from pathlib import Path

import pytest
import torch
import yaml

from engramlm.cli import main
from engramlm.config import EngramConfig
from engramlm.experiments.lm import (
    LMExperimentConfig,
    _engram_config,
    evaluate_full_context,
    evaluate_lm,
    scored_bytes,
    split_articles,
)
from engramlm.experiments.recall import RecallExperimentConfig, run_experiment, summarize
from engramlm.model import EngramLM
from tests.conftest import VOCAB, make_tiny_llama

QUICK_RECALL = {
    "task": {"pairs": 4, "queries": 4, "segment_len": 16, "gap_segments": 1, "n_filler": 8},
    "hidden_size": 32,
    "num_layers": 2,
    "memory_layer": 0,
    "key_dim": 8,
    "writes_per_segment": 8,
    "capacities": [8],
    "variants": ["no-memory", "learned"],
    "seeds": [0],
    "train": {"steps": 2, "batch_size": 4, "log_every": 1, "warmup_steps": 1},
    "eval_docs": 8,
}


class ByteTokenizer:
    """Minimal stand-in: every id decodes to one ASCII character."""

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(65 + i % 26) for i in ids)


def test_recall_experiment_writes_results(tmp_path: Path) -> None:
    config = RecallExperimentConfig.from_dict(QUICK_RECALL)
    rows = run_experiment(config, tmp_path, log=lambda _: None)
    assert [r["variant"] for r in rows] == ["no-memory", "learned"]
    assert rows[0]["capacity"] == 0
    saved = json.loads((tmp_path / "results.json").read_text())
    assert len(saved["rows"]) == 2
    table = (tmp_path / "results.md").read_text()
    assert "| learned | 8 | 1 |" in table
    assert "| no-memory | — | 1 |" in table


def test_recall_experiment_can_save_checkpoints(tmp_path: Path) -> None:
    config = RecallExperimentConfig.from_dict(
        {**QUICK_RECALL, "variants": ["learned"], "save_checkpoints": True}
    )
    run_experiment(config, tmp_path, log=lambda _: None)
    checkpoint = tmp_path / "checkpoints" / "learned-cap8-seed0"
    assert (checkpoint / "engram_config.json").exists()
    assert (checkpoint / "base").exists()


def test_recall_config_rejects_unknown_variants() -> None:
    with pytest.raises(ValueError, match="unknown variants"):
        RecallExperimentConfig.from_dict({"variants": ["telepathy"]})


def test_summarize_orders_variants() -> None:
    rows = [
        {"variant": "learned", "capacity": 8, "accuracy": 0.5},
        {"variant": "no-memory", "capacity": 0, "accuracy": 0.1},
    ]
    table = summarize(rows, chance=0.1)
    assert table.index("no-memory") < table.index("learned")


def test_split_articles_groups_by_top_level_titles() -> None:
    lines = [
        " = Alpha = \n",
        " body a \n",
        " = = Section = = \n",
        " more a \n",
        " = Beta = \n",
        " b \n",
    ]
    articles = split_articles(lines)
    assert len(articles) == 2
    assert articles[0].startswith("= Alpha =")
    assert "Section" in articles[0]


def test_lm_variants_map_to_configs() -> None:
    config = LMExperimentConfig()
    assert not _engram_config(config, "base").memory_enabled
    assert _engram_config(config, "lora").lora_rank == config.lora_rank
    assert not _engram_config(config, "lora").memory_enabled
    learned = _engram_config(config, "lora+learned")
    assert learned.scorer == "learned"
    assert learned.capacity == config.memory["capacity"]
    with pytest.raises(ValueError, match="unknown variants"):
        LMExperimentConfig.from_dict({"variants": ["lora+magic"]})


def test_full_context_and_segmented_scores_cover_the_same_targets() -> None:
    model = EngramLM(EngramConfig(memory_enabled=False), base_model=make_tiny_llama()).eval()
    docs = [torch.randint(3, VOCAB, (40,)), torch.randint(3, VOCAB, (32,))]
    tokenizer = ByteTokenizer()
    segmented = evaluate_lm(model, docs, 8, tokenizer)
    full = evaluate_full_context(model, docs, 8, tokenizer)
    assert segmented["tokens"] == full["tokens"] == (40 - 5) + (32 - 4)
    assert scored_bytes(tokenizer, docs[0], 8) == 35
    assert all(map(math.isfinite, (full["ppl"], full["bpb"], segmented["ppl"], segmented["bpb"])))


def test_cli_audit_tiny(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["audit", "--tiny", "--trials", "3", "--segment-len", "16"]) == 0
    assert "PASS" in capsys.readouterr().out


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip()


def test_cli_recall(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "recall.yaml"
    config_path.write_text(yaml.safe_dump({**QUICK_RECALL, "variants": ["learned"]}))
    assert main(["recall", "--config", str(config_path), "--out", str(tmp_path / "run")]) == 0
    assert "Chance accuracy" in capsys.readouterr().out
