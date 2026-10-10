"""Experiment E1: cross-segment associative recall across memory variants.

Small Llama-style models are trained from scratch on :class:`RecallTask` with each
memory variant and capacity, over several seeds. Results are written as
``results.json`` (every run) and ``results.md`` (mean accuracy with a bootstrap
95% interval across seeds).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from engramlm.config import EngramConfig
from engramlm.eval.recall import PAD, RecallTask, evaluate_recall
from engramlm.eval.stats import bootstrap_ci
from engramlm.model import EngramLM
from engramlm.train import TrainConfig, train

VARIANTS: dict[str, dict[str, Any]] = {
    "no-memory": {"memory_enabled": False},
    "recency": {"scorer": "recency", "eviction": "fifo"},
    "random": {"scorer": "random"},
    "surprise": {"scorer": "surprise"},
    "learned": {"scorer": "learned"},
    "learned+sleep": {"scorer": "learned", "consolidate_every": 1},
}


@dataclass
class RecallExperimentConfig:
    task: RecallTask = field(default_factory=RecallTask)
    hidden_size: int = 128
    num_layers: int = 3
    num_heads: int = 4
    memory_layer: int = 1
    key_dim: int = 32
    read_top_k: int = 8
    writes_per_segment: int = 16
    consolidate_threshold: float = 0.95
    capacities: list[int] = field(default_factory=lambda: [16, 32, 64])
    variants: list[str] = field(default_factory=lambda: list(VARIANTS))
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    train: TrainConfig = field(default_factory=TrainConfig)
    eval_docs: int = 512
    save_checkpoints: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecallExperimentConfig:
        data = dict(data)
        task = RecallTask(**data.pop("task", {}))
        train_cfg = TrainConfig(**data.pop("train", {}))
        unknown = [v for v in data.get("variants", []) if v not in VARIANTS]
        if unknown:
            raise ValueError(f"unknown variants {unknown}; choose from {list(VARIANTS)}")
        return cls(task=task, train=train_cfg, **data)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["task"] = self.task.to_dict()
        data["train"] = self.train.to_dict()
        return data


def build_model(config: RecallExperimentConfig, variant: str, capacity: int, seed: int) -> EngramLM:
    torch.manual_seed(seed)
    base = LlamaForCausalLM(  # type: ignore[no-untyped-call]
        LlamaConfig(
            vocab_size=config.task.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.hidden_size * 2,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            max_position_embeddings=config.task.segment_len,
            bos_token_id=None,
            eos_token_id=None,
            pad_token_id=PAD,
        )
    )
    engram = EngramConfig(
        base_model_name="scratch-llama",
        freeze_base=False,
        memory_layer=config.memory_layer,
        capacity=capacity,
        writes_per_segment=config.writes_per_segment,
        read_top_k=config.read_top_k,
        key_dim=config.key_dim,
        consolidate_threshold=config.consolidate_threshold,
        **VARIANTS[variant],
    )
    return EngramLM(engram, base_model=base)


def _batches(
    task: RecallTask, batch_size: int, seed: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    gen = torch.Generator().manual_seed(10_000 + seed)
    while True:
        yield task.sample(batch_size, gen)


def run_one(
    config: RecallExperimentConfig,
    variant: str,
    capacity: int,
    seed: int,
    device: torch.device,
    log: Callable[[str], None] = print,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    model = build_model(config, variant, capacity, seed).to(device)
    train_cfg = TrainConfig(**{**config.train.to_dict(), "seed": seed})
    started = time.perf_counter()
    history = train(
        model,
        _batches(config.task, train_cfg.batch_size, seed),
        segment_len=config.task.segment_len,
        config=train_cfg,
        log=lambda rec: log(
            f"  [{variant} cap={capacity} seed={seed}] step {int(rec['step'])} "
            f"loss {rec['loss']:.3f} gate {rec['gate']:.3f}"
        ),
    )
    seconds = time.perf_counter() - started
    result = evaluate_recall(model, config.task, n_docs=config.eval_docs, seed=99_000 + seed)
    if checkpoint_dir is not None:
        from engramlm.io import save_pretrained

        save_pretrained(model, checkpoint_dir)
    return {
        "variant": variant,
        "capacity": capacity if model.config.memory_enabled else 0,
        "seed": seed,
        "accuracy": result.accuracy,
        "target_nll": result.target_nll,
        "n_targets": result.n_targets,
        "train_seconds": round(seconds, 1),
        "final_loss": history[-1]["loss"] if history else float("nan"),
        "final_gate": history[-1]["gate"] if history else float("nan"),
    }


def summarize(rows: list[dict[str, Any]], chance: float) -> str:
    groups: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        groups.setdefault((row["variant"], row["capacity"]), []).append(row["accuracy"])
    order = {name: i for i, name in enumerate(VARIANTS)}
    lines = [
        f"Chance accuracy: {chance:.3f}",
        "",
        "| Variant | Capacity | Seeds | Accuracy (mean [95% CI]) |",
        "|---|---|---|---|",
    ]
    for (variant, capacity), accs in sorted(
        groups.items(), key=lambda kv: (order[kv[0][0]], kv[0][1])
    ):
        interval = bootstrap_ci(accs, n_resamples=5000)
        cap = "n/a" if capacity == 0 else str(capacity)
        lines.append(f"| {variant} | {cap} | {len(accs)} | {interval.format()} |")
    return "\n".join(lines) + "\n"


def run_experiment(
    config: RecallExperimentConfig,
    out_dir: str | Path,
    *,
    device: torch.device | str = "cpu",
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    rows: list[dict[str, Any]] = []
    results_path = out / "results.json"
    for variant in config.variants:
        capacities = [0] if not VARIANTS[variant].get("memory_enabled", True) else config.capacities
        for capacity in capacities:
            for seed in config.seeds:
                log(f"running {variant} capacity={capacity or '-'} seed={seed}")
                ckpt = (
                    out / "checkpoints" / f"{variant}-cap{capacity}-seed{seed}"
                    if config.save_checkpoints
                    else None
                )
                row = run_one(config, variant, max(capacity, 1), seed, device, log, ckpt)
                log(f"  -> accuracy {row['accuracy']:.3f} ({row['train_seconds']}s)")
                rows.append(row)
                payload = {"config": config.to_dict(), "rows": rows}
                results_path.write_text(json.dumps(payload, indent=2) + "\n")
    chance = 1.0 / config.task.n_values
    (out / "results.md").write_text(summarize(rows, chance))
    return rows
