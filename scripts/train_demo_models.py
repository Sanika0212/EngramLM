"""Train the two recall models used by the README animation and save them.

    uv run python scripts/train_demo_models.py --out runs/demo

Trains ``learned`` (priority eviction) and ``recency`` (FIFO) at capacity 16 with
the E1 settings from ``configs/recall.yaml`` and saves each with
``engramlm.io.save_pretrained``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from engramlm.eval.recall import evaluate_recall
from engramlm.experiments.recall import RecallExperimentConfig, _batches, build_model
from engramlm.io import save_pretrained
from engramlm.train import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/recall.yaml")
    parser.add_argument("--out", default="runs/demo")
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = RecallExperimentConfig.from_dict(yaml.safe_load(Path(args.config).read_text()))
    summary = {}
    for variant in ("recency", "learned"):
        model = build_model(config, variant, args.capacity, args.seed)
        train_cfg = TrainConfig(**{**config.train.to_dict(), "seed": args.seed})
        history = train(
            model,
            _batches(config.task, train_cfg.batch_size, args.seed),
            segment_len=config.task.segment_len,
            config=train_cfg,
            log=lambda rec, v=variant: print(
                f"[{v}] step {int(rec['step'])} loss {rec['loss']:.3f}", flush=True
            ),
        )
        result = evaluate_recall(model, config.task, n_docs=512, seed=99_000 + args.seed)
        print(f"[{variant}] accuracy {result.accuracy:.3f}", flush=True)
        save_pretrained(model, Path(args.out) / variant)
        summary[variant] = {"accuracy": result.accuracy, "history": history}
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
