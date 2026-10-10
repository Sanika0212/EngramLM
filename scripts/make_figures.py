"""Build every README figure from experiment outputs.

    uv run --extra viz python scripts/make_figures.py

Inputs (any that are missing are skipped):
    runs/recall/results.json + runs/recall.log   E1 accuracy chart and training curves
    runs/lm-smollm2/results.json                 E2 chart
    runs/recall/checkpoints/                     E1 checkpoints (recency and learned,
                                                 capacity 32, seed 0) for the animation
Outputs go to docs/assets/ as <name>-light.* and <name>-dark.*.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import torch
from leak_demo import measure_leaks  # sibling script in scripts/

from engramlm.eval.recall import RecallTask
from engramlm.inspector import build_report
from engramlm.io import load_pretrained
from engramlm.plots import (
    animate_memory,
    both_modes,
    plot_causality,
    plot_lm_results,
    plot_recall_accuracy,
    plot_training_curves,
    recall_timeline,
)

ASSETS = Path("docs/assets")


def recall_figures(results: Path, log: Path) -> None:
    payload = json.loads(results.read_text())
    task = RecallTask(**payload["config"]["task"])
    rows = payload["rows"]
    both_modes(
        plot_recall_accuracy, ASSETS / "recall-accuracy", rows=rows, chance=1 / task.n_values
    )
    if log.exists():
        for capacity in sorted({int(r["capacity"]) for r in rows if r["capacity"]}):
            both_modes(
                plot_training_curves,
                ASSETS / f"training-curves-cap{capacity}",
                log_text=log.read_text(),
                capacity=capacity,
                chance_nll=math.log(task.n_values),
            )


def memory_animation(ckpts: dict[str, Path], config_task: RecallTask, seed: int) -> None:
    models = {name: load_pretrained(path) for name, path in ckpts.items()}
    gen = torch.Generator().manual_seed(seed)
    ids, _ = config_task.sample(1, gen)
    timelines = {name: recall_timeline(model, config_task, ids) for name, model in models.items()}
    capacity = next(iter(models.values())).config.capacity
    both_modes(
        animate_memory, ASSETS / "memory", suffix=".gif", timelines=timelines, capacity=capacity
    )


def inspector_screenshot(checkpoint: Path, task: RecallTask, seed: int) -> None:
    chrome = (
        shutil.which("google-chrome")
        or "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    )
    if not Path(chrome).exists():
        print("Chrome not found; skipping inspector screenshot")
        return
    model = load_pretrained(checkpoint)
    ids, _ = task.sample(1, torch.Generator().manual_seed(seed))
    names = {1: "Q"}
    texts = []
    for t in ids[0].tolist():
        if t in names:
            texts.append(" Q")
        elif task.key_offset <= t < task.value_offset:
            texts.append(f" K{t - task.key_offset}")
        elif task.value_offset <= t < task.filler_offset:
            texts.append(f" V{t - task.value_offset}")
        else:
            texts.append(" ·")
    html = build_report(
        model,
        ids,
        texts,
        ASSETS / "inspector-demo.html",
        segment_len=task.segment_len,
        title="EngramLM memory trace (recall demo)",
    )
    subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--screenshot={(ASSETS / 'inspector.png').resolve()}",
            "--window-size=1280,760",
            html.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recall", default="runs/recall")
    parser.add_argument("--lm", default="runs/lm-smollm2")
    parser.add_argument("--demo-capacity", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)

    both_modes(plot_causality, ASSETS / "causality", entries=measure_leaks())
    recall_results = Path(args.recall) / "results.json"
    if recall_results.exists():
        recall_figures(recall_results, Path(args.recall).with_suffix(".log"))
    lm_results = Path(args.lm) / "results.json"
    if lm_results.exists():
        both_modes(
            plot_lm_results, ASSETS / "lm-results", rows=json.loads(lm_results.read_text())["rows"]
        )
    ckpt_root = Path(args.recall) / "checkpoints"
    ckpts = {
        "Recency (FIFO)": ckpt_root / f"recency-cap{args.demo_capacity}-seed0",
        "Learned salience": ckpt_root / f"learned-cap{args.demo_capacity}-seed0",
    }
    if recall_results.exists() and all(p.exists() for p in ckpts.values()):
        task = RecallTask(**json.loads(recall_results.read_text())["config"]["task"])
        memory_animation(ckpts, task, args.seed)
        inspector_screenshot(ckpts["Learned salience"], task, args.seed)
    for path in sorted(ASSETS.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
