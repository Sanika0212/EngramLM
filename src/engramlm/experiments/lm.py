"""Experiment E2: long-document language modelling with a pretrained base model.

Articles are split into segments that the base model sees one at a time. Variants:

- ``base``: the frozen base model, no training, no memory (segment-local context);
- ``lora``: LoRA fine-tuning only, no memory: the control that isolates what the
  memory adds beyond fine-tuning on the domain;
- ``lora+<scorer>``: LoRA plus memory with the given salience scorer, trained with
  exactly the same data, steps and optimiser settings as ``lora``;
- ``full-context`` (reference): the base model with a sliding window spanning the
  whole document, i.e. what unlimited attention would achieve.

Reported metrics are token-weighted perplexity and bits per byte on held-out
articles.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from engramlm.config import EngramConfig
from engramlm.eval.metrics import bits_per_byte, perplexity
from engramlm.model import EngramLM
from engramlm.segment import evaluate_document, token_nll_sum
from engramlm.train import TrainConfig, train

TRAINED_VARIANTS: dict[str, dict[str, Any]] = {
    "lora": {"memory_enabled": False},
    "lora+recency": {"scorer": "recency", "eviction": "fifo"},
    "lora+surprise": {"scorer": "surprise"},
    "lora+learned": {"scorer": "learned"},
}


@dataclass
class LMExperimentConfig:
    base_model_name: str = "HuggingFaceTB/SmolLM2-135M"
    dataset: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    train_split: str = "train"
    eval_split: str = "test"
    segment_len: int = 128
    doc_len: int = 512
    min_doc_tokens: int = 512
    max_eval_docs: int = 40
    variants: list[str] = field(default_factory=lambda: ["base", "full-context", *TRAINED_VARIANTS])
    memory: dict[str, Any] = field(
        default_factory=lambda: {
            "capacity": 256,
            "writes_per_segment": 32,
            "read_top_k": 16,
            "key_dim": 128,
            "memory_layer": -4,
        }
    )
    lora_rank: int = 8
    seeds: list[int] = field(default_factory=lambda: [0])
    train: TrainConfig = field(
        default_factory=lambda: TrainConfig(steps=300, batch_size=4, lr=5e-4, bptt_segments=2)
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LMExperimentConfig:
        data = dict(data)
        train_cfg = TrainConfig(**data.pop("train", {}))
        config = cls(train=train_cfg, **data)
        known = {"base", "full-context", *TRAINED_VARIANTS}
        unknown = [v for v in config.variants if v not in known]
        if unknown:
            raise ValueError(f"unknown variants {unknown}; choose from {sorted(known)}")
        return config

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["train"] = self.train.to_dict()
        return data


def split_articles(lines: list[str]) -> list[str]:
    """Group WikiText lines into articles at top-level ``= Title =`` headings."""
    articles: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        is_title = (
            stripped.startswith("= ") and stripped.endswith(" =") and not stripped.startswith("= =")
        )
        if is_title or not articles:
            articles.append([])
        articles[-1].append(line)
    return ["".join(article).strip() for article in articles if "".join(article).strip()]


def load_token_docs(config: LMExperimentConfig, tokenizer: Any, split: str) -> list[Tensor]:
    from datasets import load_dataset

    rows = load_dataset(config.dataset, config.dataset_config, split=split)
    docs = []
    for article in split_articles([row["text"] for row in rows]):
        ids = tokenizer(article, return_tensors="pt").input_ids[0]
        if ids.numel() >= config.min_doc_tokens:
            docs.append(ids)
    return docs


def _windows(
    docs: list[Tensor], doc_len: int, batch_size: int, seed: int
) -> Iterator[tuple[Tensor, None]]:
    rng = random.Random(seed)
    while True:
        batch = []
        for _ in range(batch_size):
            doc = docs[rng.randrange(len(docs))]
            start = rng.randrange(0, doc.numel() - doc_len + 1)
            batch.append(doc[start : start + doc_len])
        yield torch.stack(batch), None


def scored_bytes(tokenizer: Any, doc: Tensor, segment_len: int) -> int:
    """UTF-8 bytes of the targets scored under segmentation (all but segment starts)."""
    total = 0
    for start in range(0, doc.numel(), segment_len):
        for token in doc[start + 1 : start + segment_len]:
            total += len(tokenizer.decode([int(token)]).encode("utf-8"))
    return total


def evaluate_lm(
    model: EngramLM, docs: list[Tensor], segment_len: int, tokenizer: Any
) -> dict[str, float]:
    """Segment-by-segment evaluation (memory carries context across segments)."""
    nll, tokens, n_bytes = 0.0, 0, 0
    for doc in docs:
        result = evaluate_document(
            model, doc.unsqueeze(0).to(model.device), segment_len=segment_len
        )
        nll += result.nll_sum
        tokens += result.n_tokens
        n_bytes += scored_bytes(tokenizer, doc, segment_len)
    return {
        "nll": nll,
        "tokens": tokens,
        "ppl": perplexity(nll, tokens),
        "bpb": bits_per_byte(nll, n_bytes),
    }


@torch.no_grad()
def evaluate_full_context(
    model: EngramLM, docs: list[Tensor], segment_len: int, tokenizer: Any
) -> dict[str, float]:
    """Reference: the base model attends to the whole document.

    Scores exactly the same target tokens as :func:`evaluate_lm`, so the numbers are
    directly comparable.
    """
    nll, tokens, n_bytes = 0.0, 0, 0
    for doc in docs:
        ids = doc.unsqueeze(0).to(model.device)
        labels = ids.clone()
        labels[:, 0::segment_len] = -100
        doc_nll, count = token_nll_sum(model.base(input_ids=ids).logits, labels)
        nll += float(doc_nll)
        tokens += int(count)
        n_bytes += scored_bytes(tokenizer, doc, segment_len)
    return {
        "nll": nll,
        "tokens": tokens,
        "ppl": perplexity(nll, tokens),
        "bpb": bits_per_byte(nll, n_bytes),
    }


def _engram_config(config: LMExperimentConfig, variant: str) -> EngramConfig:
    if variant in {"base", "full-context"}:
        return EngramConfig(base_model_name=config.base_model_name, memory_enabled=False)
    return EngramConfig(
        base_model_name=config.base_model_name,
        lora_rank=config.lora_rank,
        **{**config.memory, **TRAINED_VARIANTS[variant]},
    )


def _progress(log: Callable[[str], None], tag: str) -> Callable[[dict[str, float]], None]:
    def report(record: dict[str, float]) -> None:
        log(f"  [{tag}] step {int(record['step'])} loss {record['loss']:.3f}")

    return report


def run_lm_experiment(
    config: LMExperimentConfig,
    out_dir: str | Path,
    *,
    device: torch.device | str = "cpu",
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    train_docs = load_token_docs(config, tokenizer, config.train_split)
    eval_docs = load_token_docs(config, tokenizer, config.eval_split)[: config.max_eval_docs]
    eval_docs = [d[: config.doc_len] for d in eval_docs]
    log(f"{len(train_docs)} train articles, {len(eval_docs)} eval docs x {config.doc_len} tokens")

    rows: list[dict[str, Any]] = []
    for variant in config.variants:
        seeds = [0] if variant in {"base", "full-context"} else config.seeds
        for seed in seeds:
            torch.manual_seed(seed)
            base = AutoModelForCausalLM.from_pretrained(config.base_model_name, dtype=torch.float32)
            model = EngramLM(_engram_config(config, variant), base_model=base).to(device)
            started = time.perf_counter()
            if variant in TRAINED_VARIANTS:
                train_cfg = TrainConfig(**{**config.train.to_dict(), "seed": seed})
                train(
                    model,
                    _windows(train_docs, config.doc_len, train_cfg.batch_size, seed),
                    segment_len=config.segment_len,
                    config=train_cfg,
                    log=_progress(log, f"{variant} seed={seed}"),
                )
            model.eval()
            with torch.no_grad():
                metrics = (
                    evaluate_full_context(model, eval_docs, config.segment_len, tokenizer)
                    if variant == "full-context"
                    else evaluate_lm(model, eval_docs, config.segment_len, tokenizer)
                )
            row = {
                "variant": variant,
                "seed": seed,
                "seconds": round(time.perf_counter() - started, 1),
                **metrics,
            }
            summary = f"ppl {row['ppl']:.3f} bpb {row['bpb']:.4f} ({row['seconds']}s)"
            log(f"{variant} seed={seed}: {summary}")
            rows.append(row)
            (out / "results.json").write_text(
                json.dumps({"config": config.to_dict(), "rows": rows}, indent=2) + "\n"
            )
            del model, base

    lines = [
        "| Variant | Seeds | Perplexity | Bits per byte |",
        "|---|---|---|---|",
    ]
    for variant in config.variants:
        picked = [r for r in rows if r["variant"] == variant]
        if not picked:
            continue
        ppl = sum(r["ppl"] for r in picked) / len(picked)
        bpb = sum(r["bpb"] for r in picked) / len(picked)
        lines.append(f"| {variant} | {len(picked)} | {ppl:.3f} | {bpb:.4f} |")
    (out / "results.md").write_text("\n".join(lines) + "\n")
    return rows
