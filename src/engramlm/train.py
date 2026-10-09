"""Training loop with truncated backpropagation through memory.

A document is processed segment by segment while the autograd graph is kept
across segments, so a write in segment ``k`` (its stored vector and its priority)
is trained by the loss of the segments that read it. ``bptt_segments`` bounds how
many segments one backward pass spans; ``0`` backpropagates through the whole
document, which is affordable for short synthetic documents.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor

from engramlm.model import EngramLM
from engramlm.segment import segment_bounds, token_nll_sum


@dataclass
class TrainConfig:
    steps: int = 1000
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 50
    grad_clip: float = 1.0
    bptt_segments: int = 0
    log_every: int = 100
    seed: int = 0

    def __post_init__(self) -> None:
        if self.steps < 1 or self.batch_size < 1:
            raise ValueError("steps and batch_size must be positive")
        if self.bptt_segments < 0:
            raise ValueError("bptt_segments must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_optimizer(model: EngramLM, config: TrainConfig) -> torch.optim.AdamW:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_scalar_or_norm = param.ndim < 2 or "norm" in name
        (no_decay if is_scalar_or_norm else decay).append(param)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config.lr)


def warmup_cosine(config: TrainConfig) -> Callable[[int], float]:
    def factor(step: int) -> float:
        if step < config.warmup_steps:
            return (step + 1) / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(config.steps - config.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return factor


def _segment_labels(ids: Tensor, score_mask: Tensor | None, start: int, end: int) -> Tensor:
    labels = ids[:, start:end].clone()
    if score_mask is not None:
        labels[~score_mask[:, start:end]] = -100
    return labels


def train(
    model: EngramLM,
    batches: Iterable[tuple[Tensor, Tensor | None]],
    *,
    segment_len: int,
    config: TrainConfig,
    log: Callable[[dict[str, float]], None] | None = None,
) -> list[dict[str, float]]:
    """Train on ``(input_ids, score_mask)`` batches; returns the logged history."""
    torch.manual_seed(config.seed)
    device = model.device
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(config))
    history: list[dict[str, float]] = []
    model.train()

    for step, (ids, score_mask) in enumerate(batches, start=1):
        ids = ids.to(device)
        score_mask = score_mask.to(device) if score_mask is not None else None
        bounds = segment_bounds(ids.shape[1], segment_len)
        labels = [_segment_labels(ids, score_mask, s, e) for s, e in bounds]
        total = sum(int((lab[:, 1:] != -100).sum()) for lab in labels)
        if total == 0:
            continue

        optimizer.zero_grad(set_to_none=True)
        memory = model.new_memory(ids.shape[0], device)
        window_loss = torch.zeros((), device=device)
        doc_nll = 0.0
        for index, ((start, end), seg_labels) in enumerate(zip(bounds, labels, strict=True)):
            out = model(ids[:, start:end], memory=memory, position_offset=start)
            nll, _ = token_nll_sum(out.logits, seg_labels)
            window_loss = window_loss + nll / total
            doc_nll += float(nll.detach())
            memory = out.memory
            boundary = config.bptt_segments and (index + 1) % config.bptt_segments == 0
            if boundary or index == len(bounds) - 1:
                if window_loss.requires_grad:
                    torch.autograd.backward(window_loss)
                window_loss = torch.zeros((), device=device)
                memory = memory.detach()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % config.log_every == 0 or step == config.steps:
            record = {
                "step": float(step),
                "loss": doc_nll / total,
                "grad_norm": float(grad_norm),
                "lr": float(scheduler.get_last_lr()[0]),
                "gate": float(torch.tanh(model.reader.gate.detach())),
                "read_norm": float(model.reader.out.weight.detach().norm()),
            }
            history.append(record)
            if log is not None:
                log(record)
        if step >= config.steps:
            break
    model.eval()
    return history
