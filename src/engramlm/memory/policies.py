"""Eviction policies decide which slots survive when the bank overflows.

A policy maps slot metadata to a retention score; the bank keeps the ``capacity``
slots with the highest score. Retention is computed on detached priorities: which
slot is kept is a discrete choice, while the priority value itself stays
differentiable for the read path.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

RetentionFn = Callable[[Tensor, Tensor], Tensor]


def priority_retention(decay: float) -> RetentionFn:
    """Keep slots with the highest ``priority * decay ** age``."""

    def retention(priority: Tensor, age: Tensor) -> Tensor:
        return priority.detach() * torch.pow(torch.full_like(age, decay), age)

    return retention


def fifo_retention() -> RetentionFn:
    """Keep the most recently written slots, ignoring priority."""

    def retention(priority: Tensor, age: Tensor) -> Tensor:
        return -age

    return retention


def build_policy(name: str, decay: float) -> RetentionFn:
    if name == "priority":
        return priority_retention(decay)
    if name == "fifo":
        return fifo_retention()
    raise ValueError(f"Unknown eviction policy {name!r}")
