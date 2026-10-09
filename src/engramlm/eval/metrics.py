"""Language-modelling metrics that are comparable across models and tokenizers.

Perplexity depends on the tokenizer, so models with different vocabularies should
be compared in bits per byte (BPB): total NLL in bits divided by the number of
UTF-8 bytes of the scored text.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

from engramlm.segment import token_nll_sum

LN2 = math.log(2.0)


def perplexity(nll_sum: float, n_tokens: int) -> float:
    return math.exp(nll_sum / n_tokens) if n_tokens else float("nan")


def bits_per_token(nll_sum: float, n_tokens: int) -> float:
    return nll_sum / (LN2 * n_tokens) if n_tokens else float("nan")


def bits_per_byte(nll_sum: float, n_bytes: int) -> float:
    return nll_sum / (LN2 * n_bytes) if n_bytes else float("nan")


@torch.no_grad()
def strided_nll(
    logits_fn: Callable[[Tensor], Tensor],
    input_ids: Tensor,
    *,
    max_length: int,
    stride: int,
) -> tuple[float, int]:
    """Sliding-window NLL for a plain causal LM (the standard long-text protocol).

    Every token after the first is scored exactly once, each with up to
    ``max_length - stride`` tokens of extra left context.
    """
    if not 0 < stride <= max_length:
        raise ValueError("need 0 < stride <= max_length")
    length = input_ids.shape[1]
    nll_total, count_total, prev_end = 0.0, 0, 0
    for begin in range(0, length, stride):
        end = min(begin + max_length, length)
        window = input_ids[:, begin:end]
        labels = window.clone()
        target_len = end - prev_end
        if target_len < window.shape[1]:
            labels[:, : window.shape[1] - target_len] = -100
        nll, count = token_nll_sum(logits_fn(window), labels)
        nll_total += float(nll)
        count_total += int(count)
        prev_end = end
        if end == length:
            break
    return nll_total, count_total
