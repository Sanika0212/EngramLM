"""Evaluation: tokenizer-agnostic metrics, bootstrap statistics, recall benchmark."""

from engramlm.eval.metrics import bits_per_byte, bits_per_token, perplexity, strided_nll
from engramlm.eval.recall import RecallResult, RecallTask, evaluate_recall
from engramlm.eval.stats import Interval, bootstrap_ci, paired_bootstrap_diff

__all__ = [
    "Interval",
    "RecallResult",
    "RecallTask",
    "bits_per_byte",
    "bits_per_token",
    "bootstrap_ci",
    "evaluate_recall",
    "paired_bootstrap_diff",
    "perplexity",
    "strided_nll",
]
