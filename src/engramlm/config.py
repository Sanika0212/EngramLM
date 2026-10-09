"""Typed configuration for EngramLM models.

Configs are plain dataclasses so they can be built in code, loaded from YAML, or
round-tripped through ``config.json`` next to saved weights. Validation happens
eagerly in ``__post_init__`` and raises ``ValueError`` with a readable message.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

SCORERS = ("surprise", "learned", "random", "recency")
EVICTION_POLICIES = ("priority", "fifo")


@dataclass
class EngramConfig:
    """Hyperparameters for an :class:`~engramlm.model.EngramLM`.

    Attributes:
        base_model_name: Hugging Face model id or local path of the causal LM to wrap.
        memory_layer: Decoder layer whose output reads from memory. Negative values
            count from the last layer (``-1`` is the final decoder layer).
        capacity: Number of memory slots kept per stream (batch row).
        writes_per_segment: Maximum number of tokens written after each segment.
        read_top_k: Number of memory slots each token attends to.
        key_dim: Width of the query/key space used for retrieval.
        decay: Per-segment multiplicative decay of a slot's retention score.
        eviction: ``"priority"`` keeps slots with the highest decayed priority,
            ``"fifo"`` keeps the most recent slots.
        scorer: Salience scorer deciding which tokens get written and with what
            priority. One of ``surprise``, ``learned``, ``random``, ``recency``.
        consolidate_every: Run sleep-phase consolidation every N segments (0 = off).
        consolidate_threshold: Cosine similarity above which two slots are merged.
        drift_rank: Rank of the stale-memory adapter applied at read time (0 = off).
        freeze_base: Freeze the base model's own weights.
        lora_rank: Rank of LoRA adapters on the base model (0 = no LoRA).
        lora_alpha: LoRA scaling numerator.
        lora_targets: Module names receiving LoRA adapters.
        memory_enabled: When ``False`` the wrapper never reads or writes memory, which
            gives the no-memory (or LoRA-only) baseline under the same code path.
    """

    base_model_name: str = "HuggingFaceTB/SmolLM2-135M"
    memory_layer: int = -2
    capacity: int = 1024
    writes_per_segment: int = 64
    read_top_k: int = 32
    key_dim: int = 128
    decay: float = 0.98
    eviction: str = "priority"
    scorer: str = "surprise"
    consolidate_every: int = 0
    consolidate_threshold: float = 0.9
    drift_rank: int = 0
    freeze_base: bool = True
    lora_rank: int = 0
    lora_alpha: int = 16
    lora_targets: tuple[str, ...] = field(default=("q_proj", "v_proj"))
    memory_enabled: bool = True

    def __post_init__(self) -> None:
        self.lora_targets = tuple(self.lora_targets)
        checks: list[tuple[bool, str]] = [
            (self.capacity >= 0, "capacity must be >= 0"),
            (self.writes_per_segment >= 0, "writes_per_segment must be >= 0"),
            (self.read_top_k >= 1, "read_top_k must be >= 1"),
            (self.key_dim >= 1, "key_dim must be >= 1"),
            (0.0 < self.decay <= 1.0, "decay must be in (0, 1]"),
            (self.eviction in EVICTION_POLICIES, f"eviction must be one of {EVICTION_POLICIES}"),
            (self.scorer in SCORERS, f"scorer must be one of {SCORERS}"),
            (self.consolidate_every >= 0, "consolidate_every must be >= 0"),
            (-1.0 <= self.consolidate_threshold <= 1.0, "consolidate_threshold must be in [-1, 1]"),
            (self.drift_rank >= 0, "drift_rank must be >= 0"),
            (self.lora_rank >= 0, "lora_rank must be >= 0"),
            (self.lora_alpha >= 1, "lora_alpha must be >= 1"),
        ]
        errors = [msg for ok, msg in checks if not ok]
        if errors:
            raise ValueError("Invalid EngramConfig: " + "; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["lora_targets"] = list(self.lora_targets)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EngramConfig:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown EngramConfig keys: {unknown}")
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> EngramConfig:
        """Load from ``.json``, ``.yaml`` or ``.yml``."""
        path = Path(path)
        text = path.read_text()
        data = yaml.safe_load(text) if path.suffix in {".yaml", ".yml"} else json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"{path} does not contain a mapping")
        return cls.from_dict(data)
