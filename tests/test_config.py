from pathlib import Path

import pytest

from engramlm.config import EngramConfig


def test_defaults_are_valid() -> None:
    cfg = EngramConfig()
    assert cfg.memory_enabled
    assert cfg.scorer == "surprise"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"capacity": -1}, "capacity"),
        ({"read_top_k": 0}, "read_top_k"),
        ({"decay": 0.0}, "decay"),
        ({"eviction": "lru"}, "eviction"),
        ({"scorer": "magic"}, "scorer"),
        ({"consolidate_threshold": 1.5}, "consolidate_threshold"),
    ],
)
def test_invalid_values_raise(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EngramConfig(**kwargs)  # type: ignore[arg-type]


def test_multiple_errors_are_reported_together() -> None:
    with pytest.raises(ValueError, match=r"capacity.*read_top_k"):
        EngramConfig(capacity=-1, read_top_k=0)


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_round_trip(tmp_path: Path, suffix: str) -> None:
    cfg = EngramConfig(capacity=64, scorer="learned", lora_targets=("q_proj",))
    path = tmp_path / f"cfg{suffix}"
    if suffix == ".json":
        cfg.save(path)
    else:
        import yaml

        path.write_text(yaml.safe_dump(cfg.to_dict()))
    assert EngramConfig.load(path) == cfg


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        EngramConfig.from_dict({"capacity": 8, "bogus": 1})
