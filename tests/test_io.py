from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from engramlm.config import EngramConfig
from engramlm.io import BASE_DIR, WEIGHTS_NAME, load_pretrained, save_pretrained
from engramlm.model import EngramLM
from tests.conftest import VOCAB, make_tiny_llama

SMALL = {"capacity": 16, "writes_per_segment": 4, "read_top_k": 4, "key_dim": 8}


def _perturb(model: EngramLM) -> None:
    with torch.no_grad():
        for param in model.trainable_parameters():
            param.add_(0.05 * torch.randn_like(param))


def _logits(model: EngramLM, ids: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        memory = model(ids).memory
        logits: torch.Tensor = model(ids, memory=memory).logits
        return logits


def test_round_trip_with_frozen_base(tmp_path: Path) -> None:
    model = EngramLM(EngramConfig(scorer="learned", **SMALL), base_model=make_tiny_llama())  # type: ignore[arg-type]
    _perturb(model)
    save_pretrained(model, tmp_path)
    assert not (tmp_path / BASE_DIR).exists()
    stored = load_file(str(tmp_path / WEIGHTS_NAME))
    assert not any(name.startswith("base.") for name in stored)

    loaded = load_pretrained(tmp_path, base_model=make_tiny_llama())
    ids = torch.randint(3, VOCAB, (2, 10))
    assert torch.equal(_logits(model, ids), _logits(loaded, ids))


def test_round_trip_with_trained_base(tmp_path: Path) -> None:
    model = EngramLM(EngramConfig(freeze_base=False, **SMALL), base_model=make_tiny_llama())  # type: ignore[arg-type]
    _perturb(model)
    save_pretrained(model, tmp_path)
    assert (tmp_path / BASE_DIR).exists()
    loaded = load_pretrained(tmp_path)  # no base model passed: rebuilt from base/
    ids = torch.randint(3, VOCAB, (1, 8))
    assert torch.allclose(_logits(model, ids), _logits(loaded, ids))


def test_round_trip_with_lora(tmp_path: Path) -> None:
    pytest.importorskip("peft")
    model = EngramLM(EngramConfig(lora_rank=2, **SMALL), base_model=make_tiny_llama())  # type: ignore[arg-type]
    _perturb(model)
    save_pretrained(model, tmp_path)
    stored = load_file(str(tmp_path / WEIGHTS_NAME))
    assert any("lora_" in name for name in stored)
    loaded = load_pretrained(tmp_path, base_model=make_tiny_llama())
    ids = torch.randint(3, VOCAB, (1, 8))
    assert torch.allclose(_logits(model, ids), _logits(loaded, ids))


def test_unexpected_keys_are_rejected(tmp_path: Path) -> None:
    model = EngramLM(EngramConfig(**SMALL), base_model=make_tiny_llama())  # type: ignore[arg-type]
    save_pretrained(model, tmp_path)
    weights = load_file(str(tmp_path / WEIGHTS_NAME))
    weights["reader.not_a_parameter"] = torch.zeros(1)
    save_file(weights, str(tmp_path / WEIGHTS_NAME))
    with pytest.raises(ValueError, match="unexpected"):
        load_pretrained(tmp_path, base_model=make_tiny_llama())
