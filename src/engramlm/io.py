"""Save and load EngramLM checkpoints.

Layout of a checkpoint directory::

    engram_config.json           EngramConfig
    engram_weights.safetensors   reader + scorer weights (+ LoRA adapters, if any)
    base/                        full base model, only when the base itself was trained

For the common case (frozen base, optionally with LoRA) the base model is reloaded
from ``config.base_model_name`` and only the small adapter file is stored.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, PreTrainedModel

from engramlm import __version__
from engramlm.config import EngramConfig
from engramlm.model import EngramLM

CONFIG_NAME = "engram_config.json"
WEIGHTS_NAME = "engram_weights.safetensors"
BASE_DIR = "base"


def _stores_full_base(config: EngramConfig) -> bool:
    return not config.freeze_base and config.lora_rank == 0


def _is_adapter_key(name: str) -> bool:
    return not name.startswith("base.") or "lora_" in name


def save_pretrained(model: EngramLM, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.config.save(directory / CONFIG_NAME)
    if _stores_full_base(model.config):
        model.base.save_pretrained(directory / BASE_DIR)
    tensors = {
        name: tensor.detach().contiguous().cpu().clone()
        for name, tensor in model.state_dict().items()
        if _is_adapter_key(name)
    }
    save_file(tensors, str(directory / WEIGHTS_NAME), metadata={"engramlm_version": __version__})
    return directory


def load_pretrained(
    directory: str | Path,
    base_model: PreTrainedModel | None = None,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> EngramLM:
    directory = Path(directory)
    config = EngramConfig.load(directory / CONFIG_NAME)
    if base_model is None:
        base_dir = directory / BASE_DIR
        source = str(base_dir) if base_dir.exists() else config.base_model_name
        base_model = AutoModelForCausalLM.from_pretrained(source, dtype=dtype)
    model = EngramLM(config, base_model=base_model)
    weights = load_file(str(directory / WEIGHTS_NAME))
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if unexpected:
        raise ValueError(f"checkpoint has unexpected keys: {sorted(unexpected)[:5]}")
    missing_adapters = [name for name in missing if _is_adapter_key(name)]
    if missing_adapters:
        raise ValueError(f"checkpoint is missing keys: {sorted(missing_adapters)[:5]}")
    if device is not None:
        model.to(device)
    return model
