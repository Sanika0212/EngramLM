"""Locate decoder layers in common Hugging Face causal LM architectures."""

from __future__ import annotations

from torch import nn

_LAYER_PATHS = (
    "model.layers",  # Llama, Mistral, Qwen2/3, Gemma, SmolLM, Phi-3
    "transformer.h",  # GPT-2, GPT-J, Falcon
    "gpt_neox.layers",  # Pythia / GPT-NeoX
    "model.decoder.layers",  # OPT
    "transformer.blocks",  # MPT
)


def find_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Return the list of decoder blocks of ``model``.

    Raises ``ValueError`` listing the attribute paths that were tried.
    """
    for path in _LAYER_PATHS:
        node: object = model
        for attr in path.split("."):
            node = getattr(node, attr, None)
            if node is None:
                break
        if isinstance(node, nn.ModuleList) and len(node) > 0:
            return node
    raise ValueError(
        f"Could not find decoder layers on {type(model).__name__}; tried {', '.join(_LAYER_PATHS)}"
    )


def resolve_layer_index(index: int, num_layers: int) -> int:
    resolved = index if index >= 0 else num_layers + index
    if not 0 <= resolved < num_layers:
        raise ValueError(f"memory_layer {index} is out of range for {num_layers} layers")
    return resolved
