"""Tiny randomly initialised models so tests never download weights."""

from collections.abc import Callable

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM, PreTrainedModel

VOCAB = 97


def make_tiny_llama(seed: int = 0, layers: int = 3, hidden: int = 32) -> PreTrainedModel:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    return LlamaForCausalLM(config).eval()


def make_tiny_gpt2(seed: int = 0) -> PreTrainedModel:
    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=VOCAB,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_positions=256,
        bos_token_id=1,
        eos_token_id=2,
    )
    return GPT2LMHeadModel(config).eval()


@pytest.fixture
def tiny_llama() -> PreTrainedModel:
    return make_tiny_llama()


@pytest.fixture(params=["llama", "gpt2"])
def tiny_base(request: pytest.FixtureRequest) -> PreTrainedModel:
    factory: Callable[[], PreTrainedModel] = (
        make_tiny_llama if request.param == "llama" else make_tiny_gpt2
    )
    return factory()
