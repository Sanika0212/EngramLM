"""Text generation that reads from a fixed memory, with or without a KV cache."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from engramlm.memory import MemoryState
from engramlm.model import EngramLM


def _next_token(
    logits: Tensor, temperature: float, top_p: float, generator: torch.Generator | None
) -> Tensor:
    if temperature <= 0.0:
        return logits.argmax(dim=-1)
    probs = F.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, order = probs.sort(dim=-1, descending=True)
        keep = sorted_probs.cumsum(dim=-1) - sorted_probs < top_p
        sorted_probs = sorted_probs * keep
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    if generator is not None and generator.device != probs.device:
        return torch.multinomial(probs.cpu(), 1, generator=generator).squeeze(-1).to(probs.device)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate(
    model: EngramLM,
    input_ids: Tensor,
    *,
    memory: MemoryState | None = None,
    max_new_tokens: int = 32,
    eos_token_id: int | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
    use_cache: bool = True,
) -> Tensor:
    """Extend ``input_ids`` by up to ``max_new_tokens`` tokens.

    Memory is read but never written during generation. Streams that emit
    ``eos_token_id`` keep emitting it until every stream has finished.
    """
    batch = input_ids.shape[0]
    memory = memory if memory is not None else model.new_memory(batch, input_ids.device)
    finished = torch.zeros(batch, dtype=torch.bool, device=input_ids.device)
    sequence = input_ids
    past = None
    step_input = input_ids

    for _ in range(max_new_tokens):
        if use_cache:
            out = model(
                step_input,
                memory=memory,
                write_memory=False,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
        else:
            out = model(sequence, memory=memory, write_memory=False)
        token = _next_token(out.logits[:, -1], temperature, top_p, generator)
        if eos_token_id is not None:
            token = torch.where(finished, torch.full_like(token, eos_token_id), token)
            finished |= token == eos_token_id
        sequence = torch.cat([sequence, token.unsqueeze(1)], dim=1)
        step_input = token.unsqueeze(1)
        if eos_token_id is not None and bool(finished.all()):
            break
    return sequence
