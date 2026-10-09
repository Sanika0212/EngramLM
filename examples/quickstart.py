"""Wrap a small pretrained model with EngramLM, process a long text, generate, and audit.

    uv run python examples/quickstart.py

Downloads SmolLM2-135M (about 270 MB) on first run. The memory weights here are
untrained, so the wrapper behaves exactly like the base model; train it with
``engramlm.train.train`` (see ``engramlm.experiments.lm``) to make memory useful.
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer

from engramlm.audit import audit_engram
from engramlm.config import EngramConfig
from engramlm.generation import generate
from engramlm.model import EngramLM
from engramlm.segment import evaluate_document

MODEL = "HuggingFaceTB/SmolLM2-135M"
TEXT = (
    "The lighthouse keeper, Ada Brennan, kept a logbook in which she wrote the colour of "
    "the sea every morning. " * 20
    + "Years later, when the logbook was found, the first line read: the keeper's name was"
)


def main() -> None:
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = EngramLM(EngramConfig(base_model_name=MODEL, capacity=256, writes_per_segment=32))
    model.eval()

    ids = tokenizer(TEXT, return_tensors="pt").input_ids
    result = evaluate_document(model, ids, segment_len=64)
    print(f"{ids.shape[1]} tokens in {len(result.segment_nll)} segments")
    print(f"mean NLL {result.mean_nll:.3f}; memory slots used {int(result.memory.num_valid()[0])}")

    prompt = ids[:, -16:]
    output = generate(model, prompt, memory=result.memory, max_new_tokens=8)
    print("continuation:", tokenizer.decode(output[0, prompt.shape[1] :]))

    vocab = model.base.config.get_text_config().vocab_size
    print(audit_engram(model, vocab_size=vocab, segment_len=32, trials=4).summary())


if __name__ == "__main__":
    main()
