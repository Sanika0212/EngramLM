"""``EngramLM``: wrap a Hugging Face causal LM with a causal external memory.

One call processes one segment:

1. the base model runs on the segment; a forward hook on ``memory_layer`` lets every
   token read from ``memory`` (which only holds earlier segments);
2. logits (and the loss, when ``labels`` are given) are computed;
3. a salience scorer picks tokens from this segment, and they are written into a
   *new* memory state that is returned for the next segment.

Nothing written in step 3 is visible to the segment that produced it, so the
wrapper is causal by construction (``engramlm.audit`` checks this numerically).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, PreTrainedModel

from engramlm.config import EngramConfig
from engramlm.hooks import find_decoder_layers, resolve_layer_index
from engramlm.memory import MemoryState, build_policy, consolidate, write
from engramlm.reader import MemoryReader, ReadInfo
from engramlm.salience import SalienceContext, build_scorer


@dataclass
class EngramOutput:
    logits: Tensor
    memory: MemoryState
    loss: Tensor | None = None
    past_key_values: Any = None
    priority: Tensor | None = None  # (B, T) salience used for this segment's write
    written: Tensor | None = None  # (B, T) tokens that entered memory
    reads: ReadInfo | None = None  # what each token read (when return_reads=True)
    merges: Tensor | None = None  # (B,) consolidation merges after this segment


@dataclass
class _HookContext:
    memory: MemoryState
    read_enabled: bool
    return_reads: bool
    captured: Tensor | None = None
    reads: ReadInfo | None = None


def causal_lm_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Mean next-token cross-entropy, ignoring ``-100`` labels."""
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


def _apply_lora(model: PreTrainedModel, config: EngramConfig) -> PreTrainedModel:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("LoRA requires the train extra: pip install 'engramlm[train]'") from exc
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.lora_targets),
        lora_dropout=0.0,
        bias="none",
    )
    return cast(PreTrainedModel, get_peft_model(model, lora))


class EngramLM(nn.Module):
    def __init__(
        self,
        config: EngramConfig,
        base_model: PreTrainedModel | None = None,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.config = config
        if base_model is None:
            base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name, dtype=dtype)
        layers = find_decoder_layers(base_model)
        self.memory_layer = resolve_layer_index(config.memory_layer, len(layers))
        text_config = base_model.config.get_text_config()
        self.hidden_dim = int(text_config.hidden_size)

        if config.freeze_base:
            for param in base_model.parameters():
                param.requires_grad_(False)
        if config.lora_rank > 0:
            base_model = _apply_lora(base_model, config)
        self.base = base_model

        self.reader = MemoryReader(
            self.hidden_dim, config.key_dim, config.read_top_k, drift_rank=config.drift_rank
        )
        self.scorer = build_scorer(config.scorer, self.hidden_dim)
        self._retention = build_policy(config.eviction, config.decay)
        self._ctx: _HookContext | None = None
        layers[self.memory_layer].register_forward_hook(self._layer_hook)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def new_memory(self, batch_size: int, device: torch.device | None = None) -> MemoryState:
        capacity = self.config.capacity if self.config.memory_enabled else 0
        return MemoryState.empty(
            batch_size, capacity, self.hidden_dim, device=device or self.device
        )

    def _layer_hook(self, module: nn.Module, args: tuple[Any, ...], output: Any) -> Any:
        ctx = self._ctx
        if ctx is None:
            return None
        hidden = output[0] if isinstance(output, tuple) else output
        ctx.captured = hidden
        if not ctx.read_enabled:
            return None
        fused, ctx.reads = self.reader(hidden, ctx.memory, return_reads=ctx.return_reads)
        if fused is hidden:
            return None
        return (fused, *output[1:]) if isinstance(output, tuple) else fused

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        memory: MemoryState | None = None,
        labels: Tensor | None = None,
        *,
        write_memory: bool = True,
        past_key_values: Any = None,
        use_cache: bool = False,
        return_reads: bool = False,
        position_offset: int = 0,
        generator: torch.Generator | None = None,
    ) -> EngramOutput:
        """Process one segment.

        Args:
            input_ids: ``(B, T)`` token ids of the segment.
            attention_mask: ``(B, T)`` with 0 for padding. Padding is never written.
            memory: state built from earlier segments (a fresh empty one if ``None``).
            labels: ``(B, T)`` targets for the next-token loss (``-100`` = ignore).
            write_memory: score and write this segment into the returned memory.
            past_key_values / use_cache: passed through for incremental decoding.
            return_reads: also return which slots each token read.
            position_offset: absolute position of the first token (for inspection).
            generator: RNG for stochastic scorers.
        """
        batch, seq_len = input_ids.shape
        if memory is None:
            memory = self.new_memory(batch, input_ids.device)
        mask = (
            attention_mask.bool()
            if attention_mask is not None
            else torch.ones(batch, seq_len, dtype=torch.bool, device=input_ids.device)
        )

        ctx = _HookContext(
            memory=memory, read_enabled=self.config.memory_enabled, return_reads=return_reads
        )
        self._ctx = ctx
        try:
            out = self.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        finally:
            self._ctx = None

        logits: Tensor = out.logits
        result = EngramOutput(
            logits=logits,
            memory=memory,
            loss=causal_lm_loss(logits, labels) if labels is not None else None,
            past_key_values=out.past_key_values if use_cache else None,
            reads=ctx.reads,
        )
        if not (write_memory and self.config.memory_enabled and memory.capacity > 0):
            return result
        if ctx.captured is None:  # pragma: no cover - the hook always fires
            raise RuntimeError("memory layer hook did not run")

        priority = self.scorer(
            SalienceContext(
                hidden=ctx.captured,
                logits=logits,
                input_ids=input_ids,
                mask=mask,
                generator=generator,
            )
        )
        positions = torch.arange(seq_len, device=input_ids.device) + position_offset
        written = write(
            memory,
            ctx.captured,
            priority,
            budget=self.config.writes_per_segment,
            retention=self._retention,
            mask=mask,
            positions=positions.expand(batch, seq_len),
        )
        result.memory, result.priority, result.written = written.memory, priority, written.written

        every = self.config.consolidate_every
        if every and result.memory.segments_seen % every == 0:
            # At most an eighth of the bank merges per sleep, so a burst of similar
            # hidden states early in training cannot collapse the whole memory.
            limit = max(1, self.config.capacity // 8)
            result.memory, stats = consolidate(
                result.memory, self.config.consolidate_threshold, max_merges=limit
            )
            result.merges = stats.merges
        return result
