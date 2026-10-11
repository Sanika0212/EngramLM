# Evaluation protocol

## Principles

1. **Causality is checked, not assumed.** Every variant passes `engramlm audit`
   before its numbers are reported.
2. **Same targets, same bytes.** All variants in an experiment score exactly the
   same target tokens. Under segmentation the first token of each segment has no
   in-segment prediction, so it is excluded everywhere, including from the
   full-context reference.
3. **Tokenizer-agnostic headline metric.** Perplexity depends on the vocabulary.
   Bits per byte (total NLL in bits divided by UTF-8 bytes of the scored targets)
   is reported next to it.
4. **Controls that isolate the memory.** A LoRA-only model trained with identical
   data, steps and optimiser settings is the baseline for any memory variant.
5. **Uncertainty is reported.** Seed-level results are summarised with a
   percentile-bootstrap 95% interval.

## E1: cross-segment associative recall

Each document has a study segment (key/value pairs at a random offset among
filler), `gap_segments` filler segments, and a test segment that queries the
keys. The base model only sees one segment at a time, so without memory the
values in the test segment are at chance (`1 / n_values`). Small Llama-style
models are trained from scratch per variant, capacity and seed; accuracy is
measured on 512 held-out documents.

What it tests: whether the write policy keeps the right tokens under capacity
pressure (gap segments write filler that competes for slots), and whether reads
retrieve them.

Training uses a curriculum: for the first 1,500 steps the gap between study and test
grows from zero to the full task. Without it, many seeds never learned to read memory.
Evaluation always uses the full gap.

```bash
uv run engramlm recall --config configs/recall.yaml --out runs/recall
```

## E1b: choosing what to write

This is E1 with `writes_per_segment: 4`. All writes fit in memory, so eviction never
happens and the scorer alone decides which tokens are stored. It compares recency,
random, surprise and learned scorers at capacity 32, with three seeds.

```bash
uv run engramlm recall --config configs/recall-tight.yaml --out runs/recall-tight
```

## E2: long-document language modelling

WikiText-2 articles of at least `doc_len` tokens are cut into `segment_len`
segments. Variants: frozen `base`, `lora` (control), `lora+recency`,
`lora+surprise`, `lora+learned`, and a `full-context` reference in which the base
model attends to the whole document. Memory should move a variant from `lora`
towards `full-context`.

```bash
uv run engramlm lm --config configs/lm-smollm2.yaml --out runs/lm-smollm2 --device mps
```

A GPU-scale configuration (`configs/lm-gemma2b-ccr.yaml`, WikiText-103, Gemma-2
2B, three seeds) and a Slurm script (`scripts/ccr.slurm`) are included but have
not been run yet.

## Auditing a model

```bash
uv run engramlm audit --tiny                     # offline smoke test
uv run engramlm audit --model HuggingFaceTB/SmolLM2-135M
uv run engramlm audit --checkpoint runs/my-checkpoint
```

By default the audit temporarily gives the reader a random output projection
(`probe_reads`) so that a leak through memory would be visible even in an
untrained model. Pass `--no-probe` to audit the weights exactly as they are.
