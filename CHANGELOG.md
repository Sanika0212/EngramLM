# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-09

### Added

- `EngramLM` wrapper for Hugging Face causal LMs with per-token, past-only memory
  reads injected at a configurable decoder layer.
- Explicit `MemoryState` with vectorized writes and `priority` / `fifo` eviction.
- Salience scorers: `surprise`, `learned` (trained through later reads), `random`,
  `recency`.
- Sleep-phase consolidation that merges near-duplicate slots and conserves
  priority mass.
- Optional low-rank stale-memory adapter.
- KV-cached generation that reads memory; segment-by-segment document scoring.
- Adapter-only checkpoints (`save_pretrained` / `load_pretrained`), with full base
  weights stored only when the base itself was trained; optional LoRA via PEFT.
- `engramlm audit`: numerical causality audit for any model or memory wrapper.
- HTML memory inspector showing salience, writes and read provenance.
- Evaluation toolkit: bits per byte, strided perplexity, bootstrap intervals.
- Experiments E1 (cross-segment associative recall) and E2 (long-document
  language modelling with LoRA-only and full-context controls).
- CLI, experiment configs, Slurm script, CUDA Dockerfile, CI on Python 3.11-3.13.

### Fixed during development

- The reader originally used a zero-initialised gate, which blocked gradients to
  every reader weight; the output projection is now zero-initialised instead.
- Study pairs in the recall benchmark are placed at a random offset so no scorer
  is favoured by position.
