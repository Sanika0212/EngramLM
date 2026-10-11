# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-09

### Added

1. `EngramLM` wrapper for Hugging Face causal LMs with per-token, past-only memory
   reads injected at a configurable decoder layer.
2. Explicit `MemoryState` with vectorized writes and `priority` / `fifo` eviction.
3. Salience scorers: `surprise`, `learned` (trained through later reads), `random`,
   `recency`.
4. Sleep-phase consolidation that merges near-duplicate slots and conserves
   priority mass.
5. Optional low-rank stale-memory adapter.
6. KV-cached generation that reads memory; segment-by-segment document scoring.
7. Adapter-only checkpoints (`save_pretrained` / `load_pretrained`), with full base
   weights stored only when the base itself was trained; optional LoRA via PEFT.
8. `engramlm audit`: numerical causality audit for any model or memory wrapper.
9. HTML memory inspector showing salience, writes and read provenance.
10. Evaluation toolkit: bits per byte, strided perplexity, bootstrap intervals.
11. Experiments E1 (cross-segment associative recall) and E2 (long-document
    language modelling with LoRA-only and full-context controls).
12. CLI, experiment configs, Slurm script, CUDA Dockerfile, CI on Python 3.11-3.13.

### Fixed during development

1. The reader originally used a zero-initialised gate, which blocked gradients to
   every reader weight; the output projection is now zero-initialised instead.
2. Study pairs in the recall benchmark are placed at a random offset so no scorer
   is favoured by position.
