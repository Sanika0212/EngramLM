# Contributing

Thanks for your interest in EngramLM.

## Development setup

```bash
git clone https://github.com/Sanika0212/EngramLM.git
cd EngramLM
uv sync --extra train
uv run pre-commit install
```

## Checks

Every pull request must pass the same checks CI runs:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy
uv run pytest --cov --cov-fail-under=85
uv run engramlm audit --tiny
```

Tests use tiny randomly initialised models and never download weights. Please
keep it that way for new tests.

## Ground rules

1. **Causality is non-negotiable.** Any change to the read or write path must keep
   `tests/test_model.py::test_wrapper_is_causal` passing with an exact-zero delta.
2. **Identity at init.** An untrained wrapper must reproduce its base model.
3. **Results come from runs.** Numbers in the README must be reproducible from a
   config in `configs/` and the commands in `docs/evaluation.md`.
4. Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`,
   `fix:`, `docs:`, `test:`, `chore:`, `ci:`).
