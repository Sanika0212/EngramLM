"""EngramLM: causal, salience-gated external memory for causal language models."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("engramlm")
except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
    __version__ = "0.0.0"

__all__ = ["__version__"]
