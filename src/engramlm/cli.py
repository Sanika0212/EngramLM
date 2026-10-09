"""Command-line interface: ``engramlm {audit,recall,lm,inspect,version}``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
import yaml

from engramlm import __version__


def _load_yaml(path: str) -> dict[str, object]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a mapping")
    return data


def _tiny_model() -> tuple[object, int]:
    from transformers import LlamaConfig, LlamaForCausalLM

    from engramlm.config import EngramConfig
    from engramlm.model import EngramLM

    torch.manual_seed(0)
    vocab = 128
    base = LlamaForCausalLM(  # type: ignore[no-untyped-call]
        LlamaConfig(
            vocab_size=vocab,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=256,
            bos_token_id=None,
            eos_token_id=None,
        )
    )
    config = EngramConfig(capacity=64, writes_per_segment=16, read_top_k=8, key_dim=32)
    return EngramLM(config, base_model=base), vocab


def cmd_audit(args: argparse.Namespace) -> int:
    from engramlm.audit import audit_engram
    from engramlm.config import EngramConfig
    from engramlm.io import load_pretrained
    from engramlm.model import EngramLM

    if args.tiny:
        model, vocab = _tiny_model()
    elif args.checkpoint:
        model = load_pretrained(args.checkpoint)
        vocab = int(model.base.config.get_text_config().vocab_size)
    else:
        model = EngramLM(EngramConfig(base_model_name=args.model))
        vocab = int(model.base.config.get_text_config().vocab_size)
    assert isinstance(model, EngramLM)
    report = audit_engram(
        model,
        vocab_size=vocab,
        segment_len=args.segment_len,
        warmup_segments=args.warmup_segments,
        trials=args.trials,
        probe=not args.no_probe,
    )
    print(report.summary())
    return 0 if report.passed else 1


def cmd_recall(args: argparse.Namespace) -> int:
    from engramlm.experiments.recall import RecallExperimentConfig, run_experiment

    config = RecallExperimentConfig.from_dict(_load_yaml(args.config) if args.config else {})
    run_experiment(config, args.out, device=args.device)
    print((Path(args.out) / "results.md").read_text())
    return 0


def cmd_lm(args: argparse.Namespace) -> int:
    from engramlm.experiments.lm import LMExperimentConfig, run_lm_experiment

    config = LMExperimentConfig.from_dict(_load_yaml(args.config))
    run_lm_experiment(config, args.out, device=args.device)
    print((Path(args.out) / "results.md").read_text())
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    from engramlm.inspector import build_report
    from engramlm.io import load_pretrained

    model = load_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or model.config.base_model_name)
    text = Path(args.text).read_text()
    ids = tokenizer(text, return_tensors="pt").input_ids[:, : args.max_tokens]
    texts = [str(tokenizer.decode([int(t)])) for t in ids[0]]
    path = build_report(model, ids, texts, args.out, segment_len=args.segment_len)
    print(f"wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="engramlm", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="numerically check that a model cannot see the future")
    source = audit.add_mutually_exclusive_group(required=True)
    source.add_argument("--tiny", action="store_true", help="audit a tiny random model (offline)")
    source.add_argument("--model", help="Hugging Face model id to wrap with default settings")
    source.add_argument("--checkpoint", help="EngramLM checkpoint directory")
    audit.add_argument("--segment-len", type=int, default=32)
    audit.add_argument("--warmup-segments", type=int, default=2)
    audit.add_argument("--trials", type=int, default=8)
    audit.add_argument("--no-probe", action="store_true", help="audit the weights as they are")
    audit.set_defaults(func=cmd_audit)

    recall = sub.add_parser("recall", help="run the associative-recall benchmark (E1)")
    recall.add_argument("--config", help="YAML experiment config (defaults if omitted)")
    recall.add_argument("--out", required=True)
    recall.add_argument("--device", default="cpu")
    recall.set_defaults(func=cmd_recall)

    lm = sub.add_parser("lm", help="run the long-document language-modelling experiment (E2)")
    lm.add_argument("--config", required=True)
    lm.add_argument("--out", required=True)
    lm.add_argument("--device", default="cpu")
    lm.set_defaults(func=cmd_lm)

    inspect = sub.add_parser("inspect", help="render an HTML memory trace for a text file")
    inspect.add_argument("--checkpoint", required=True)
    inspect.add_argument("--text", required=True)
    inspect.add_argument("--out", required=True)
    inspect.add_argument("--tokenizer", help="defaults to the checkpoint's base model")
    inspect.add_argument("--segment-len", type=int, default=64)
    inspect.add_argument("--max-tokens", type=int, default=1024)
    inspect.set_defaults(func=cmd_inspect)

    version = sub.add_parser("version", help="print the installed version")
    version.set_defaults(func=lambda _: print(__version__) or 0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
