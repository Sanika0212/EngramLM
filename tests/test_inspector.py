from pathlib import Path

import pytest
import torch

from engramlm.config import EngramConfig
from engramlm.inspector import build_report, collect_trace, render_html
from engramlm.model import EngramLM
from tests.conftest import VOCAB, activate_reads, make_tiny_llama


def _model(**overrides: object) -> EngramLM:
    settings: dict[str, object] = {
        "capacity": 8,
        "writes_per_segment": 3,
        "read_top_k": 4,
        "key_dim": 8,
    }
    settings.update(overrides)
    model = EngramLM(EngramConfig(**settings), base_model=make_tiny_llama())  # type: ignore[arg-type]
    activate_reads(model.reader)
    return model


def test_trace_records_writes_and_past_only_reads() -> None:
    model = _model()
    ids = torch.randint(3, VOCAB, (1, 30))
    texts = [f"t{i}" for i in range(30)]
    trace = collect_trace(model, ids, texts, segment_len=10)
    assert len(trace.tokens) == 30
    assert [s.segment for s in trace.segments] == [0, 1, 2]
    for seg in range(3):
        written = [t for t in trace.tokens if t.segment == seg and t.written]
        assert len(written) == 3
    first_segment_reads = [r for t in trace.tokens if t.segment == 0 for r in t.reads]
    assert first_segment_reads == []
    for tok in trace.tokens:
        for read in tok.reads:
            assert read.source_segment < tok.segment
            assert read.source_position < tok.segment * 10
    assert trace.segments[-1].slots_used <= trace.segments[-1].capacity


def test_html_is_self_contained_and_escaped(tmp_path: Path) -> None:
    model = _model()
    ids = torch.randint(3, VOCAB, (1, 12))
    texts = ["<b>"] + ["</script>"] + [f"w{i}" for i in range(10)]
    path = build_report(
        model, ids, texts, tmp_path / "trace.html", segment_len=6, title="demo & test"
    )
    page = path.read_text()
    assert page.startswith("<!doctype html>")
    assert "&lt;b&gt;" in page
    assert "demo &amp; test" in page
    assert "<\\/script>" in page  # escaped inside the JSON payload
    assert "http://" not in page
    assert "https://" not in page


def test_trace_validates_inputs() -> None:
    model = _model()
    with pytest.raises(ValueError, match="single document"):
        collect_trace(model, torch.zeros(2, 4, dtype=torch.long), ["a"] * 4, segment_len=2)
    with pytest.raises(ValueError, match="one entry per token"):
        collect_trace(model, torch.zeros(1, 4, dtype=torch.long), ["a"], segment_len=2)


def test_memory_disabled_trace_has_no_priorities() -> None:
    model = _model(memory_enabled=False)
    trace = collect_trace(model, torch.randint(3, VOCAB, (1, 8)), ["x"] * 8, segment_len=4)
    assert all(t.priority is None and not t.reads for t in trace.tokens)
    assert "Select a token" in render_html(trace)
