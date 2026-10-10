"""Memory inspector: trace what a model wrote and read, and render it as HTML.

The report is a single self-contained HTML file (no external scripts or fonts):

- every token is shaded by the salience priority it received;
- tokens that entered memory are outlined;
- clicking a token lists the memory slots it read, with the source token, its
  position and segment, and the attention weight.

Because reads can only target earlier segments, the trace doubles as a visible
causality check: every listed source sits before the reading token's segment.
"""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from engramlm.model import EngramLM
from engramlm.segment import segment_bounds


@dataclass
class Read:
    source_position: int
    source_segment: int
    weight: float


@dataclass
class TokenTrace:
    position: int
    segment: int
    text: str
    priority: float | None
    written: bool
    reads: list[Read] = field(default_factory=list)


@dataclass
class SegmentStats:
    segment: int
    slots_used: int
    capacity: int
    mean_age: float
    merges: int


@dataclass
class Trace:
    tokens: list[TokenTrace]
    segments: list[SegmentStats]


@torch.no_grad()
def collect_trace(
    model: EngramLM,
    input_ids: Tensor,
    token_texts: list[str],
    *,
    segment_len: int,
    top_reads: int = 5,
) -> Trace:
    """Run one document (``input_ids`` of shape ``(1, L)``) and record every read/write."""
    if input_ids.shape[0] != 1:
        raise ValueError("collect_trace expects a single document (batch size 1)")
    if len(token_texts) != input_ids.shape[1]:
        raise ValueError("token_texts must have one entry per token")
    model.eval()
    memory = model.new_memory(1, input_ids.device)
    tokens: list[TokenTrace] = []
    segments: list[SegmentStats] = []

    for index, (start, end) in enumerate(segment_bounds(input_ids.shape[1], segment_len)):
        out = model(
            input_ids[:, start:end], memory=memory, return_reads=True, position_offset=start
        )
        for t in range(end - start):
            record = TokenTrace(
                position=start + t,
                segment=index,
                text=token_texts[start + t],
                priority=float(out.priority[0, t]) if out.priority is not None else None,
                written=bool(out.written[0, t]) if out.written is not None else False,
            )
            if out.reads is not None:
                slots = out.reads.slots[0, t]
                weights = out.reads.weights[0, t]
                order = weights.argsort(descending=True)[:top_reads]
                for j in order.tolist():
                    weight = float(weights[j])
                    if weight <= 0.0:
                        continue
                    slot = int(slots[j])
                    record.reads.append(
                        Read(
                            source_position=int(memory.origin_position[0, slot]),
                            source_segment=int(memory.origin_segment[0, slot]),
                            weight=weight,
                        )
                    )
            tokens.append(record)
        memory = out.memory
        valid = memory.valid[0]
        segments.append(
            SegmentStats(
                segment=index,
                slots_used=int(valid.sum()),
                capacity=memory.capacity,
                mean_age=float(memory.age[0, valid].mean()) if bool(valid.any()) else 0.0,
                merges=int(out.merges[0]) if out.merges is not None else 0,
            )
        )
    return Trace(tokens=tokens, segments=segments)


_CSS = """
:root { --bg:#ffffff; --fg:#1f2328; --muted:#656d76; --line:#d0d7de; --hot:255,140,0; --panel:#f6f8fa; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0d1117; --fg:#e6edf3; --muted:#8d96a0; --line:#30363d; --hot:255,166,87; --panel:#161b22; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.6 system-ui, sans-serif; }
header { padding:16px 20px; border-bottom:1px solid var(--line); }
h1 { font-size:18px; margin:0 0 4px; }
.sub { color:var(--muted); font-size:13px; }
main { display:grid; grid-template-columns:minmax(0,1fr) 340px; gap:0; }
@media (max-width: 800px) { main { grid-template-columns:1fr; } }
#text { padding:16px 20px; font-family:ui-monospace, monospace; white-space:pre-wrap; }
.seg { border-top:1px dashed var(--line); margin:10px 0 4px; color:var(--muted); font:12px system-ui; }
.tok { cursor:pointer; border-radius:3px; padding:0 1px; outline-offset:-1px; }
.tok.w { outline:1.5px solid rgb(var(--hot)); }
.tok.sel { box-shadow:0 0 0 2px var(--fg); }
.tok.src { text-decoration:underline 2px; }
aside { border-left:1px solid var(--line); background:var(--panel); padding:16px; }
.read { margin:8px 0; font-size:13px; }
.bar { height:6px; background:rgb(var(--hot)); border-radius:3px; }
table { border-collapse:collapse; font-size:12px; margin-top:12px; }
td, th { border:1px solid var(--line); padding:2px 6px; text-align:right; }
"""

_JS = """
const data = JSON.parse(document.getElementById('trace').textContent);
const spans = Array.from(document.querySelectorAll('.tok'));
const panel = document.getElementById('panel');
function show(i) {
  spans.forEach(s => s.classList.remove('sel', 'src'));
  spans[i].classList.add('sel');
  const t = data.tokens[i];
  let out = `<b>token ${t.position}</b> (segment ${t.segment})<br>`;
  out += t.priority === null ? '' : `priority ${t.priority.toFixed(3)}${t.written ? ', written' : ''}<br>`;
  if (!t.reads.length) { out += '<p class="sub">No memory reads (memory empty or reads off).</p>'; }
  t.reads.forEach(r => {
    const src = data.tokens[r.source_position];
    if (spans[r.source_position]) spans[r.source_position].classList.add('src');
    const label = src ? src.text.replace(/</g, '&lt;') : '?';
    out += `<div class="read">pos ${r.source_position} (seg ${r.source_segment}) <code>${label}</code>` +
           ` ${(r.weight * 100).toFixed(1)}%<div class="bar" style="width:${r.weight * 100}%"></div></div>`;
  });
  panel.innerHTML = out;
}
spans.forEach((s, i) => s.addEventListener('click', () => show(i)));
"""


def render_html(trace: Trace, *, title: str = "EngramLM memory trace") -> str:
    parts: list[str] = []
    current = -1
    for i, tok in enumerate(trace.tokens):
        if tok.segment != current:
            current = tok.segment
            parts.append(f'<div class="seg">segment {current}</div>')
        alpha = 0.0 if tok.priority is None else max(0.0, min(1.0, tok.priority))
        style = f"background:rgba(var(--hot),{alpha * 0.55:.3f})"
        cls = "tok w" if tok.written else "tok"
        parts.append(
            f'<span class="{cls}" style="{style}" data-i="{i}">{html.escape(tok.text)}</span>'
        )

    rows = "".join(
        f"<tr><td>{s.segment}</td><td>{s.slots_used}/{s.capacity}</td>"
        f"<td>{s.mean_age:.2f}</td><td>{s.merges}</td></tr>"
        for s in trace.segments
    )
    payload = json.dumps({"tokens": [asdict(t) for t in trace.tokens]}).replace("</", "<\\/")
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>"
        f"<header><h1>{html.escape(title)}</h1><div class='sub'>Shading = write priority; "
        "outline = written to memory; click a token to see what it read.</div></header>"
        f"<main><div id='text'>{''.join(parts)}</div><aside><div id='panel' class='sub'>"
        "Select a token.</div><table><tr><th>seg</th><th>slots</th><th>mean age</th>"
        f"<th>merges</th></tr>{rows}</table></aside></main>"
        f"<script id='trace' type='application/json'>{payload}</script>"
        f"<script>{_JS}</script></body></html>"
    )


def build_report(
    model: EngramLM,
    input_ids: Tensor,
    token_texts: list[str],
    out_path: str | Path,
    *,
    segment_len: int,
    title: str = "EngramLM memory trace",
) -> Path:
    trace = collect_trace(model, input_ids, token_texts, segment_len=segment_len)
    path = Path(out_path)
    path.write_text(render_html(trace, title=title), encoding="utf-8")
    return path
