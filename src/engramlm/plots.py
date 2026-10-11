"""Figures and animations for EngramLM results (needs the ``viz`` extra).

Every figure is rendered in a light and a dark theme so a README can switch with
``<picture>``. Colours come from a fixed, colour-vision-checked palette:

- memory variants always use the same categorical slot in every chart;
- capacity (an ordered quantity) uses two steps of one blue ramp;
- salience (a magnitude) uses a single-hue sequential ramp.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle

from engramlm.eval.recall import QUERY, RecallTask
from engramlm.eval.stats import bootstrap_ci
from engramlm.model import EngramLM


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    empty: str
    series: tuple[str, ...]
    ordinal: tuple[str, str]
    sequential: tuple[str, ...]


THEMES = {
    "light": Theme(
        name="light",
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink2="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        empty="#ecebe6",
        series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"),
        ordinal=("#86b6ef", "#1c5cab"),
        sequential=("#eef4fc", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"),
    ),
    "dark": Theme(
        name="dark",
        surface="#1a1a19",
        ink="#ffffff",
        ink2="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        empty="#262624",
        series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"),
        ordinal=("#256abf", "#9ec5f4"),
        sequential=("#202a36", "#184f95", "#256abf", "#3987e5", "#86b6ef", "#cde2fb"),
    ),
}

VARIANT_ORDER = ("recency", "random", "surprise", "learned", "learned+sleep")
VARIANT_LABELS = {
    "no-memory": "no memory",
    "recency": "recency (FIFO)",
    "random": "random",
    "surprise": "surprise",
    "learned": "learned",
    "learned+sleep": "learned + sleep",
}
CSS_PX = 96.0
DPI = 200


def variant_color(theme: Theme, variant: str) -> str:
    """The same variant gets the same colour in every chart."""
    return theme.series[VARIANT_ORDER.index(variant)]


def use_theme(mode: str) -> Theme:
    theme = THEMES[mode]
    plt.rcParams.update(
        {
            "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9.5,
            "figure.facecolor": theme.surface,
            "axes.facecolor": theme.surface,
            "savefig.facecolor": theme.surface,
            "text.color": theme.ink,
            "axes.labelcolor": theme.ink2,
            "axes.edgecolor": theme.axis,
            "xtick.color": theme.muted,
            "ytick.color": theme.muted,
            "xtick.labelcolor": theme.ink2,
            "ytick.labelcolor": theme.ink2,
            "grid.color": theme.grid,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )
    return theme


def _style_axes(ax: Axes, theme: Theme, grid_axis: Literal["x", "y"] = "y") -> None:
    ax.grid(True, axis=grid_axis)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)
    ax.spines["bottom"].set_color(theme.axis)
    if grid_axis == "x":
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color(theme.axis)


def _title(fig: Figure, title: str, subtitle: str, theme: Theme) -> None:
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=12.5, weight="bold")
    fig.text(0.012, 0.905, subtitle, ha="left", va="top", fontsize=9, color=theme.ink2)


def _px_per_unit(ax: Axes) -> tuple[float, float]:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bbox = ax.get_window_extent()
    return bbox.width / abs(x1 - x0), bbox.height / abs(y1 - y0)


def _rounded_bar(
    ax: Axes, start: float, thickness: float, length: float, color: str, *, horizontal: bool = False
) -> None:
    """Bar with a 4px rounded data end and a square baseline.

    ``start``/``thickness`` run across the bar, ``length`` along it. Axis limits must
    already be final, because the corner radius is converted from pixels.
    """
    if length <= 0:
        return
    ppx, ppy = _px_per_unit(ax)
    radius_px = 4.0 * ax.figure.dpi / CSS_PX
    aspect = ppx / ppy  # y-units per x-unit for an equal on-screen distance
    if horizontal:
        rounding = min(radius_px / ppx, length / 2, thickness / 2 / aspect)
        xy, width, height = (0.0, start), length, thickness
        cover = Rectangle((0.0, start), max(length - rounding, 0.0), thickness)
    else:
        rounding = min(radius_px / ppx, thickness / 2, length / 2 / aspect)
        xy, width, height = (start, 0.0), thickness, length
        cover = Rectangle((start, 0.0), thickness, max(length - rounding * aspect, 0.0))
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            mutation_aspect=aspect,
            facecolor=color,
            edgecolor="none",
            linewidth=0,
        )
    )
    cover.set_facecolor(color)
    cover.set_linewidth(0)
    ax.add_patch(cover)


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- E1


def plot_recall_accuracy(
    rows: Sequence[dict[str, Any]], out: str | Path, *, mode: str = "light", chance: float
) -> Path:
    theme = use_theme(mode)
    by_key: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        by_key[(row["variant"], int(row["capacity"]))].append(float(row["accuracy"]))
    variants = [v for v in VARIANT_ORDER if any(k[0] == v for k in by_key)]
    capacities = sorted({k[1] for k in by_key if k[0] != "no-memory"})
    baseline = by_key.get(("no-memory", 0))

    fig = plt.figure(figsize=(7.4, 3.9))
    ax = fig.add_axes((0.075, 0.2, 0.9, 0.58))
    _title(
        fig,
        "Cross-segment associative recall (E1)",
        "Accuracy on test-segment values; mean of seeds with 95% bootstrap interval",
        theme,
    )
    stats = {k: bootstrap_ci(v, n_resamples=4000) for k, v in by_key.items()}
    top = max(s.high for s in stats.values())
    ymax = max(0.1, math.ceil(top * 10.5) / 10)
    ax.set_xlim(-0.6, len(variants) - 0.4)
    ax.set_ylim(0, ymax)
    _style_axes(ax, theme)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")

    ppx, _ = _px_per_unit(ax)
    width = min(0.34, (22 * fig.dpi / CSS_PX) / ppx)
    gap = (2 * fig.dpi / CSS_PX) / ppx
    offsets = [((i - (len(capacities) - 1) / 2) * (width + gap)) for i in range(len(capacities))]
    for ci, cap in enumerate(capacities):
        color = theme.ordinal[ci % 2]
        for vi, variant in enumerate(variants):
            s = stats.get((variant, cap))
            if s is None:
                continue
            x = vi + offsets[ci] - width / 2
            _rounded_bar(ax, x, width, s.mean, color)
            center = vi + offsets[ci]
            ax.vlines(center, s.low, s.high, color=theme.ink2, linewidth=1.1, zorder=3)
            ax.hlines(
                [s.low, s.high],
                center - width * 0.18,
                center + width * 0.18,
                color=theme.ink2,
                linewidth=1.1,
                zorder=3,
            )
            ax.text(
                center,
                s.high + ymax * 0.02,
                f"{s.mean * 100:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8.2,
                color=theme.ink2,
            )
    if baseline:
        level = sum(baseline) / len(baseline)
        same = abs(level - chance) < 0.01
        ax.axhline(level, color=theme.muted, linewidth=1, zorder=2)
        label = (
            f"no memory ≈ chance ({chance * 100:.1f}%)"
            if same
            else f"no memory ({level * 100:.1f}%)"
        )
        ax.text(
            len(variants) - 0.42,
            level + ymax * 0.015,
            label,
            ha="right",
            va="bottom",
            fontsize=8.2,
            color=theme.ink2,
        )
        if not same:
            ax.axhline(chance, color=theme.grid, linewidth=1)
    ax.set_xticks(range(len(variants)), [VARIANT_LABELS[v] for v in variants])
    handles = [Rectangle((0, 0), 1, 1, color=theme.ordinal[i % 2]) for i in range(len(capacities))]
    ax.legend(
        handles,
        [f"capacity {c}" for c in capacities],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.13),
        ncol=len(capacities),
        handlelength=1,
        handleheight=1,
        columnspacing=1.4,
    )
    fig.text(
        0.012,
        0.03,
        "Capacity = memory slots per stream. Lower capacity means more "
        "competition from distractor segments.",
        fontsize=8,
        color=theme.muted,
    )
    return _save(fig, Path(out))


_LOG_LINE = re.compile(
    r"\[(?P<variant>[\w+-]+) cap=(?P<cap>\d+) seed=(?P<seed>\d+)\] "
    r"step (?P<step>\d+) loss (?P<loss>[\d.]+)"
)


def parse_training_log(text: str) -> dict[tuple[str, int], dict[int, list[float]]]:
    """``(variant, capacity) -> step -> [loss per seed]`` from a recall run log."""
    curves: dict[tuple[str, int], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for match in _LOG_LINE.finditer(text):
        cap = int(match["cap"]) if match["variant"] != "no-memory" else 0
        curves[(match["variant"], cap)][int(match["step"])].append(float(match["loss"]))
    return curves


def plot_training_curves(
    log_text: str, out: str | Path, *, mode: str = "light", capacity: int, chance_nll: float
) -> Path:
    theme = use_theme(mode)
    curves = parse_training_log(log_text)
    fig = plt.figure(figsize=(7.4, 3.6))
    ax = fig.add_axes((0.08, 0.13, 0.7, 0.66))
    _title(
        fig,
        f"Training loss on recall targets (capacity {capacity})",
        "Mean over seeds; nats per test-segment value token",
        theme,
    )
    _style_axes(ax, theme)
    series: list[tuple[str, str]] = [("no-memory", theme.muted)]
    series += [(v, variant_color(theme, v)) for v in VARIANT_ORDER]
    end_points = []
    for variant, color in series:
        cap = 0 if variant == "no-memory" else capacity
        points = curves.get((variant, cap))
        if not points:
            continue
        steps = sorted(points)
        losses = [sum(points[s]) / len(points[s]) for s in steps]
        ax.plot(
            steps,
            losses,
            color=color,
            linewidth=2,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=VARIANT_LABELS[variant],
            zorder=3,
        )
        ax.plot(
            steps[-1],
            losses[-1],
            "o",
            markersize=6,
            color=color,
            markeredgecolor=theme.surface,
            markeredgewidth=2,
            zorder=4,
        )
        end_points.append((losses[-1], variant))
    ax.axhline(chance_nll, color=theme.grid, linewidth=1, zorder=1)
    ax.text(ax.get_xlim()[0], chance_nll, " chance", va="bottom", fontsize=8, color=theme.muted)
    ax.set_xlabel("training step")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), handlelength=1.6)
    return _save(fig, Path(out))


# --------------------------------------------------------------------- causality


def plot_causality(
    entries: Sequence[tuple[str, float]], out: str | Path, *, mode: str = "light"
) -> Path:
    theme = use_theme(mode)
    fig = plt.figure(figsize=(7.4, 2.9))
    ax = fig.add_axes((0.36, 0.2, 0.6, 0.56))
    _title(
        fig,
        "Does changing a future token move earlier predictions?",
        "Largest change in earlier logits when every token after a random cut changes "
        "(a causal model must show 0)",
        theme,
    )
    largest = max(v for _, v in entries)
    scale = 10.0 ** math.floor(math.log10(largest)) if largest > 0 else 1.0
    xmax = math.ceil(largest / scale * 1.3) * scale if largest > 0 else 1.0
    ax.set_xlim(0, xmax)
    ax.set_ylim(len(entries) - 0.5, -0.5)
    _style_axes(ax, theme, grid_axis="x")
    _, ppy = _px_per_unit(ax)
    height = min(0.5, (20 * fig.dpi / CSS_PX) / ppy)
    for i, (_, value) in enumerate(entries):
        if value > 0:
            _rounded_bar(ax, i - height / 2, height, value, theme.series[1], horizontal=True)
            ax.text(
                value + xmax * 0.012,
                i,
                f"{value:.3f}" if value >= 1e-3 else f"{value:.1e}",
                va="center",
                fontsize=8.5,
                color=theme.ink2,
            )
        else:
            ax.plot(
                0,
                i,
                "o",
                markersize=7,
                color=theme.series[0],
                markeredgecolor=theme.surface,
                markeredgewidth=2,
                clip_on=False,
                zorder=4,
            )
            ax.text(xmax * 0.025, i, "0 (exact)", va="center", fontsize=8.5, color=theme.ink2)
    ax.set_yticks(range(len(entries)), [label for label, _ in entries])
    ax.set_xlabel("max |Δ logits| at or before the cut")
    return _save(fig, Path(out))


# --------------------------------------------------------------------------- E2


def plot_lm_results(
    rows: Sequence[dict[str, Any]], out: str | Path, *, mode: str = "light"
) -> Path:
    theme = use_theme(mode)
    by_variant: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(float(row["bpb"]))
    order = [
        v
        for v in ("base", "lora", "lora+recency", "lora+surprise", "lora+learned")
        if v in by_variant
    ]
    values = {v: sum(by_variant[v]) / len(by_variant[v]) for v in order}
    reference = by_variant.get("full-context")

    fig = plt.figure(figsize=(7.4, 3.0))
    ax = fig.add_axes((0.2, 0.2, 0.74, 0.56))
    _title(
        fig,
        "Long-document language modelling (E2)",
        "Bits per byte on held-out WikiText-2 articles (lower is better)",
        theme,
    )
    all_vals = list(values.values()) + (reference or [])
    lo, hi = min(all_vals), max(all_vals)
    pad = (hi - lo) * 0.25 or 0.02
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(len(order) - 0.5, -0.8)
    _style_axes(ax, theme, grid_axis="x")
    colors = {"base": theme.muted, "lora": theme.ink2}
    for i, variant in enumerate(order):
        name = variant.removeprefix("lora+")
        color = colors.get(variant) or variant_color(theme, name)
        ax.plot(
            values[variant],
            i,
            "o",
            markersize=8,
            color=color,
            markeredgecolor=theme.surface,
            markeredgewidth=2,
            zorder=4,
        )
        ax.text(
            values[variant],
            i - 0.32,
            f"{values[variant]:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color=theme.ink2,
        )
    if reference:
        level = sum(reference) / len(reference)
        ax.axvline(level, color=theme.axis, linewidth=1, zorder=1)
        ax.text(
            level, -0.72, "full context ", ha="right", va="center", fontsize=8, color=theme.muted
        )
    pretty = {
        "base": "base (no training)",
        "lora": "LoRA only",
        "lora+recency": "LoRA + recency",
        "lora+surprise": "LoRA + surprise",
        "lora+learned": "LoRA + learned",
    }
    ax.set_yticks(range(len(order)), [pretty[v] for v in order])
    ax.set_xlabel("bits per byte")
    return _save(fig, Path(out))


# -------------------------------------------------------------------- animation


@dataclass
class SegmentFrame:
    """One segment of a recall document as seen by one model."""

    index: int
    kind: str  # "study", "gap", "test"
    token_kind: list[str]  # per token: key, value, filler, query
    priority: list[float]
    written: list[bool]
    bank_origin: list[int]  # per slot: origin segment, -1 if empty
    bank_is_fact: list[bool]  # per slot: holds a study key/value token
    recalled: list[bool] = field(default_factory=list)  # test segment: per query


def _token_kind(task: RecallTask, token: int) -> str:
    if token == QUERY:
        return "query"
    if task.key_offset <= token < task.value_offset:
        return "key"
    if task.value_offset <= token < task.filler_offset:
        return "value"
    return "filler"


@torch.no_grad()
def recall_timeline(model: EngramLM, task: RecallTask, ids: torch.Tensor) -> list[SegmentFrame]:
    """Run one recall document (``(1, L)``) and record writes, bank contents and recall."""
    model.eval()
    memory = model.new_memory(1, ids.device)
    seg = task.segment_len
    frames: list[SegmentFrame] = []
    for k in range(task.num_segments):
        chunk = ids[:, k * seg : (k + 1) * seg]
        out = model(chunk, memory=memory, position_offset=k * seg)
        memory = out.memory
        kinds = [_token_kind(task, int(t)) for t in chunk[0]]
        origin = [
            int(o) if bool(v) else -1
            for o, v in zip(memory.origin_segment[0], memory.valid[0], strict=True)
        ]
        is_fact = []
        for slot, o in enumerate(origin):
            pos = int(memory.origin_position[0, slot])
            is_fact.append(o == 0 and _token_kind(task, int(ids[0, pos])) in {"key", "value"})
        frame = SegmentFrame(
            index=k,
            kind="study" if k == 0 else ("test" if k == task.num_segments - 1 else "gap"),
            token_kind=kinds,
            priority=[
                float(p)
                for p in (out.priority[0] if out.priority is not None else torch.zeros(seg))
            ],
            written=[
                bool(w)
                for w in (
                    out.written[0]
                    if out.written is not None
                    else torch.zeros(seg, dtype=torch.bool)
                )
            ],
            bank_origin=origin,
            bank_is_fact=is_fact,
        )
        if frame.kind == "test":
            predicted = out.logits[0, :-1].argmax(-1)
            for pos in range(1, seg):
                if kinds[pos] == "value" and kinds[pos - 1] == "key":
                    frame.recalled.append(bool(predicted[pos - 1] == chunk[0, pos]))
        frames.append(frame)
    return frames


def _luminance(color: str) -> float:
    r, g, b = to_rgb(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def animate_memory(
    timelines: dict[str, list[SegmentFrame]],
    out: str | Path,
    *,
    mode: str = "light",
    capacity: int,
) -> Path:
    """Side-by-side GIF of how each model's memory bank evolves over one document."""
    theme = use_theme(mode)
    cmap = LinearSegmentedColormap.from_list("salience", list(theme.sequential))
    origin_colors = [theme.series[0], theme.series[1], theme.series[2], theme.series[6]]
    names = list(timelines)
    n_segments = len(next(iter(timelines.values())))
    seg_len = len(next(iter(timelines.values()))[0].token_kind)
    cols = 8
    rows = math.ceil(capacity / cols)
    # One frame per second: show each segment being read, then hold the write for 2s.
    steps = [(k, phase) for k in range(n_segments) for phase in ("read", "write", "write")]
    steps += [(n_segments - 1, "write")] * 3  # hold the final result

    fig = plt.figure(figsize=(9.6, 4.7))

    def draw(step: int) -> None:
        fig.clear()
        k, phase = steps[step]
        frame0 = timelines[names[0]][k]
        stage = {"study": "study facts", "gap": "distractor filler", "test": "test queries"}[
            frame0.kind
        ]
        fig.text(
            0.012,
            0.975,
            "What stays in memory?",
            ha="left",
            va="top",
            fontsize=13,
            weight="bold",
        )
        fig.text(
            0.012,
            0.912,
            f"Segment {k + 1} of {n_segments}: {stage}. "
            f"Each model writes 16 tokens per segment into a {capacity}-slot memory.",
            ha="left",
            va="top",
            fontsize=9.3,
            color=theme.ink2,
        )
        for p, name in enumerate(names):
            frame = timelines[name][k]
            left = 0.03 + p * 0.5
            ax = fig.add_axes((left, 0.17, 0.44, 0.64))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            ax.text(0, 1.06, name, fontsize=11, weight="bold", va="top", clip_on=False)
            # token strip: salience of this segment's tokens, written ones marked below
            cell = 1 / seg_len
            for t in range(seg_len):
                shade = cmap(frame.priority[t]) if frame.priority else theme.empty
                ax.add_patch(
                    Rectangle(
                        (t * cell + cell * 0.06, 0.78),
                        cell * 0.88,
                        0.09,
                        facecolor=shade,
                        linewidth=0,
                    )
                )
                if frame.token_kind[t] in {"key", "value"}:
                    ax.plot(
                        t * cell + cell / 2, 0.825, "o", markersize=2.6, color=theme.ink, zorder=4
                    )
                if phase == "write" and frame.written[t]:
                    ax.plot(
                        t * cell + cell / 2,
                        0.755,
                        marker=7,
                        markersize=4.5,
                        color=theme.ink2,
                        zorder=4,
                    )
            ax.text(0, 0.9, "this segment (shade = write priority)", fontsize=8, color=theme.muted)
            # memory bank grid
            ax.text(0, 0.66, f"memory bank ({capacity} slots)", fontsize=8, color=theme.muted)
            bank = frame if phase == "write" else (timelines[name][k - 1] if k > 0 else None)
            cw, ch = 1 / cols, 0.52 / rows
            for slot in range(capacity):
                r, c = divmod(slot, cols)
                origin = bank.bank_origin[slot] if bank and slot < len(bank.bank_origin) else -1
                color = origin_colors[min(origin, 3)] if origin >= 0 else theme.empty
                x, y = c * cw, 0.6 - (r + 1) * ch
                ax.add_patch(
                    FancyBboxPatch(
                        (x + cw * 0.04, y + ch * 0.06),
                        cw * 0.92,
                        ch * 0.88,
                        boxstyle="round,pad=0,rounding_size=0.012",
                        facecolor=color,
                        linewidth=0,
                    )
                )
                if bank and origin >= 0 and bank.bank_is_fact[slot]:
                    dot = "#ffffff" if _luminance(color) < 0.55 else theme.ink
                    ax.plot(x + cw / 2, y + ch / 2, "o", markersize=4.2, color=dot, zorder=4)
            facts = sum(bank.bank_is_fact) if bank else 0
            ax.text(
                0,
                -0.02,
                f"study facts still in memory: {facts}",
                fontsize=9.5,
                color=theme.ink,
                va="bottom",
            )
            if frame.kind == "test" and phase == "write" and frame.recalled:
                hits = sum(frame.recalled)
                ax.text(
                    1,
                    -0.02,
                    f"recalled {hits}/{len(frame.recalled)} values",
                    fontsize=9.5,
                    weight="bold",
                    ha="right",
                    va="bottom",
                )
        # legend
        lax = fig.add_axes((0.03, 0.02, 0.94, 0.08))
        lax.axis("off")
        lax.set_xlim(0, 1)
        lax.set_ylim(0, 1)
        items = [
            (origin_colors[0], "written in segment 1 (study)"),
            (origin_colors[1], "segment 2 (filler)"),
            (origin_colors[2], "segment 3 (filler)"),
            (origin_colors[3], "segment 4 (test)"),
            (theme.empty, "empty"),
        ]
        positions = (0.0, 0.22, 0.37, 0.52, 0.66, 0.75)
        for (color, label), x in zip(items, positions, strict=False):
            lax.add_patch(Rectangle((x, 0.3), 0.014, 0.4, facecolor=color, linewidth=0))
            lax.text(x + 0.02, 0.5, label, va="center", fontsize=8.3, color=theme.ink2)
        lax.plot(positions[-1] + 0.007, 0.5, "o", markersize=4, color=theme.ink)
        lax.text(
            positions[-1] + 0.02,
            0.5,
            "study key/value token",
            va="center",
            fontsize=8.3,
            color=theme.ink2,
        )

    anim = FuncAnimation(fig, draw, frames=len(steps), interval=1000)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(
        path,
        writer=PillowWriter(fps=1),
        dpi=100,
        savefig_kwargs={"facecolor": theme.surface},
    )
    plt.close(fig)
    return path


def both_modes(fn: Any, out_stem: str | Path, suffix: str = ".png", **kwargs: Any) -> list[Path]:
    """Render ``fn`` in light and dark themes as ``<stem>-light<suffix>`` / ``-dark``."""
    stem = Path(out_stem)
    return [
        fn(out=stem.with_name(f"{stem.name}-{m}{suffix}"), mode=m, **kwargs)
        for m in ("light", "dark")
    ]


def iter_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    import json

    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(json.loads(Path(path).read_text())["rows"])
    return rows
