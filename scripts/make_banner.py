"""Write the README header banner as light and dark SVGs in docs/assets/.

    uv run python scripts/make_banner.py

The motif is a memory bank: slots tinted by the segment that wrote them, with the
"kept" study facts (blue) surviving to the right while filler fades out.
"""

from __future__ import annotations

import random
from pathlib import Path

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "empty": "#ecebe6",
        "series": ("#2a78d6", "#eb6834", "#1baf7a"),
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "empty": "#262624",
        "series": ("#3987e5", "#d95926", "#199e70"),
    },
}


def banner(mode: str, seed: int = 4) -> str:
    t = THEMES[mode]
    rng = random.Random(seed)
    width, height = 1200, 260
    cells = []
    cols, rows, size, gap = 17, 4, 26, 6
    x0, y0 = 604, 66
    for c in range(cols):
        progress = c / (cols - 1)
        for r in range(rows):
            roll = rng.random()
            # Left: a mix of every segment. Right: filler drops away, study facts remain.
            if roll < 0.35 + 0.45 * progress:
                color, opacity = t["series"][0], 1.0
            elif roll < 0.75:
                color, opacity = t["series"][1 + (r + c) % 2], max(0.15, 1.0 - progress * 1.1)
            else:
                color, opacity = t["empty"], 1.0
            x = x0 + c * (size + gap)
            y = y0 + r * (size + gap) + (rng.random() - 0.5) * 2
            cells.append(
                f'<rect x="{x:.0f}" y="{y:.0f}" width="{size}" height="{size}" rx="5" '
                f'fill="{color}" fill-opacity="{opacity:.2f}"/>'
            )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="EngramLM: causal, salience-gated memory for language models">
  <rect width="{width}" height="{height}" rx="18" fill="{t["surface"]}"/>
  <text x="56" y="118" font-family="system-ui, -apple-system, 'Segoe UI', sans-serif" font-size="68" font-weight="700" fill="{t["ink"]}">EngramLM</text>
  <text x="58" y="162" font-family="system-ui, -apple-system, 'Segoe UI', sans-serif" font-size="22" fill="{t["ink2"]}">Causal, salience-gated memory</text>
  <text x="58" y="192" font-family="system-ui, -apple-system, 'Segoe UI', sans-serif" font-size="22" fill="{t["ink2"]}">for Hugging Face language models</text>
  {"".join(cells)}
</svg>
"""


def main() -> None:
    out = Path("docs/assets")
    out.mkdir(parents=True, exist_ok=True)
    for mode in THEMES:
        (out / f"banner-{mode}.svg").write_text(banner(mode))
        print(out / f"banner-{mode}.svg")


if __name__ == "__main__":
    main()
