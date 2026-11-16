#!/usr/bin/env python3
"""Render the reader-reform comparison figure as deterministic SVG.

The figure visualises the reader reform (RESULTS.md section 12): per-arm
token-F1 and answer containment on the restricted 21-answerable set, with the
2026-09 BART-large reader against the 2026-11 instruction-tuned flan-t5-small
reader. Same arms, same retrieval, same prompts; only the reader changed.

SVG is written by hand -- no matplotlib -- so the output is byte-for-byte
reproducible in any environment with the standard library. Regenerate with:

    python tools/make_reader_reform_figure.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

from make_restricted_figure import ARM_COLOURS, ARM_ORDER, PANELS, bar  # noqa: E402

WIDTH, HEIGHT = 720, 390
PANEL_W, PANEL_H = 300, 220
PANEL_X = (60, 400)
PANEL_Y = 60
BAR_W = 28
READERS = ("bart", "flan")
READER_LABELS = {"bart": "BART-large reader (2026-09)",
                 "flan": "flan-t5-small reader (2026-11)"}


def load_values(bart_path: str, flan_path: str) -> dict:
    """Load the two committed restricted-set artefacts.

    Args:
        bart_path: ``results/restricted_answerable.json`` (2026-09 reader,
            restricted re-aggregation).
        flan_path: ``results/instruct_reader_ablation.json`` (2026-11 run).

    Returns:
        Mapping ``{"bart": {arm: {...}}, "flan": {arm: {...}}}``.
    """
    with open(bart_path) as handle:
        bart = json.load(handle)
    with open(flan_path) as handle:
        flan = json.load(handle)
    return {"bart": bart["arms_restricted"], "flan": flan["restricted"]}


def render(values: dict, y_max: float = 1.0) -> str:
    """Render the complete SVG document.

    Args:
        values: output of :func:`load_values`.
        y_max: shared axis maximum for both panels.

    Returns:
        The SVG document as a string.
    """
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{HEIGHT}" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="12">',
        "<defs>",
    ]
    for colour in list(dict.fromkeys(ARM_COLOURS.values())) + ["#888888"]:
        parts.append(
            f'<pattern id="hatch-{colour.strip("#")}" width="6" height="6" '
            f'patternTransform="rotate(45)" patternUnits="userSpaceOnUse">'
            f'<rect width="6" height="6" fill="white"/>'
            f'<line x1="0" y1="0" x2="0" y2="6" stroke="{colour}" '
            f'stroke-width="3"/></pattern>'
        )
    parts.append("</defs>")
    parts.append(
        f'<text x="{WIDTH / 2}" y="24" text-anchor="middle" font-size="14" '
        f'font-weight="bold">Reader reform on the answerable subset (n = 21): '
        f"BART-large (2026-09) vs flan-t5-small (2026-11)</text>"
    )

    for (metric, title), panel_x in zip(PANELS, PANEL_X):
        parts.append(
            f'<text x="{panel_x + PANEL_W / 2:.0f}" y="{PANEL_Y - 12}" '
            f'text-anchor="middle" font-weight="bold">{title}</text>'
        )
        parts.append(
            f'<line x1="{panel_x}" y1="{PANEL_Y + PANEL_H}" '
            f'x2="{panel_x + PANEL_W}" y2="{PANEL_Y + PANEL_H}" '
            f'stroke="black"/>'
            f'<line x1="{panel_x}" y1="{PANEL_Y}" x2="{panel_x}" '
            f'y2="{PANEL_Y + PANEL_H}" stroke="black"/>'
        )
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            ty = PANEL_Y + PANEL_H - PANEL_H * (tick / y_max)
            parts.append(
                f'<line x1="{panel_x - 4}" y1="{ty:.1f}" x2="{panel_x}" '
                f'y2="{ty:.1f}" stroke="black"/>'
                f'<text x="{panel_x - 8}" y="{ty + 4:.1f}" text-anchor="end" '
                f'font-size="10">{tick:.2f}</text>'
            )
        for group, arm in enumerate(ARM_ORDER):
            gx = panel_x + 30 + group * (2 * BAR_W + 42)
            for offset, reader in ((0, "bart"), (BAR_W + 4, "flan")):
                value = values[reader][arm]["F1" if metric == "F1"
                                            else "containment"]
                parts.append(bar(gx + offset, value, ARM_COLOURS[arm],
                                 reader == "flan", y_max))
                height = PANEL_H * (value / y_max)
                parts.append(
                    f'<text x="{gx + offset + BAR_W / 2:.1f}" '
                    f'y="{PANEL_Y + PANEL_H - height - 4:.1f}" '
                    f'text-anchor="middle" font-size="9">{value:.3f}</text>'
                )
            parts.append(
                f'<text x="{gx + BAR_W:.0f}" y="{PANEL_Y + PANEL_H + 16}" '
                f'text-anchor="middle" font-size="11">{arm}</text>'
            )

    legend_y = PANEL_Y + PANEL_H + 40
    parts.append(
        f'<rect x="{PANEL_X[0]}" y="{legend_y}" width="14" height="14" '
        f'fill="#888888"/><text x="{PANEL_X[0] + 20}" y="{legend_y + 12}">'
        f"{READER_LABELS['bart']}</text>"
        f'<rect x="{PANEL_X[0] + 230}" y="{legend_y}" width="14" height="14" '
        f'fill="url(#hatch-888888)" stroke="#888888"/>'
        f'<text x="{PANEL_X[0] + 250}" y="{legend_y + 12}">'
        f"{READER_LABELS['flan']}</text>"
    )
    parts.append(
        f'<text x="{WIDTH / 2}" y="{HEIGHT - 8}" text-anchor="middle" '
        f'font-size="10" fill="#555">Same arms, retrieval, prompts and '
        f"metrics; only the reader changed. Sources: "
        f"results/restricted_answerable.json, "
        f"results/instruct_reader_ablation.json.</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    """Entry point: load both artefacts and write the SVG figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bart", default=os.path.join(
        REPO_ROOT, "results", "restricted_answerable.json"))
    parser.add_argument("--flan", default=os.path.join(
        REPO_ROOT, "results", "instruct_reader_ablation.json"))
    parser.add_argument("--output", default=os.path.join(
        REPO_ROOT, "docs", "figures", "reader_reform_comparison.svg"))
    args = parser.parse_args()

    svg = render(load_values(args.bart, args.flan))
    with open(args.output, "w") as handle:
        handle.write(svg)
    print(f"[figure] wrote {args.output}")


if __name__ == "__main__":
    main()
