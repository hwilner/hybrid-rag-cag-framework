#!/usr/bin/env python3
"""Render the restricted-vs-full comparison figure as deterministic SVG.

The figure visualises ``results/restricted_answerable.json`` (restricted set,
n = 21) against ``results/ablation/ablation_main.json`` (full set, n = 55):
per-arm token-F1 and answer containment, side by side.

SVG is written by hand -- no matplotlib -- so the output is byte-for-byte
reproducible in any environment with the standard library, and committable
as text. Regenerate with:

    python tools/make_restricted_figure.py
"""

from __future__ import annotations

import argparse
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARM_ORDER = ("RAG", "CAG", "Hybrid")
ARM_COLOURS = {"RAG": "#4C78A8", "CAG": "#F58518", "Hybrid": "#54A24B"}
PANELS = (("F1", "token-F1"), ("containment", "answer containment"))

WIDTH, HEIGHT = 720, 390
PANEL_W, PANEL_H = 300, 220
PANEL_X = (60, 400)
PANEL_Y = 60
BAR_W, GROUP_GAP = 28, 14
FULL_HATCH = "none"  # full-set bars are solid; restricted bars are hatched


def load_values(ablation_path: str, restricted_path: str) -> dict:
    """Load full-set and restricted-set arm aggregates.

    Args:
        ablation_path: committed three-arm ablation artefact (n = 55).
        restricted_path: committed restricted re-analysis artefact (n = 21).

    Returns:
        Mapping ``{"full": {arm: {...}}, "restricted": {arm: {...}}}``.
    """
    with open(ablation_path) as handle:
        ablation = json.load(handle)
    with open(restricted_path) as handle:
        restricted = json.load(handle)
    return {
        "full": {arm: ablation["arms"][arm] for arm in ARM_ORDER},
        "restricted": restricted["arms_restricted"],
    }


def bar(x: float, value: float, colour: str, hatched: bool, y_max: float) -> str:
    """Return the SVG for one bar.

    Args:
        x: left edge of the bar, in pixels.
        value: bar value in [0, 1]; NaN/None are rejected upstream.
        colour: fill colour.
        hatched: whether to use the hatch pattern (restricted set).
        y_max: axis maximum (1.0).

    Returns:
        An SVG ``<rect>`` string.
    """
    height = PANEL_H * (value / y_max)
    y = PANEL_Y + PANEL_H - height
    fill = f"url(#hatch-{colour.strip('#')})" if hatched else colour
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{BAR_W}" '
            f'height="{height:.1f}" fill="{fill}" stroke="{colour}" '
            f'stroke-width="1"/>')


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
    # The legend swatch for the hatched (restricted) bars uses neutral gray,
    # which is not an arm colour, so define its pattern explicitly.
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
        f'font-weight="bold">Full benchmark (n = 55) vs answerable subset '
        f"(n = 21), per arm</text>"
    )

    for (metric, title), panel_x in zip(PANELS, PANEL_X):
        parts.append(
            f'<text x="{panel_x + PANEL_W / 2:.0f}" y="{PANEL_Y - 12}" '
            f'text-anchor="middle" font-weight="bold">{title}</text>'
        )
        # axes
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
            gx = panel_x + 30 + group * (2 * BAR_W + GROUP_GAP + 28)
            parts.append(bar(gx, values["full"][arm][metric],
                             ARM_COLOURS[arm], False, y_max))
            parts.append(bar(gx + BAR_W + 4, values["restricted"][arm][metric],
                             ARM_COLOURS[arm], True, y_max))
            parts.append(
                f'<text x="{gx + BAR_W:.0f}" y="{PANEL_Y + PANEL_H + 16}" '
                f'text-anchor="middle" font-size="11">{arm}</text>'
            )
            for offset, src in ((0, "full"), (BAR_W + 4, "restricted")):
                value = values[src][arm][metric]
                height = PANEL_H * (value / y_max)
                parts.append(
                    f'<text x="{gx + offset + BAR_W / 2:.1f}" '
                    f'y="{PANEL_Y + PANEL_H - height - 4:.1f}" '
                    f'text-anchor="middle" font-size="9">{value:.3f}</text>'
                )

    legend_y = PANEL_Y + PANEL_H + 40
    parts.append(
        f'<rect x="{PANEL_X[0]}" y="{legend_y}" width="14" height="14" '
        f'fill="#888888"/><text x="{PANEL_X[0] + 20}" y="{legend_y + 12}">'
        f"full set (n = 55)</text>"
        f'<rect x="{PANEL_X[0] + 150}" y="{legend_y}" width="14" height="14" '
        f'fill="url(#hatch-888888)" stroke="#888888"/>'
        f'<text x="{PANEL_X[0] + 170}" y="{legend_y + 12}">answerable subset '
        f"(n = 21)</text>"
    )
    parts.append(
        f'<text x="{WIDTH / 2}" y="{HEIGHT - 8}" text-anchor="middle" '
        f'font-size="10" fill="#555">Re-aggregation of the committed '
        f"2026-09 per-question measurements; no new model runs. "
        f"Source: results/restricted_answerable.json.</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    """Entry point: load both artefacts and write the SVG figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ablation",
                        default=os.path.join(REPO_ROOT, "results", "ablation",
                                             "ablation_main.json"))
    parser.add_argument("--restricted",
                        default=os.path.join(REPO_ROOT, "results",
                                             "restricted_answerable.json"))
    parser.add_argument("--output",
                        default=os.path.join(REPO_ROOT, "docs", "figures",
                                             "restricted_answerable_comparison.svg"))
    args = parser.parse_args()

    values = load_values(args.ablation, args.restricted)
    svg = render(values)
    with open(args.output, "w") as handle:
        handle.write(svg)
    print(f"[figure] wrote {args.output}")


if __name__ == "__main__":
    main()
