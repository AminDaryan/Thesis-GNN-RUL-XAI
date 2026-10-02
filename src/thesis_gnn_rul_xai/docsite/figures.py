"""Draws the website's diagrams, every one of them from this repository's data.

Read README.md in this folder. It explains everything this file does, which
figures need the NASA files and what each one is teaching.
"""

from __future__ import annotations

import html
import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from ..data.labeling import piecewise_rul_curve
from ..data.sensors import PAPER_SENSOR_SYMBOLS, SENSORS, symbol_to_column
from ..graph.adjacency import (
    adjacency_report,
    build_physics_adjacency,
    normalize_adjacency,
    pairs_beyond_reach,
    to_networkx,
)
from ..utils.paths import raw_data_dir
from .worked_example import draw as draw_worked_example

__all__ = [
    "FIGURES",
    "DataMissing",
    "Figure",
    "figure_names",
    "render_figure",
]


class DataMissing(Exception):
    """Raised when a figure needs the NASA files and they are not present."""


@dataclass(frozen=True)
class Figure:
    """One diagram: the name a page asks for it by, how to draw it, and what it needs first."""

    name: str
    draw: Callable[[], str]
    needs_data: bool = False
    needs_torch: bool = False


# The drawing area every figure is laid out in, before the browser scales it.
# _graph_layout scales the sensor positions to it, so the playground, which
# draws its own sensor graph from those positions, imports these two as well.
_WIDTH = 860.0
_HEIGHT = 520.0

# The colour of each part of the engine, used only by the engine drawing.
# Stations in one group share a colour: HPT cooling and LPT cooling are both
# "turbine cooling". draw_engine reads which sensors sit at a station from the
# station field of the sensor table in data/sensors/. It chooses by hand only
# where each station is drawn, and types the footnote naming the kept sensors
# it leaves out, which a test in tests/test_docsite.py holds to the table.
# Corrected 2026-10-02, when this file was checked against its own code: the
# comment said the sensor graph used these colours too, untrue from the day it
# was written (2026-09-18, in the change that removed that graph), and that no
# station was typed here, while draw_engine typed all seven with their sensors.
_GROUP_HUE: dict[str, str] = {
    "fan and bypass": "var(--fig-fan, #2f7fd1)",
    "LPC outlet": "var(--fig-lpc, #6b4fbb)",
    "HPC outlet": "var(--fig-hpc, #c2521f)",
    "combustor": "var(--fig-burn, #b3261e)",
    "core shaft": "var(--fig-core, #8a6d00)",
    "LPT outlet": "var(--fig-lpt, #1c7a5e)",
    "turbine cooling": "var(--fig-cool, #0f6b86)",
    "bleed": "var(--fig-bleed, #6a6a6a)",
}
_STATION_GROUP: dict[str, str] = {
    "Fan inlet": "fan and bypass",
    "Bypass duct": "fan and bypass",
    "Fan shaft": "fan and bypass",
    "LPC outlet": "LPC outlet",
    "HPC outlet": "HPC outlet",
    "Combustor": "combustor",
    "Core shaft": "core shaft",
    "LPT outlet": "LPT outlet",
    "HPT cooling": "turbine cooling",
    "LPT cooling": "turbine cooling",
    "Bleed system": "bleed",
}
_STATION_HUE = {station: _GROUP_HUE[group] for station, group in _STATION_GROUP.items()}


_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def _word(count: int) -> str:
    """A small count in words, as a sentence says it, and in digits from eleven up."""
    return _WORDS[count] if 0 <= count < len(_WORDS) else str(count)


def _open(width: float = _WIDTH, height: float = _HEIGHT, extra: str = "") -> str:
    """The opening tag every figure shares, sized so the browser can scale it."""
    return (
        f'<svg class="figure-svg" viewBox="0 0 {width:g} {height:g}" '
        f'preserveAspectRatio="xMidYMid meet" role="img" '
        f'xmlns="http://www.w3.org/2000/svg">{extra}'
    )


def _text(
    x: float,
    y: float,
    body: str,
    *,
    size: float = 13,
    anchor: str = "middle",
    weight: str = "400",
    fill: str = "var(--fig-ink, #1a1a1a)",
    opacity: float = 1.0,
    family: str = "inherit",
    halo: bool = False,
) -> str:
    """One piece of text, escaped, with the settings the figures keep reusing."""
    style = f"font-family:{family};" if family != "inherit" else ""
    # A halo is an outline in the background colour drawn under the letters,
    # so a label stays readable over a fill whose shade changes as it animates.
    outline = (
        ' stroke="var(--fig-bg, #ffffff)" stroke-width="3" stroke-linejoin="round" '
        'paint-order="stroke"'
        if halo
        else ""
    )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-size="{size:g}" font-weight="{weight}" fill="{fill}" '
        f'opacity="{opacity:g}" style="{style}"{outline}>{html.escape(body)}</text>'
    )


def _mono(x: float, y: float, body: str, **kwargs: object) -> str:
    """A piece of text in the fixed-width face, for shapes and numbers."""
    # The website's stylesheet sets --mono; a notebook displaying the same
    # figure does not, so the list after the comma is what it falls back to.
    kwargs.setdefault("family", "var(--mono, Consolas, Menlo, monospace)")
    return _text(x, y, body, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The sensor graph
# --------------------------------------------------------------------------


def _graph_layout() -> tuple[nx.Graph, dict[str, tuple[float, float]], dict[str, int]]:
    """Place the 14 real sensors on the canvas, and count how many neighbours each has."""
    matrix, symbols = build_physics_adjacency()
    graph = to_networkx(matrix, symbols)
    raw = nx.kamada_kawai_layout(graph)

    xs = [float(p[0]) for p in raw.values()]
    ys = [float(p[1]) for p in raw.values()]
    span_x = max(xs) - min(xs) or 1.0
    span_y = max(ys) - min(ys) or 1.0
    margin_x, margin_y = 96.0, 84.0

    placed = {
        node: (
            margin_x + (float(p[0]) - min(xs)) / span_x * (_WIDTH - 2 * margin_x),
            margin_y + (float(p[1]) - min(ys)) / span_y * (_HEIGHT - 2 * margin_y - 40),
        )
        for node, p in raw.items()
    }
    degrees = {node: int(degree) for node, degree in graph.degree()}
    return graph, placed, degrees


def _node_radius(degree: int) -> float:
    """Node size, growing with how many neighbours a sensor has."""
    return 15.0 + 1.9 * degree


def _cut_sensors(graph: nx.Graph) -> tuple[set[str], str]:
    """The sensors whose removal splits the graph, and the words that key their ring."""
    cut = sorted(nx.articulation_points(graph))
    if not cut:
        return set(), ""
    if len(cut) > 1:
        return set(cut), "dashed ring: remove any one of these sensors and the graph splits"
    rest = graph.subgraph(set(graph.nodes()) - set(cut))
    parts = nx.number_connected_components(rest)
    return set(cut), (
        f"dashed ring: remove this sensor and the graph splits into {_word(parts)} parts"
    )


def _ring(x: float, y: float, radius: float) -> str:
    """The dashed ring drawn round a sensor whose removal splits the graph."""
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius + 5:.1f}" fill="none" '
        f'stroke="var(--fig-ink, #1a1a1a)" stroke-opacity="0.75" stroke-width="2" '
        f'stroke-dasharray="4 3" />'
    )


def draw_message_passing() -> str:
    """Information leaving P15 through three graph layers, animated over the real sensor graph."""
    graph, pos, degrees = _graph_layout()
    source, layers = "P15", 3
    distance = nx.single_source_shortest_path_length(graph, source)
    cut, cut_key = _cut_sensors(graph)

    # One second per layer, then a pause, then the whole thing again.
    step, pause = 1.6, 2.2
    cycle = step * (layers + 1) + pause

    edges = []
    for a, b in graph.edges():
        near = min(distance.get(a, 99), distance.get(b, 99))
        far = max(distance.get(a, 99), distance.get(b, 99))
        carries = far <= layers and far == near + 1
        begin = step * far
        pulse = (
            f'<animate attributeName="stroke-opacity" values="0.25;1;0.25" '
            f'keyTimes="0;0.5;1" dur="{step:g}s" begin="{begin:g}s" '
            f'repeatCount="indefinite" '
            f'repeatDur="{cycle:g}s" fill="freeze" />'
            if carries
            else ""
        )
        edges.append(
            f'<line x1="{pos[a][0]:.1f}" y1="{pos[a][1]:.1f}" '
            f'x2="{pos[b][0]:.1f}" y2="{pos[b][1]:.1f}" '
            f'stroke="var(--fig-edge, #b9c0cb)" stroke-width="{2.4 if carries else 1.3:g}" '
            f'stroke-opacity="0.3">{pulse}</line>'
        )

    nodes = []
    for node, (x, y) in pos.items():
        radius = _node_radius(degrees[node])
        hops = distance.get(node, 99)
        reached = hops <= layers
        colour = (
            "var(--fig-accent, #2f7fd1)" if reached else "var(--fig-muted, #9aa3b0)"
        )
        fill_animation = (
            f'<animate attributeName="fill-opacity" '
            f'values="0.12;0.12;0.95;0.95" '
            f'keyTimes="0;{max(hops, 0.001) * step / cycle:.4f};'
            f'{min((hops * step + 0.45) / cycle, 0.999):.4f};1" '
            f'dur="{cycle:g}s" repeatCount="indefinite" />'
            if reached
            else ""
        )
        badge = (
            _mono(x, y + radius + 15, f"{hops} hop" + ("s" if hops != 1 else ""),
                  size=10, opacity=0.6)
            if reached
            else _mono(x, y + radius + 15, f"{hops} hops — never", size=10,
                       fill="var(--fig-warn, #c2521f)", opacity=0.85)
        )
        nodes.append(
            f"<g>{_ring(x, y, radius) if node in cut else ''}"
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{radius:.1f}' fill='{colour}' "
            f"fill-opacity='{0.95 if hops == 0 else 0.12:g}' "
            f"stroke='{colour}' stroke-width='2'>{fill_animation}</circle>"
            + _mono(x, y + 4, node, size=12, weight="600",
                    fill="var(--fig-ink, #1a1a1a)", halo=True)
            + badge
            + "</g>"
        )

    counts = ", ".join(_word(n) for n in range(1, layers))
    caption = _text(
        _WIDTH / 2,
        _HEIGHT - 22,
        f"Information leaving {source} after {counts} and {_word(layers)} graph layers",
        size=13,
        opacity=0.72,
    ) + (
        _text(_WIDTH / 2, _HEIGHT - 4, cut_key, size=11.5, opacity=0.6) if cut_key else ""
    )
    return _open() + "".join(edges) + "".join(nodes) + caption + "</svg>"


def draw_adjacency_matrix() -> str:
    """The 14 by 14 table itself, before and after the row normalisation."""
    matrix, symbols = build_physics_adjacency()
    normalized = normalize_adjacency(matrix, method="row")

    cell, left, top = 21.0, 74.0, 112.0
    panels = []
    for index, (data, title) in enumerate(
        ((matrix, "A — the paper's Table 3"), (normalized, "D⁻¹(A + I) — row normalised")),
    ):
        offset = index * (14 * cell + 108) + left
        squares = []
        for row in range(14):
            for column in range(14):
                value = float(data[row, column])
                if value <= 0:
                    continue
                squares.append(
                    f'<rect x="{offset + column * cell:.1f}" '
                    f'y="{top + row * cell:.1f}" width="{cell - 1.5:.1f}" '
                    f'height="{cell - 1.5:.1f}" rx="2" '
                    f'fill="var(--fig-accent, #2f7fd1)" '
                    f'fill-opacity="{0.18 + 0.82 * min(value, 1.0):.3f}">'
                    f"<title>{symbols[row]} → {symbols[column]} = {value:.3f}</title>"
                    f"</rect>"
                )
        labels = "".join(
            f'<g transform="translate({offset + i * cell + cell / 2 + 2:.1f},{top - 6:.1f}) '
            f'rotate(-90)">'
            + _mono(0, 0, symbols[i], size=8.5, anchor="start", opacity=0.7)
            + "</g>"
            for i in range(14)
        ) + "".join(
            _mono(offset - 6, top + i * cell + cell / 2 + 3, symbols[i],
                  size=8.5, anchor="end", opacity=0.7)
            for i in range(14)
        )
        panels.append(
            _text(offset + 14 * cell / 2, top - 58, title, size=13, weight="600")
            + "".join(squares)
            + labels
        )

    return _open(880, top + 14 * cell + 20) + "".join(panels) + "</svg>"


def draw_reach() -> str:
    """How far apart every pair of sensors is, measured against what three graph layers reach."""
    matrix, symbols = build_physics_adjacency()
    report = adjacency_report(matrix, symbols)
    histogram: dict[int, int] = dict(report["distance_histogram"])  # type: ignore[arg-type]
    layers = 3
    beyond = pairs_beyond_reach(matrix, symbols, layers)

    total = sum(histogram.values())
    left, base, height = 180.0, 280.0, 200.0
    bar_w, gap = 92.0, 40.0

    bars = []
    for index, (distance, count) in enumerate(sorted(histogram.items())):
        x = left + index * (bar_w + gap)
        bar_h = height * count / max(histogram.values())
        reachable = distance <= layers
        colour = (
            "var(--fig-accent, #2f7fd1)" if reachable
            else "var(--fig-warn, #c2521f)"
        )
        bars.append(
            f'<rect x="{x:.1f}" y="{base - bar_h:.1f}" width="{bar_w:g}" '
            f'height="{bar_h:.1f}" rx="5" fill="{colour}" fill-opacity="0.82" />'
            + _mono(x + bar_w / 2, base - bar_h - 10, str(count), size=14,
                    weight="600")
            + _text(x + bar_w / 2, base + 22,
                    f"{distance} edge" + ("s" if distance != 1 else ""), size=12)
            + _text(x + bar_w / 2, base + 40,
                    f"{100 * count / total:.0f} % of pairs", size=10, opacity=0.55)
            + _text(x + bar_w / 2, base + 60,
                    f"{'within' if reachable else 'beyond'} {_word(layers)} layers",
                    size=10, weight="600", fill=colour)
        )

    names = ", ".join(f"{a} and {b}" for a, b, _ in beyond)
    return (
        _open(_WIDTH, base + 116)
        + _text(_WIDTH / 2, 40, f"How far apart the {total} sensor pairs are", size=16,
                weight="600")
        + "".join(bars)
        + (
            _text(_WIDTH / 2, base + 96,
                  f"The {_word(len(beyond))} pair{'s' if len(beyond) != 1 else ''} "
                  f"beyond {_word(layers)} layers: {names}.",
                  size=12, opacity=0.72)
            if beyond
            else ""
        )
        + "</svg>"
    )


# --------------------------------------------------------------------------
# The label, the filter and the window — these read the real NASA files
# --------------------------------------------------------------------------


def _require_data() -> Path:
    """The raw data directory, or an explanation of how to get the files."""
    directory = raw_data_dir()
    if not (directory / "train_FD001.txt").is_file():
        raise DataMissing(
            "This figure is drawn from the real FD001 file, and "
            "data/raw/train_FD001.txt is not present. This project never "
            "substitutes invented signals. Download the Turbofan Engine "
            "Degradation Simulation Data Set from the NASA Prognostics Center "
            "of Excellence repository and put the three FD001 files in "
            "data/raw/, then reload this page."
        )
    return directory


def _one_engine(unit: int = 1) -> tuple[pd.DataFrame, int]:
    """One engine's real history from FD001, and how many cycles it ran."""
    from ..data.loading import load_subset

    _require_data()
    data = load_subset("FD001", data_dir=raw_data_dir())
    frame = data.train[data.train["unit"] == unit]
    return frame, int(frame["cycle"].max())


def _cycle_axis(unit: int = 1) -> str:
    """The horizontal axis title of a figure that plots one training engine of FD001."""
    return f"flight cycle of engine {unit} (FD001 training data)"


def _axes(
    left: float,
    top: float,
    width: float,
    height: float,
    x_label: str,
    y_label: str,
) -> str:
    """The two lines and two titles every plotted figure sits inside."""
    return (
        f'<line x1="{left}" y1="{top + height}" x2="{left + width}" '
        f'y2="{top + height}" stroke="var(--fig-axis, #8a93a0)" stroke-width="1.2" />'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" '
        f'stroke="var(--fig-axis, #8a93a0)" stroke-width="1.2" />'
        + _text(left + width / 2, top + height + 40, x_label, size=12, opacity=0.7)
        + f'<g transform="translate({left - 46},{top + height / 2}) rotate(-90)">'
        + _text(0, 0, y_label, size=12, opacity=0.7)
        + "</g>"
    )


def draw_rul_curve() -> str:
    """The piecewise-linear target, on the true length of a real engine."""
    _, total = _one_engine(1)
    rul_max = 125
    curve = piecewise_rul_curve(total, rul_max)

    left, top, width, height = 90.0, 60.0, 700.0, 300.0
    to_x = lambda c: left + width * (c - 1) / max(total - 1, 1)  # noqa: E731
    to_y = lambda r: top + height * (1 - r / max(total - 1, rul_max))  # noqa: E731

    points = " ".join(f"{to_x(i):.1f},{to_y(float(v)):.1f}" for i, v in enumerate(curve, start=1))
    knee = total - rul_max

    naive = (
        f'<polyline data-curve="raw" points="{to_x(1):.1f},{to_y(total - 1):.1f} '
        f'{to_x(total):.1f},{to_y(0):.1f}" fill="none" '
        f'stroke="var(--fig-muted, #9aa3b0)" stroke-width="1.6" '
        f'stroke-dasharray="6 4" />'
    )

    ticks = "".join(
        _mono(left - 10, to_y(value) + 4, str(value), size=10, anchor="end", opacity=0.6)
        + f'<line x1="{left - 5}" y1="{to_y(value):.1f}" x2="{left}" '
        f'y2="{to_y(value):.1f}" stroke="var(--fig-axis, #8a93a0)" />'
        for value in (0, 50, 100, 125, 150, total - 1)
    ) + "".join(
        _mono(to_x(value), top + height + 20, str(value), size=10, opacity=0.6)
        for value in (1, 50, 100, 150, total)
        if value <= total
    )

    return (
        _open(_WIDTH, top + height + 60)
        + _text(_WIDTH / 2, 24, f"train_FD001.txt · engine 1 · recorded cycles 1-{total}", size=13)
        + _axes(left, top, width, height, _cycle_axis(1), "remaining life (cycles)")
        + ticks
        + naive
        + f'<line x1="{to_x(knee):.1f}" y1="{to_y(rul_max)}" x2="{to_x(knee):.1f}" '
        f'y2="{top + height}" stroke="var(--fig-warn, #c2521f)" '
        f'stroke-width="1.2" stroke-dasharray="3 3" opacity="0.7" />'
        + f'<polyline data-curve="capped" points="{points}" fill="none" '
        f'stroke="var(--fig-accent, #2f7fd1)" stroke-width="2.6" '
        f'stroke-linejoin="round" />'
        + _text(to_x(knee / 2), to_y(rul_max) - 14, "held at the ceiling", size=12,
                fill="var(--fig-accent, #2f7fd1)", weight="600")
        + _text(to_x(knee) + 8, to_y(rul_max) + 22, f"cycle {knee}: {rul_max} left", size=11,
                anchor="start", fill="var(--fig-warn, #c2521f)")
        + _text(to_x(total * 0.76), to_y(rul_max * 0.72), "both curves coincide after the ceiling",
                size=12, fill="var(--fig-accent, #2f7fd1)", weight="600")
        + _text(to_x(total * 0.35), to_y(total - 20), f"Dashed: {total} - cycle (before capping)",
                size=11, opacity=0.8)
        + _text(to_x(total * 0.23), to_y(75), f"Solid: min({total} - cycle, {rul_max})",
                size=11, fill="var(--fig-accent, #2f7fd1)")
        + "</svg>"
    )


def draw_smoothing() -> str:
    """One real sensor trace, before and after the noise filter of step 5."""
    from ..data.smoothing import ewma_series

    frame, _ = _one_engine(1)
    column = symbol_to_column("T50")
    raw = frame[column].to_numpy(dtype=float)
    smooth = ewma_series(raw, 0.5)

    left, top, width, height = 90.0, 60.0, 700.0, 300.0
    low, high = float(raw.min()), float(raw.max())
    pad = (high - low) * 0.08 or 1.0
    low, high = low - pad, high + pad
    to_x = lambda i: left + width * i / max(len(raw) - 1, 1)  # noqa: E731
    to_y = lambda v: top + height * (1 - (v - low) / (high - low))  # noqa: E731

    def path(values: Sequence[float]) -> str:
        return " ".join(f"{to_x(i):.1f},{to_y(float(v)):.1f}" for i, v in enumerate(values))

    jitter_before = float(np.abs(np.diff(raw)).mean())
    jitter_after = float(np.abs(np.diff(smooth)).mean())
    removed = 100.0 * (1 - jitter_after / jitter_before)

    ticks = "".join(
        _mono(left - 10, to_y(value) + 4, f"{value:.0f}", size=10, anchor="end",
              opacity=0.6)
        for value in np.linspace(low + pad, high - pad, 5)
    )

    return (
        _open(_WIDTH, top + height + 84)
        + _axes(left, top, width, height, _cycle_axis(1), "T50 (degrees Rankine)")
        + ticks
        + f'<polyline points="{path(raw)}" fill="none" '
        f'stroke="var(--fig-muted, #9aa3b0)" stroke-width="1.2" opacity="0.85" />'
        + f'<polyline points="{path(smooth)}" fill="none" '
        f'stroke="var(--fig-accent, #2f7fd1)" stroke-width="2.4" />'
        + _text(left + 12, top + 18, "raw reading", size=12, anchor="start",
                fill="var(--fig-muted, #9aa3b0)", weight="600")
        + _text(left + 12, top + 36, "after the filter (alpha = 0.5)", size=12,
                anchor="start", fill="var(--fig-accent, #2f7fd1)", weight="600")
        + _text(
            _WIDTH / 2,
            top + height + 70,
            f"Average step between neighbouring cycles: {jitter_before:.3f} before, "
            f"{jitter_after:.3f} after — {removed:.0f} % of the jitter removed.",
            size=12,
            opacity=0.7,
        )
        + "</svg>"
    )


def draw_windowing() -> str:
    """The sliding window of step 8, animated over a real engine's history."""
    frame, _ = _one_engine(1)
    column = symbol_to_column("T50")
    values = frame[column].to_numpy(dtype=float)
    window = 30

    left, top, width, height = 90.0, 50.0, 700.0, 230.0
    low, high = float(values.min()), float(values.max())
    pad = (high - low) * 0.1 or 1.0
    to_x = lambda i: left + width * i / max(len(values) - 1, 1)  # noqa: E731
    to_y = lambda v: top + height * (1 - (v - low + pad) / (high - low + 2 * pad))  # noqa: E731

    trace = " ".join(f"{to_x(i):.1f},{to_y(float(v)):.1f}" for i, v in enumerate(values))
    window_width = width * (window - 1) / max(len(values) - 1, 1)
    travel = width - window_width

    box = (
        f'<rect x="{left:.1f}" y="{top - 10:.1f}" width="{window_width:.1f}" '
        f'height="{height + 20:.1f}" rx="4" fill="var(--fig-accent, #2f7fd1)" '
        f'fill-opacity="0.14" stroke="var(--fig-accent, #2f7fd1)" '
        f'stroke-width="1.8">'
        f'<animate attributeName="x" from="{left:.1f}" to="{left + travel:.1f}" '
        f'dur="9s" repeatCount="indefinite" /></rect>'
    )
    marker = (
        f'<line x1="{left + window_width:.1f}" y1="{top - 16:.1f}" '
        f'x2="{left + window_width:.1f}" y2="{top + height + 16:.1f}" '
        f'stroke="var(--fig-warn, #c2521f)" stroke-width="2.4">'
        f'<animate attributeName="x1" from="{left + window_width:.1f}" '
        f'to="{left + width:.1f}" dur="9s" repeatCount="indefinite" />'
        f'<animate attributeName="x2" from="{left + window_width:.1f}" '
        f'to="{left + width:.1f}" dur="9s" repeatCount="indefinite" /></line>'
    )

    return (
        _open(_WIDTH, top + height + 84)
        + _axes(left, top, width, height, _cycle_axis(1), "T50")
        + f'<polyline points="{trace}" fill="none" '
        f'stroke="var(--fig-muted, #9aa3b0)" stroke-width="1.6" />'
        + box
        + marker
        + _text(left + window_width / 2, top - 20,
                "the window the model receives", size=11.5,
                fill="var(--fig-accent, #2f7fd1)")
        + _text(
            _WIDTH / 2,
            top + height + 70,
            "The orange line is the last cycle of the window.",
            size=11.5,
            opacity=0.55,
        )
        + "</svg>"
    )


# --------------------------------------------------------------------------
# The network and the explanation
# --------------------------------------------------------------------------


def draw_architecture() -> str:
    """Every layer of the w/o LSTM network, with the shape and the learned numbers each really has."""
    from ..config import PreprocessingConfig
    from ..models.estgnn import ESTGNN

    matrix, _ = build_physics_adjacency()
    # With no stage named, the network keeps only each window's last cycle:
    # the configuration this project calls w/o LSTM.
    model = ESTGNN(
        normalize_adjacency(matrix, method="row"),
        window_size=PreprocessingConfig().window_size,
    )
    summary = model.layer_summary()
    # layer_summary lists the temporal stage and then Table 4's Input row, and
    # both describe the same tensor, so they are drawn as one Input box that
    # names the stage its numbers come from and counts what that stage learns.
    stage, first = summary[0], summary[1]
    if (stage["layer"], first["layer"]) != ("Temporal", "Input"):
        raise ValueError(
            "layer_summary no longer starts with the temporal stage and then the "
            f"Input row, but with {stage['layer']} and {first['layer']}."
        )
    rows = [
        {**first, "activation": "w/o LSTM", "parameters": stage["parameters"]},
        *summary[2:],
    ]

    # The boxes shrink to fit the figure's width, because the number of layers
    # depends on how many graph layers the network has.
    box_height, gap, margin = 96.0, 20.0, 24.0
    box_width = min(96.0, (_WIDTH - 2 * margin - (len(rows) - 1) * gap) / len(rows))
    left = (_WIDTH - (len(rows) * box_width + (len(rows) - 1) * gap)) / 2
    top = 100.0

    blocks = []
    for index, row in enumerate(rows):
        x = left + index * (box_width + gap)
        is_graph = str(row["type"]) == "GCN"
        colour = (
            "var(--fig-accent, #2f7fd1)" if is_graph else "var(--fig-lpt, #1c7a5e)"
        )
        if int(row["parameters"]) == 0:
            colour = "var(--fig-muted, #9aa3b0)"
        arrow = (
            f'<path d="M{x - gap + 4:.1f} {top + box_height / 2:.1f} '
            f'h{gap - 12:.1f}" stroke="var(--fig-axis, #8a93a0)" stroke-width="1.6" '
            f'marker-end="url(#arrow)" />'
            if index
            else ""
        )
        # A layer with no activation, such as Flatten, leaves its second line empty.
        second = "" if str(row["activation"]) == "-" else str(row["activation"])
        blocks.append(
            arrow
            + f'<rect x="{x:.1f}" y="{top:.1f}" width="{box_width:g}" '
            f'height="{box_height:g}" rx="8" fill="{colour}" fill-opacity="0.14" '
            f'stroke="{colour}" stroke-width="1.8" />'
            + _text(x + box_width / 2, top + 26, str(row["layer"]), size=13,
                    weight="600")
            + _text(x + box_width / 2, top + 45, second, size=10.5, opacity=0.65)
            + _mono(x + box_width / 2, top + 68, str(row["out"]).replace("B, ", ""),
                    size=9.5, opacity=0.8)
            + _mono(x + box_width / 2, top + 86,
                    f"{int(row['parameters']):,}".replace(",", " "), size=10,
                    weight="600", fill=colour)
            + _text(x + box_width / 2, top + box_height + 22,
                    "graph" if is_graph else "", size=10, opacity=0.5)
        )

    defs = (
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="var(--fig-axis, #8a93a0)" />'
        "</marker></defs>"
    )

    return (
        _open(_WIDTH, top + box_height + 44, defs)
        + _text(_WIDTH / 2, 40, "The network of the paper's Table 4", size=16,
                weight="600")
        + _text(_WIDTH / 2, 64, "Each box shows what comes out of it, and how "
                "many numbers it learns", size=12, opacity=0.65)
        + "".join(blocks)
        + "</svg>"
    )


def _architecture_block(x: float, y: float, columns: int, rows: int, colour: str) -> str:
    """Draw a sensor-by-feature grid with a shallow raised edge."""
    width, height, depth = max(26, columns * 1.65), rows * 15, 17
    out = (
        f'<path d="M{x} {y} l{depth} {-depth} h{width} l{-depth} {depth} Z" '
        f'fill="{colour}" fill-opacity="0.45" stroke="{colour}" stroke-width="0.8"/>'
        f'<path d="M{x + width} {y} l{depth} {-depth} v{height} l{-depth} {depth} Z" '
        f'fill="{colour}" fill-opacity="0.85" stroke="{colour}" stroke-width="0.8"/>'
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'fill="{colour}" fill-opacity="0.64" stroke="{colour}"/>'
    )
    for row in range(1, rows):
        out += (f'<path d="M{x} {y + row * 15} h{width}" '
                'stroke="var(--fig-bg)" stroke-opacity="0.5" stroke-width="0.8"/>')
    for column in range(1, columns):
        out += (f'<path d="M{x + width * column / columns} {y} v{height}" '
                'stroke="var(--fig-bg)" stroke-opacity="0.3" stroke-width="0.55"/>')
    return out


def draw_architecture_overview() -> str:
    """Illustrate the default model with sensor graphs and connected output units."""
    from ..config import PreprocessingConfig
    from ..models.estgnn import ESTGNN

    matrix, symbols = build_physics_adjacency()
    model = ESTGNN(normalize_adjacency(matrix, method="row"),
                   window_size=PreprocessingConfig().window_size)
    summary = model.layer_summary()
    rows = [{**summary[1], "parameters": summary[0]["parameters"]}, *summary[2:]]
    blue, rose, purple, gold, green = "#6f9fc7", "#d69ba8", "#b294d2", "#d6b74f", "#99b66a"
    centres = [70, 220, 373, 526, 680, 842, 979]
    defs = ('<title id="architecture-overview-title">From sensor readings to remaining life</title>'
            '<desc id="architecture-overview-desc">The default network without LSTM. '
            'Each graph has fourteen sensor nodes and the connections used by the model. '
            'After three graph layers with ReLU, the sensor values become one list, then fully connected layers '
            'produce one prediction. Only some list entries, units and connections are drawn.</desc>'
            '<defs><marker id="architecture-overview-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="5" markerHeight="5" orient="auto">'
            '<path d="M0 0 L10 5 L0 10 Z" fill="var(--fig-axis)"/></marker></defs>')
    out = (_open(1120, 548, defs)
           .replace('role="img"', 'role="img" aria-labelledby="architecture-overview-title architecture-overview-desc"'))
    for left, right, label, colour in [(12, 140, "Input", blue),
                                       (150, 935, "Hidden layers", gold),
                                       (945, 1110, "Output", green)]:
        points = f"{left},20 {right - 19},20 {right},42 {right - 19},64 {left},64 {left + 19},42"
        out += (f'<polygon points="{points}" fill="{colour}" fill-opacity="0.24"/>'
                + _text((left + right) / 2 + 5, 48, label, size=20, weight="600"))

    titles = ["Sensor input", "GCN-1", "GCN-2", "GCN-3", "Flatten", "FC-1", "FC-2"]
    subtitles = ["Last reading", "ReLU", "ReLU", "ReLU", "One list", "ReLU", "Sigmoid"]
    widths = [model.in_features, *(layer.out_features for layer in model.graph_layers)]
    for index, (row, centre, title, subtitle) in enumerate(zip(rows, centres, titles, subtitles, strict=True)):
        count = int(row["parameters"])
        shape = str(row["out"]).replace("B, ", "")
        colour = [blue, rose, rose, rose, purple, gold, green][index]
        out += (f'<g data-architecture-layer="{html.escape(str(row["layer"]))}" '
                f'data-parameters="{count}" data-shape="{html.escape(shape)}">'
                + _text(centre, 107, title, size=18, weight="600")
                + _text(centre, 132, subtitle, size=14, opacity=0.75))
        if index < 4:
            width = max(26, widths[index] * 1.65)
            if index == 0:
                out += _architecture_block(centre - width / 2 - 8.5, 190,
                                           widths[index], model.n_nodes, colour)
            else:
                angles = np.linspace(-np.pi / 2, 3 * np.pi / 2, model.n_nodes, endpoint=False)
                points = [(centre + 61 * np.cos(angle), 295 + 89 * np.sin(angle)) for angle in angles]
                for a in range(model.n_nodes):
                    for b in range(a + 1, model.n_nodes):
                        if matrix[a, b] or matrix[b, a]:
                            ax, ay = points[a]
                            bx, by = points[b]
                            out += (f'<path data-graph-edge="{symbols[a]} {symbols[b]}" '
                                    f'd="M{ax:.2f} {ay:.2f} L{bx:.2f} {by:.2f}" '
                                    'stroke="var(--fig-axis)" stroke-opacity="0.5" stroke-width="1"/>')
                for symbol, (node_x, node_y) in zip(symbols, points, strict=True):
                    out += (f'<circle data-sensor="{symbol}" cx="{node_x:.2f}" cy="{node_y:.2f}" '
                            f'r="6" fill="{rose}" stroke="var(--fig-bg)" stroke-width="1.2">'
                            f'<title>{symbol}</title></circle>')
            if index < 3:
                start = centre + (width / 2 + 15 if index == 0 else 70)
                end = centres[index + 1] - 70
                out += (f'<path d="M{start} 295 H{end}" stroke="var(--fig-axis)" '
                        'stroke-width="1.8" fill="none" marker-end="url(#architecture-overview-arrow)"/>')
            dimensions = f"{model.n_nodes} \u00d7 {widths[index]}"
            meaning = ""
        elif index == 4:
            for offset, y in enumerate([181, 209, 237, 265, 321, 349, 377]):
                out += (f'<rect x="{centre - 13}" y="{y}" width="26" height="22" rx="3" '
                        f'fill="{purple}" fill-opacity="{0.55 + (offset % 2) * 0.15}"/>')
            out += _text(centre, 310, "⋮", size=24, fill=purple)
            for start_y, end_y in [(206, 193), (295, 276), (384, 388)]:
                out += (f'<path d="M{centres[3] + 70} {start_y} L{centre - 25} {end_y}" '
                        f'stroke="{rose}" stroke-width="2" fill="none" '
                        'marker-end="url(#architecture-overview-arrow)"/>')
            dimensions, meaning = str(model.flattened_size), "numbers"
        elif index == 5:
            for source_y in [192, 248, 360, 388]:
                for dest_y in [185, 229, 273, 329, 373, 417]:
                    out += (f'<path d="M{centres[4] + 14} {source_y} L{centre - 16} {dest_y}" '
                            f'stroke="{purple}" stroke-opacity="0.34" stroke-width="1"/>')
            for y in [185, 229, 273, 329, 373, 417]:
                out += (f'<path d="M{centre + 16} {y} L{centres[6] - 23} 300" '
                        f'stroke="{gold}" stroke-width="1.4" stroke-opacity="0.8"/>'
                        f'<circle cx="{centre}" cy="{y}" r="16" fill="{gold}" '
                        'fill-opacity="0.6"/>')
            out += _text(centre, 309, "⋮", size=24, fill=gold)
            dimensions, meaning = str(model.fc1.out_features), "numbers"
        else:
            out += (f'<circle cx="{centre}" cy="300" r="24" fill="{green}" fill-opacity="0.7"/>'
                    f'<path d="M{centre + 29} 300 H1042" stroke="var(--fig-axis)" '
                    'stroke-width="1.8" marker-end="url(#architecture-overview-arrow)"/>'
                    + _text(1067, 247, "Remaining", size=15, weight="600")
                    + _text(1067, 268, "life", size=15, weight="600")
                    + _text(1067, 306, "RUL", size=19, weight="600", fill="var(--fig-ink)")
                    + _text(1067, 335, "in cycles", size=14)
                    + _text(centre, 369, "Sigmoid → 0 to 1", size=14)
                    + _text(centre, 392, f"then \u00d7 {model.rul_max}", size=14))
            dimensions, meaning = str(model.fc2.out_features), "fraction"
        out += (_text(centre, 456, dimensions, size=23, weight="600")
                + _text(centre, 478, meaning, size=15, opacity=0.75)
                + _text(centre, 507, f'{count:,}'.replace(',', ' ') + " learned", size=16, weight="600")
                + '</g>')

    out += (_text(20, 535, "Default model: last reading per sensor (w/o LSTM).",
                  size=14, anchor="start", opacity=0.8) + '</svg>')
    return ('<div class="architecture-overview-scroll" tabindex="0" role="region" '
            'aria-label="Network architecture; scroll horizontally on a narrow screen">' + out + '</div>'
            '<p class="architecture-overview-caption">GCN = graph convolution; FC = fully connected. '
            'Each graph has the same sensors and connections. Sizes show sensors \u00d7 numbers per sensor. '
            'Only some entries and connections are shown after flattening. '
            '“Learned” counts the weights and added constants adjusted during training.</p>')


def draw_shapley() -> str:
    """What a Shapley value averages over, drawn for a three-player example."""
    players = ("A", "B", "C")
    orders = list(itertools.permutations(players))

    # The rows are about 350 wide with their notes, so this centres them.
    left, top, row_height = 290.0, 96.0, 44.0
    cell, gap = 56.0, 12.0

    rows = []
    for index, order in enumerate(orders):
        y = top + index * row_height
        rows.append(_mono(left - 22, y + 21, f"#{index + 1}", size=11,
                          anchor="end", opacity=0.5))
        for position, player in enumerate(order):
            x = left + position * (cell + gap)
            highlight = player == "B"
            colour = (
                "var(--fig-accent, #2f7fd1)" if highlight
                else "var(--fig-muted, #9aa3b0)"
            )
            rows.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell:g}" height="{32:g}" '
                f'rx="5" fill="{colour}" fill-opacity="{0.85 if highlight else 0.2:g}" '
                f'stroke="{colour}" stroke-width="1.4" />'
                + _mono(x + cell / 2, y + 21, player, size=13, weight="600",
                        fill="#ffffff" if highlight else "var(--fig-ink, #1a1a1a)")
            )
        before = order[: order.index("B")]
        rows.append(
            _text(
                left + 3 * (cell + gap) + 16,
                y + 21,
                "B joins " + (f"after {', '.join(before)}" if before else "first"),
                size=11.5,
                anchor="start",
                opacity=0.7,
            )
        )

    return (
        _open(_WIDTH, top + len(orders) * row_height + 12)
        + _text(_WIDTH / 2, 40, "A Shapley value is an average over every "
                "joining order", size=16, weight="600")
        + "".join(
            _text(left + position * (cell + gap) + cell / 2, top - 10, word,
                  size=10.5, opacity=0.55)
            for position, word in enumerate(("first", "second", "third"))
        )
        + "".join(rows)
        + "</svg>"
    )


# --------------------------------------------------------------------------
# The engine and its sensors: where each sits, and which columns are kept
# --------------------------------------------------------------------------


def draw_engine() -> str:
    """A turbofan drawn to place the sensors, with each station named."""
    # Where each station sits along the flow path is chosen here; which
    # sensors sit at it is read from the station field of the sensor table.
    along_the_flow: tuple[tuple[float, str], ...] = (
        (120, "Fan inlet"),
        (232, "LPC outlet"),
        (348, "HPC outlet"),
        (452, "Combustor"),
        (556, "HPT cooling"),
        (664, "LPT outlet"),
        (760, "LPT cooling"),
    )
    stations = [
        (x, name, tuple(sensor.symbol for sensor in SENSORS if sensor.station == name))
        for x, name in along_the_flow
    ]
    empty = [name for _, name, symbols in stations if not symbols]
    if empty:
        raise ValueError(f"No sensor in the sensor table sits at the station(s) {empty}.")
    axis_y = 232.0
    body = (
        f'<path d="M100 {axis_y - 86} L200 {axis_y - 62} L720 {axis_y - 40} '
        f'L800 {axis_y - 52} L800 {axis_y + 52} L720 {axis_y + 40} '
        f'L200 {axis_y + 62} L100 {axis_y + 86} Z" '
        f'fill="var(--fig-muted, #9aa3b0)" fill-opacity="0.12" '
        f'stroke="var(--fig-axis, #8a93a0)" stroke-width="1.6" />'
        f'<path d="M210 {axis_y - 30} L710 {axis_y - 20} L710 {axis_y + 20} '
        f'L210 {axis_y + 30} Z" fill="var(--fig-hpc, #c2521f)" '
        f'fill-opacity="0.14" stroke="var(--fig-axis, #8a93a0)" '
        f'stroke-width="1.2" />'
        f'<rect x="112" y="{axis_y - 78}" width="14" height="156" rx="7" '
        f'fill="var(--fig-fan, #2f7fd1)" fill-opacity="0.5" />'
        f'<rect x="430" y="{axis_y - 24}" width="46" height="48" rx="6" '
        f'fill="var(--fig-burn, #b3261e)" fill-opacity="0.35" />'
    )

    flow = (
        f'<path d="M60 {axis_y} L96 {axis_y}" stroke="var(--fig-axis, #8a93a0)" '
        f'stroke-width="2" marker-end="url(#arrow2)" />'
        + _text(58, axis_y - 12, "air in", size=11, anchor="end", opacity=0.6)
        + f'<path d="M806 {axis_y} L846 {axis_y}" '
        f'stroke="var(--fig-axis, #8a93a0)" stroke-width="2" '
        f'marker-end="url(#arrow2)" />'
        + _text(848, axis_y - 12, "thrust", size=11, anchor="end", opacity=0.6)
    )

    marks = []
    for index, (x, name, symbols) in enumerate(stations):
        above = index % 2 == 0
        tip = axis_y - 96 if above else axis_y + 96
        label_y = tip - 44 if above else tip + 20
        colour = _STATION_HUE.get(name, "var(--fig-accent, #2f7fd1)")
        kept = [s for s in symbols if s in PAPER_SENSOR_SYMBOLS]
        dropped = [s for s in symbols if s not in PAPER_SENSOR_SYMBOLS]
        marks.append(
            f'<line x1="{x}" y1="{axis_y - 70 if above else axis_y + 70}" '
            f'x2="{x}" y2="{tip}" stroke="{colour}" stroke-width="1.4" '
            f'stroke-dasharray="3 3" />'
            f'<circle cx="{x}" cy="{axis_y - 70 if above else axis_y + 70}" r="4" '
            f'fill="{colour}" />'
            + _text(x, label_y, name, size=11, weight="600", fill=colour)
            + _mono(x, label_y + 16, " ".join(kept), size=11, weight="600")
            + _mono(x, label_y + 31, " ".join(dropped), size=10, opacity=0.4)
        )

    defs = (
        '<defs><marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="var(--fig-axis, #8a93a0)" />'
        "</marker></defs>"
    )

    return (
        _open(_WIDTH, 486, defs)
        + _text(_WIDTH / 2, 34, "Where the sensors sit", size=16, weight="600")
        + body
        + flow
        + "".join(marks)
        + _text(_WIDTH / 2, 424, "Bold names are among the 14 the paper keeps; "
                "faded ones are dropped.", size=11, opacity=0.55)
        + _text(_WIDTH / 2, 441, "Six of the 14 are not drawn: the shaft speeds "
                "(Nf, NRf, Nc), the bypass duct (P15, BPR) and the bleed system "
                "(htBleed).", size=11, opacity=0.55)
        + _text(_WIDTH / 2, 457, "They belong to a shaft, the bypass duct or the "
                "bleed system, not to one station on the flow path drawn here.",
                size=11, opacity=0.55)
        + _text(_WIDTH / 2, 476, "Schematic, drawn for this page. It is not a "
                "figure from the paper and is not to scale.", size=10.5,
                opacity=0.42)
        + "</svg>"
    )


def draw_sensor_selection() -> str:
    """Move the real selected column names into a second schematic table."""
    width, step, left = 940, 42, 26
    column_height = 124
    out = (_open(width, 400).replace('class="figure-svg"', 'class="figure-svg sensor-selection"', 1)
           + '<title>Selecting sensor columns</title><desc>Twenty-one raw sensor columns above; '
           'the fourteen model input columns move below, retaining their height, every five seconds. '
           'Cells are empty schematic rows, not generated readings.</desc>'
           + _text(left, 25, "Raw sensor table: all 21 sensor columns", anchor="start", size=15)
           + _text(left, 237, "Model input table: selected columns, same row order", anchor="start", size=15))
    kept = set(PAPER_SENSOR_SYMBOLS)
    for index, sensor in enumerate(SENSORS):
        x = left + index * step
        colour = 'var(--fig-accent)' if sensor.symbol in kept else 'var(--fig-axis)'
        out += f'<g class="select-source {"is-kept" if sensor.symbol in kept else "is-excluded"}">'
        out += f'<rect x="{x}" y="44" width="{step - 2}" height="{column_height}" rx="3" fill="{colour}" fill-opacity=".08" stroke="{colour}"/>'
        out += f'<text transform="translate({x + 24},106) rotate(-90)" fill="{colour}" font-size="11">{sensor.symbol}</text>'
        for y in (116, 133, 150):
            out += f'<path d="M{x} {y}h{step - 2}" stroke="{colour}" stroke-opacity=".3"/>'
        out += '</g>'
        if sensor.symbol not in kept:
            continue
        target = left + PAPER_SENSOR_SYMBOLS.index(sensor.symbol) * 63
        out += f'<g class="select-picked" style="--from-x:{x}px;--to-x:{target}px">'
        out += f'<rect width="{step - 2}" height="{column_height}" rx="3" fill="var(--fig-bg)" stroke="var(--fig-accent)"/>'
        out += f'<text transform="translate(24,62) rotate(-90)" fill="var(--fig-accent)" font-size="11">{sensor.symbol}</text>'
        out += '<path d="M0 72h40M0 89h40M0 106h40" stroke="var(--fig-accent)" stroke-opacity=".3"/></g>'
    return out + '</svg>'


# --------------------------------------------------------------------------
# The preprocessing steps, and notebook 01 that runs them end to end
# --------------------------------------------------------------------------


def draw_pipeline() -> str:
    """The eight preprocessing steps and the shape the data has after each."""
    # Each note is two short lines: on one line, neighbouring notes ran into
    # each other at the width the pages draw this figure.
    steps: tuple[tuple[str, str, tuple[str, str]], ...] = (
        ("1", "Load", ("three text files", "→ one table")),
        ("2", "Diagnose", ("which sensors", "never move")),
        ("3", "Select", ("keep 14, in", "Table 3's order")),
        ("4", "Cap", ("hold the label", "at 125")),
        ("5", "Smooth", ("backward-looking", "filter")),
        ("6", "Split", ("by engine, before", "any statistic")),
        ("7", "Rescale", ("training engines", "only")),
        ("8", "Window", ("30 cycles →", "one example")),
    )
    box_w, box_h, gap = 88.0, 82.0, 18.0
    left = (_WIDTH - (len(steps) * box_w + (len(steps) - 1) * gap)) / 2
    top = 76.0

    blocks = []
    for index, (number, name, what) in enumerate(steps):
        x = left + index * (box_w + gap)
        leak_critical = number in ("5", "6", "7")
        colour = (
            "var(--fig-warn, #c2521f)" if leak_critical
            else "var(--fig-accent, #2f7fd1)"
        )
        arrow = (
            f'<path d="M{x - gap + 3:.1f} {top + box_h / 2:.1f} h{gap - 10:.1f}" '
            f'stroke="var(--fig-axis, #8a93a0)" stroke-width="1.5" '
            f'marker-end="url(#arrow3)" />'
            if index
            else ""
        )
        blocks.append(
            arrow
            + f'<rect x="{x:.1f}" y="{top:.1f}" width="{box_w:g}" height="{box_h:g}" '
            f'rx="8" fill="{colour}" fill-opacity="0.12" stroke="{colour}" '
            f'stroke-width="1.6" />'
            + _mono(x + box_w / 2, top + 22, number, size=11, opacity=0.55)
            + _text(x + box_w / 2, top + 42, name, size=13, weight="600")
            + "".join(
                _text(x + box_w / 2, top + box_h + 20 + line * 14, words, size=10.5,
                      opacity=0.6)
                for line, words in enumerate(what)
            )
        )

    defs = (
        '<defs><marker id="arrow3" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="var(--fig-axis, #8a93a0)" />'
        "</marker></defs>"
    )

    return (
        _open(_WIDTH, top + box_h + 98, defs)
        + _text(_WIDTH / 2, 40, "The eight preprocessing steps", size=16,
                weight="600")
        + "".join(blocks)
        + _text(_WIDTH / 2, top + box_h + 76, "The three in orange are the "
                "leakage-critical ones", size=12, weight="600",
                fill="var(--fig-warn, #c2521f)")
        + "</svg>"
    )


def _spaced(value: int) -> str:
    """A count with a space every three digits, as the figures and the Learn pages print it."""
    return f"{value:,}".replace(",", " ")


def _map_band(x: float, y: float, width: float, number: str, name: str, colour: str) -> str:
    """One step of the notebook, as a band holding its number and its name."""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:g}" height="26" '
        f'rx="6" fill="{colour}" fill-opacity="0.11" stroke="{colour}" '
        f'stroke-width="1.3" />'
        + _mono(x + 16, y + 18, number, size=11, weight="600", fill=colour)
        + _text(x + 34, y + 18, name, size=12.5, anchor="start", weight="600")
    )


def _map_lines(x: float, y: float, lines: Sequence[str], colour: str) -> str:
    """The measured state a step leaves behind, one short line at a time."""
    return "".join(
        _text(
            x,
            y + index * 14.5,
            line,
            size=10.5,
            anchor="start",
            fill=colour if index == 0 else "var(--fig-ink, #1a1a1a)",
            weight="600" if index == 0 else "400",
            opacity=1.0 if index == 0 else 0.62,
        )
        for index, line in enumerate(lines)
    )


def draw_notebook_map() -> str:
    """Notebook 01 from the three raw files to the five saved ones, with the count each step leaves."""
    from ..config import PreprocessingConfig
    from ..data.pipeline import run_preprocessing
    from ..data.sensors import PAPER_SENSOR_COLUMNS, UNIT_COL

    _require_data()
    config = PreprocessingConfig()
    prepared = run_preprocessing(config, data_dir=raw_data_dir(), keep_frames=True)

    raw = prepared.raw
    assert raw is not None and prepared.frames is not None  # keep_frames=True
    splits = prepared.diagnostics["splits"]
    graph = prepared.diagnostics["graph"]
    sensors = prepared.diagnostics["sensor_diagnostics"]
    assert isinstance(splits, dict) and isinstance(graph, dict) and isinstance(sensors, dict)

    n_raw_rows = len(raw.train)
    n_raw_test_rows = len(raw.test)
    n_engines = int(raw.train[UNIT_COL].nunique())
    n_test_engines = int(raw.test[UNIT_COL].nunique())
    longest_life = int(raw.train["rul"].max())
    at_ceiling = float((raw.train["rul"] > config.rul_max).mean())

    n_sensors = sum(len(sensors[kind]) for kind in ("constant", "low_resolution", "informative"))
    # A sensor that moves but takes too few distinct values, with how many it takes.
    coarse = [
        f"{symbol} takes only {_word(int(raw.train[symbol_to_column(symbol)].nunique()))} "
        "distinct values (finding F-01)"
        for symbol in sensors["low_resolution"]
    ]

    fitting_engines = list(splits["train_engines"])
    held_back = list(splits["validation_engines"])
    columns = list(PAPER_SENSOR_COLUMNS)
    scaled = prepared.frames["train"]
    is_fitting = scaled[UNIT_COL].isin(fitting_engines)
    fitted_on = scaled[is_fitting][columns].to_numpy()
    test_scaled = prepared.frames["test"][columns].to_numpy()
    conditions = int(prepared.diagnostics["n_operating_conditions"])  # type: ignore[call-overload]

    n_train = len(prepared.train)
    n_validation = len(prepared.validation)
    n_windows = n_train + n_validation
    n_test = len(prepared.test)
    # Every engine's first window_size - 1 cycles have too little history
    # behind them to be the last cycle of a window, so they never end one.
    no_window = n_raw_rows - n_windows
    stride = f"{config.window_stride} cycle" + ("s" if config.window_stride != 1 else "")

    accent = "var(--fig-accent, #2f7fd1)"
    warn = "var(--fig-warn, #c2521f)"
    paper = "var(--fig-lpc, #6b4fbb)"

    left, lane_gap = 46.0, 20.0
    band_w = _WIDTH - 2 * left
    lane_w = (band_w - lane_gap) / 2
    right_lane = left + lane_w + lane_gap

    steps: tuple[tuple[str, str, str, list[str], list[str]], ...] = (
        (
            "1", "Load", accent,
            [
                f"{_spaced(n_raw_rows)} rows and {raw.train.shape[1]} columns, "
                f"{n_engines} engines",
                "each engine runs to failure, so life left is counted back",
                f"from its own last cycle: {longest_life} down to 0",
            ],
            [
                f"{_spaced(n_raw_test_rows)} rows and {raw.test.shape[1]} columns, "
                f"{n_test_engines} engines",
                "recording stops early, so life left is cycles to the",
                "cut-off plus the number RUL_FD001.txt gives for that engine",
            ],
        ),
        (
            "2", "Diagnose", accent,
            [
                "; ".join([
                    f"{len(sensors['constant'])} of the {n_sensors} sensors never move at all",
                    *coarse,
                ]),
            ],
            [],
        ),
        (
            "3", "Select", paper,
            [
                f"{n_sensors} sensor columns → {len(columns)}, in the order the paper's "
                "Table 3 uses",
            ],
            [],
        ),
        (
            "4", "Cap", accent,
            [
                f"labels {longest_life} … 0  →  {config.rul_max} … 0",
                f"{100 * at_ceiling:.1f} % of rows now sit at the ceiling",
            ],
            [],
        ),
        (
            "5", "Smooth", warn,
            [
                f"strength {config.smoothing_alpha}, one engine at a time, reading only "
                "backwards (assumption A-04); row counts unchanged",
            ],
            [],
        ),
        (
            "6", "Split", warn,
            [
                f"{n_engines} engines → {len(fitting_engines)} to learn from "
                f"+ {len(held_back)} held back",
            ],
            ["untouched: the test engines were never in this pool"],
        ),
        (
            "7", "Rescale", warn,
            [
                f"{len(columns)} means and {len(columns)} spreads, from the "
                f"{len(fitting_engines)} learning engines alone",
                f"values land in [{fitted_on.min():+.2f}, {fitted_on.max():+.2f}]; "
                f"FD001 has {conditions} operating condition"
                + ("s" if conditions != 1 else ""),
            ],
            [
                f"the same {len(columns)} pairs applied unchanged: "
                f"[{test_scaled.min():+.2f}, {test_scaled.max():+.2f}]",
            ],
        ),
        (
            "8", "Window", accent,
            [
                f"every {config.window_size}-cycle window, moving {stride} at a time "
                f"→ {_spaced(n_windows)}",
                f"{_spaced(no_window)} of {_spaced(n_raw_rows)} rows are too early to end one",
            ],
            [
                f"one window per engine, its last {config.window_size} cycles → {n_test}",
            ],
        ),
        # The three boxes under this band are what the split leaves, so the
        # band itself carries no lines.
        ("↳", "Apply the split", accent, [], []),
    )

    y = 116.0
    body: list[str] = []
    for number, name, colour, train_lines, test_lines in steps:
        body.append(
            f'<path d="M{_WIDTH / 2:.1f} {y - 14:.1f} v10" '
            f'stroke="var(--fig-axis, #8a93a0)" stroke-width="1.4" '
            f'marker-end="url(#mapArrow)" />'
        )
        body.append(_map_band(left, y, band_w, number, name, colour))
        top = y + 44
        rows = max(len(train_lines), len(test_lines))
        body.append(_map_lines(left + 14, top, train_lines, colour))
        if test_lines:
            body.append(_map_lines(right_lane + 14, top, test_lines, colour))
            body.append(
                f'<line x1="{right_lane - lane_gap / 2:.1f}" y1="{top - 16:.1f}" '
                f'x2="{right_lane - lane_gap / 2:.1f}" '
                f'y2="{top + 14.5 * rows - 6:.1f}" '
                f'stroke="var(--fig-edge, #c3cad4)" stroke-width="1" />'
            )
        y = top + rows * 14.5 + 16 if rows else y + 42

    # The three sets the notebook ends with.
    results = (
        ("training", n_train, accent),
        ("held back", n_validation, accent),
        ("test", n_test, warn),
    )
    box_w = (band_w - 2 * 16) / 3
    for index, (label, count, colour) in enumerate(results):
        x = left + index * (box_w + 16)
        body.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{box_w:.1f}" height="52" '
            f'rx="7" fill="{colour}" fill-opacity="0.13" stroke="{colour}" '
            f'stroke-width="1.4" />'
            + _text(x + box_w / 2, y + 21, label, size=11.5, weight="600")
            + _mono(x + box_w / 2, y + 39,
                    f"({_spaced(count)}, {graph['n_nodes']}, "
                    f"{config.window_size})", size=12,
                    weight="600", fill=colour)
        )

    # The graph, which is read from the paper rather than from the engines.
    y += 76
    reach = (
        "every sensor reachable from every other"
        if graph["is_connected"]
        else "not every sensor reachable from every other"
    )
    body.append(
        f'<rect x="{left:.1f}" y="{y:.1f}" width="{band_w:g}" height="58" '
        f'rx="8" fill="{paper}" fill-opacity="0.08" stroke="{paper}" '
        f'stroke-width="1.3" stroke-dasharray="5 4" />'
        + _text(left + 16, y + 22, "The sensor graph", size=12,
                anchor="start", weight="600", fill=paper)
        + _text(left + 16, y + 42,
                f"Table 3 of the paper, transcribed: {graph['n_nodes']} by "
                f"{graph['n_nodes']}, {reach}. It never touches the engine data.",
                size=10.5, anchor="start", opacity=0.68)
    )

    y += 84
    body.append(
        f'<path d="M{_WIDTH / 2:.1f} {y - 22:.1f} v12" '
        f'stroke="var(--fig-axis, #8a93a0)" stroke-width="1.4" '
        f'marker-end="url(#mapArrow)" />'
        + _text(_WIDTH / 2, y + 4, "Every claim above re-checked, then saved to "
                "data/processed/", size=12, weight="600")
        + _mono(_WIDTH / 2, y + 24,
                "windows.npz · adjacency.npz · normalization_statistics.csv "
                "· config.yaml · diagnostics.json", size=10, opacity=0.6)
    )

    height = y + 44
    defs = (
        '<defs><marker id="mapArrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="var(--fig-axis, #8a93a0)" />'
        "</marker></defs>"
    )

    header = (
        _text(_WIDTH / 2, 40, "What notebook 01 does", size=16, weight="600")
        + _text(left + 14, 72, "train_FD001.txt", size=11.5, anchor="start",
                weight="600", fill=accent)
        + _text(right_lane + 14, 72, "test_FD001.txt  +  RUL_FD001.txt",
                size=11.5, anchor="start", weight="600", fill=warn)
        + _text(_WIDTH / 2, 94, "All eight steps are section 4 of the notebook.",
                size=10.5, opacity=0.55)
    )

    return _open(_WIDTH, height, defs) + header + "".join(body) + "</svg>"


# --------------------------------------------------------------------------
# Two of the paper's own figures, drawn again: Fig. 5 recomputed from the
# FD001 training file, and Fig. 2's LSTM cell redrawn
# --------------------------------------------------------------------------


def correlation_table() -> tuple[list[str], np.ndarray]:
    """The correlation between every pair of columns of FD001's training file, laid out as the paper's Fig. 5."""
    from ..data.loading import load_subset
    from ..data.sensors import CYCLE_COL, OP_SETTING_COLS, UNIT_COL

    _require_data()
    frame = load_subset("FD001", data_dir=raw_data_dir()).train
    # The paper's figure keeps the two operating settings that vary in FD001
    # and drops the third with the sensors that never change.
    settings = [column for column in OP_SETTING_COLS if frame[column].nunique() > 1][:2]
    kept = [UNIT_COL, CYCLE_COL, *settings]
    labels = [*kept, *PAPER_SENSOR_SYMBOLS]
    columns = [*kept, *(symbol_to_column(symbol) for symbol in PAPER_SENSOR_SYMBOLS)]
    matrix = frame[columns].corr(method="pearson").to_numpy(dtype=float)
    return labels, matrix


def _diverging(value: float) -> str:
    """White at zero, deepening to red above and blue below, like the paper's Fig. 5."""
    strength = min(1.0, abs(float(value)))
    target = (178, 24, 43) if value >= 0 else (33, 102, 172)
    red, green, blue = (round(255 - (255 - channel) * strength) for channel in target)
    return f"rgb({red},{green},{blue})"


def draw_correlation() -> str:
    """The paper's Fig. 5, recomputed: eighteen columns of FD001 against each other."""
    labels, matrix = correlation_table()
    count = len(labels)
    cell, left, top = 34.0, 112.0, 24.0
    size = cell * count
    bar_x = left + size + 26
    total_height = top + size + 64

    out = _open(_WIDTH, total_height) + (
        '<defs><linearGradient id="corr-scale" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="rgb(178,24,43)"/>'
        '<stop offset="0.5" stop-color="rgb(255,255,255)"/>'
        '<stop offset="1" stop-color="rgb(33,102,172)"/></linearGradient></defs>'
    )
    for row, row_label in enumerate(labels):
        y = top + row * cell
        out += _text(left - 8, y + cell / 2 + 3.5, row_label, size=10, anchor="end")
        for column in range(count):
            value = float(matrix[row, column])
            x = left + column * cell
            out += (
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell}" height="{cell}" '
                f'fill="{_diverging(value)}" stroke="var(--fig-bg, #ffffff)" stroke-width="1">'
                f"<title>{html.escape(row_label)} against {html.escape(labels[column])}: "
                f"{value:.3f}</title></rect>"
                + _mono(x + cell / 2, y + cell / 2 + 3, f"{value:.2f}", size=7.5,
                        fill="#ffffff" if abs(value) > 0.55 else "#16191d")
            )
    for column, column_label in enumerate(labels):
        x = left + column * cell + cell / 2
        out += (
            f'<g transform="translate({x:.1f},{top + size + 10:.1f}) rotate(-50)">'
            + _text(0, 0, column_label, size=10, anchor="end")
            + "</g>"
        )
    out += (
        f'<rect x="{bar_x}" y="{top}" width="16" height="{size}" fill="url(#corr-scale)" '
        f'stroke="var(--fig-edge, #c3cad4)" stroke-width="1"/>'
    )
    for tick in (1.0, 0.5, 0.0, -0.5, -1.0):
        y = top + size * (1 - tick) / 2
        out += _mono(bar_x + 22, y + 3.5, f"{tick:+.2f}", size=9, anchor="start", opacity=0.7)
    return out + "</svg>"


def draw_lstm_cell() -> str:
    """One LSTM cell, as the paper's Fig. 2 draws it and its Equations 1 to 6 describe it."""
    ink = "var(--fig-ink, #16191d)"
    wire_colour = "var(--fig-axis, #8a93a0)"
    accent = "var(--fig-accent, #1f5fa8)"
    memory = "var(--fig-lpc, #6b4fbb)"
    green = "var(--fig-lpt, #1c7a5e)"
    warm = "var(--fig-warn, #c2521f)"

    def wire(x1: float, y1: float, x2: float, y2: float, colour: str = wire_colour,
             arrow: bool = False, width: float = 1.7) -> str:
        head = ' marker-end="url(#lstm-head)"' if arrow else ""
        return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colour}" '
                f'stroke-width="{width}"{head} />')

    def gate(cx: float, cy: float, symbol: str, label: str, colour: str) -> str:
        # The name sits inside the box, so the wire arriving from below never
        # runs through it.
        return (
            f'<rect x="{cx - 43}" y="{cy - 23}" width="86" height="46" rx="8" '
            f'fill="{colour}" fill-opacity="0.14" stroke="{colour}" stroke-width="1.6" />'
            + _text(cx, cy - 2, symbol, size=12.5, weight="700", fill=colour)
            + _text(cx, cy + 14, label, size=9, opacity=0.75)
        )

    def op(cx: float, cy: float, symbol: str) -> str:
        return (
            f'<circle cx="{cx}" cy="{cy}" r="13" fill="var(--fig-bg, #ffffff)" '
            f'stroke="{ink}" stroke-width="1.6" />'
            + _text(cx, cy + 5.5, symbol, size=15, weight="700")
        )

    out = _open(_WIDTH, 476) + (
        '<defs><marker id="lstm-head" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{wire_colour}"/></marker></defs>'
    )
    # The cell itself.
    out += (f'<rect x="150" y="98" width="570" height="322" rx="18" fill="{accent}" '
            f'fill-opacity="0.05" stroke="{accent}" stroke-width="1.6" />'
            + _text(435, 84, "one LSTM cell, at one cycle t", size=12.5, weight="600",
                    fill=accent))
    # The memory line along the top, from the last cycle to this one.
    out += (
        _text(52, 154, "C(t-1)", size=13, weight="700", fill=memory)
        + wire(88, 150, 237, 150, memory, width=2.4)
        + op(250, 150, "⊗")
        + wire(263, 150, 372, 150, memory, width=2.4)
        + op(385, 150, "⊕")
        + wire(398, 150, 790, 150, memory, arrow=True, width=2.4)
        + _text(818, 154, "C(t)", size=13, weight="700", fill=memory)
        + _text(385, 126, "new memory (4)", size=10.5, opacity=0.7)
    )
    # The three gates and the candidate, fed from the joined input below.
    out += (
        gate(250, 300, "sigmoid", "forget gate (1)", warm)
        + gate(340, 300, "sigmoid", "input gate (2)", green)
        + gate(430, 300, "tanh", "candidate C̃ (3)", green)
        + gate(620, 300, "sigmoid", "output gate (5)", accent)
        + wire(250, 277, 250, 165, warm, arrow=True)
        + wire(340, 277, 377, 233, green)
        + wire(430, 277, 393, 233, green)
        + op(385, 222, "⊗")
        + wire(385, 209, 385, 165, green, arrow=True)
    )
    # The output: the new memory squashed, then scaled by the output gate.
    out += (
        wire(540, 150, 540, 205, memory, arrow=True)
        + f'<rect x="512" y="207" width="56" height="30" rx="7" fill="{memory}" '
        f'fill-opacity="0.14" stroke="{memory}" stroke-width="1.6" />'
        + _text(540, 227, "tanh", size=13, weight="700", fill=memory)
        + wire(568, 222, 605, 222, memory)
        + op(620, 222, "⊗")
        + wire(620, 277, 620, 237, accent, arrow=True)
        + wire(633, 222, 790, 222, accent, arrow=True, width=2.2)
        + _text(818, 226, "h(t)", size=13, weight="700", fill=accent)
        + _text(700, 210, "output (6)", size=10.5, opacity=0.7)
    )
    # The joined input: the last output and this cycle's reading.
    out += (
        _text(52, 354, "h(t-1)", size=13, weight="700", fill=accent)
        + _text(52, 396, "x(t)", size=13, weight="700")
        + wire(88, 350, 160, 350, accent, width=2)
        + wire(88, 392, 160, 392, ink, width=2)
        + wire(160, 392, 160, 350, ink, width=2)
        + wire(160, 350, 640, 350, ink, width=2)
        + "".join(wire(x, 350, x, 325, ink, arrow=True) for x in (250, 340, 430, 620))
        + _text(400, 372, "[h(t-1), x(t)]: the last output and this cycle's reading, joined",
                size=10.5, opacity=0.7)
    )
    # What the two operation symbols and the bracketed numbers mean.
    out += _text(_WIDTH / 2, 456,
                 "⊗ multiplies element by element; ⊕ adds. Equation numbers are the paper's.",
                 size=11.5, opacity=0.7)
    return out + "</svg>"


# --------------------------------------------------------------------------
# Every figure by name, and drawing one for a page or a notebook
# --------------------------------------------------------------------------


FIGURES: tuple[Figure, ...] = (
    Figure("engine", draw_engine),
    Figure("rul-curve", draw_rul_curve, needs_data=True),
    Figure("worked-example", draw_worked_example, needs_data=True, needs_torch=True),
    Figure("sensor-selection", draw_sensor_selection),
    Figure("correlation", draw_correlation, needs_data=True),
    Figure("smoothing", draw_smoothing, needs_data=True),
    Figure("windowing", draw_windowing, needs_data=True),
    Figure("pipeline", draw_pipeline),
    Figure("notebook-map", draw_notebook_map, needs_data=True),
    Figure("adjacency", draw_adjacency_matrix),
    Figure("message-passing", draw_message_passing),
    Figure("reach", draw_reach),
    Figure("architecture", draw_architecture, needs_torch=True),
    Figure("architecture-overview", draw_architecture_overview, needs_torch=True),
    Figure("lstm-cell", draw_lstm_cell),
    Figure("shapley", draw_shapley),
)

_BY_NAME = {figure.name: figure for figure in FIGURES}


def figure_names() -> tuple[str, ...]:
    """Every figure this website can draw, in the order they are introduced."""
    return tuple(figure.name for figure in FIGURES)


def _notice(kind: str, title: str, body: str) -> str:
    """A block explaining why a figure could not be drawn, instead of a figure."""
    return (
        f'<div class="figure-notice figure-notice--{kind}">'
        f"<strong>{html.escape(title)}</strong><p>{html.escape(body)}</p></div>"
    )


def _torch_is_missing(error: ImportError) -> bool:
    """Whether the import that failed was of PyTorch or one of its parts."""
    name = error.name or ""
    return name == "torch" or name.startswith("torch.")


def _failed(error: Exception) -> str:
    """The notice for a figure that broke, naming the error rather than guessing a cause."""
    return _notice(
        "error",
        "This figure could not be drawn",
        f"{error.__class__.__name__}: {error}",
    )


def render_figure(name: str) -> str:
    """Draw one figure by name, or explain why it cannot be drawn."""
    figure = _BY_NAME.get(name)
    if figure is None:
        return _notice(
            "error",
            "No such figure",
            f"{name!r} is not one of: {', '.join(figure_names())}.",
        )
    try:
        return figure.draw()
    except DataMissing as missing:
        return _notice("data", "The NASA files are needed for this figure", str(missing))
    except ImportError as error:
        # PyTorch is the only optional library a figure imports, so a failed
        # import of PyTorch gets the notice saying what to install. The check
        # reads only the name of the module that failed, so an installed
        # PyTorch asked for a name it does not have gets that notice too. Any
        # other failed import, such as a module of this repository renamed
        # without the figure that imports it, is shown as the error it is.
        if not _torch_is_missing(error):
            return _failed(error)
        return _notice(
            "torch",
            "PyTorch is needed for this figure",
            "This figure builds or loads the project's network, which is "
            "written with PyTorch, and reads real numbers off it. Install it "
            "with 'pip install torch' and reload this page.",
        )
    except Exception as error:  # pragma: no cover - defensive
        return _failed(error)
