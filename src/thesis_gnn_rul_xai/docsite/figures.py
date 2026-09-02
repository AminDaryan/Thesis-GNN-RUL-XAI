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
from ..utils.paths import project_root, raw_data_dir

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
    """One diagram: how to draw it, what it shows, and what it needs first."""

    name: str
    title: str
    caption: str
    draw: Callable[[], str]
    needs_data: bool = False
    needs_torch: bool = False


# The drawing area every figure is laid out in, before the browser scales it.
_WIDTH = 860.0
_HEIGHT = 520.0

# The eight sensors of the engine core and the four of the bypass side, used
# only to colour the graph figure. The grouping is read from the station field
# of the sensor table in data/sensors/, never invented here.
_STATION_HUE: dict[str, str] = {
    "Fan inlet": "var(--fig-fan, #2f7fd1)",
    "Bypass duct": "var(--fig-fan, #2f7fd1)",
    "Fan shaft": "var(--fig-fan, #2f7fd1)",
    "LPC outlet": "var(--fig-lpc, #6b4fbb)",
    "HPC outlet": "var(--fig-hpc, #c2521f)",
    "Combustor": "var(--fig-burn, #b3261e)",
    "Core shaft": "var(--fig-core, #8a6d00)",
    "LPT outlet": "var(--fig-lpt, #1c7a5e)",
    "HPT cooling": "var(--fig-cool, #0f6b86)",
    "LPT cooling": "var(--fig-cool, #0f6b86)",
    "Bleed system": "var(--fig-bleed, #6a6a6a)",
}

_STATION_OF = {sensor.symbol: sensor.station for sensor in SENSORS}
_DESCRIPTION_OF = {sensor.symbol: sensor.description for sensor in SENSORS}


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
) -> str:
    """One piece of text, escaped, with the settings the figures keep reusing."""
    style = f"font-family:{family};" if family != "inherit" else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-size="{size:g}" font-weight="{weight}" fill="{fill}" '
        f'opacity="{opacity:g}" style="{style}">{html.escape(body)}</text>'
    )


def _mono(x: float, y: float, body: str, **kwargs: object) -> str:
    """A piece of text in the fixed-width face, for shapes and numbers."""
    kwargs.setdefault("family", "var(--mono)")
    return _text(x, y, body, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The sensor graph
# --------------------------------------------------------------------------


def _graph_layout() -> tuple[nx.Graph, dict[str, tuple[float, float]], dict[str, int]]:
    """Place the 14 real sensors on the canvas, and report their degrees."""
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


def draw_sensor_graph() -> str:
    """The paper's Table 3, drawn: 14 sensors, 41 edges, one articulation point."""
    graph, pos, degrees = _graph_layout()
    report = adjacency_report(*build_physics_adjacency())
    articulation = set(report["articulation_points"])  # type: ignore[arg-type]

    edges = "".join(
        f'<line x1="{pos[a][0]:.1f}" y1="{pos[a][1]:.1f}" '
        f'x2="{pos[b][0]:.1f}" y2="{pos[b][1]:.1f}" '
        f'stroke="var(--fig-edge, #b9c0cb)" stroke-width="1.6" />'
        for a, b in graph.edges()
    )

    nodes = []
    for node, (x, y) in pos.items():
        radius = _node_radius(degrees[node])
        colour = _STATION_HUE.get(_STATION_OF[node], "var(--fig-accent, #2f7fd1)")
        ring = (
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius + 5:.1f}" fill="none" '
            f'stroke="var(--fig-warn, #c2521f)" stroke-width="2.4" '
            f'stroke-dasharray="4 3" />'
            if node in articulation
            else ""
        )
        nodes.append(
            f"<g>{ring}"
            f'<title>{html.escape(node)} — {html.escape(_DESCRIPTION_OF[node])}'
            f" ({degrees[node]} neighbours)</title>"
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{colour}" '
            f'fill-opacity="0.9" stroke="var(--fig-bg, #ffffff)" stroke-width="2" />'
            + _mono(x, y + 4, node, size=12, weight="600", fill="#ffffff")
            + "</g>"
        )

    legend = (
        _text(
            _WIDTH / 2,
            _HEIGHT - 26,
            f"{report['n_nodes']} sensors · {report['n_edges']} edges · "
            f"diameter {report['diameter']} · "
            f"T24 is the only articulation point",
            size=13,
            opacity=0.72,
        )
        + _text(
            _WIDTH / 2,
            _HEIGHT - 8,
            "Node size is the number of neighbours. Colour is the engine station. "
            "Hover a node for its name.",
            size=11.5,
            opacity=0.55,
        )
    )
    return _open() + edges + "".join(nodes) + legend + "</svg>"


def draw_message_passing() -> str:
    """Three layers of message passing, animated over the real sensor graph."""
    graph, pos, degrees = _graph_layout()
    source = "P15"
    distance = nx.single_source_shortest_path_length(graph, source)
    unreachable = [
        node for node in graph.nodes() if distance.get(node, 99) > 3
    ]

    # One second per layer, then a pause, then the whole thing again.
    step, pause = 1.6, 2.2
    cycle = step * 4 + pause

    edges = []
    for a, b in graph.edges():
        near = min(distance.get(a, 99), distance.get(b, 99))
        far = max(distance.get(a, 99), distance.get(b, 99))
        carries = far <= 3 and far == near + 1
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
        reached = hops <= 3
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
            else _mono(x, y + radius + 15, "4 hops — never", size=10,
                       fill="var(--fig-warn, #c2521f)", opacity=0.85)
        )
        nodes.append(
            f"<g><circle cx='{x:.1f}' cy='{y:.1f}' r='{radius:.1f}' fill='{colour}' "
            f"fill-opacity='{0.95 if hops == 0 else 0.12:g}' "
            f"stroke='{colour}' stroke-width='2'>{fill_animation}</circle>"
            + _mono(x, y + 4, node, size=12, weight="600",
                    fill="var(--fig-ink, #1a1a1a)")
            + badge
            + "</g>"
        )

    caption = _text(
        _WIDTH / 2,
        _HEIGHT - 22,
        f"Information leaving {source} after one, two and three graph layers. "
        f"{', '.join(unreachable)} is never reached.",
        size=13,
        opacity=0.72,
    ) + _text(
        _WIDTH / 2,
        _HEIGHT - 4,
        "Three layers reach three edges away — no further. This is measured by "
        "tests/test_models.py::test_three_layers_cannot_reach_four_hops.",
        size=11.5,
        opacity=0.55,
    )
    return _open() + "".join(edges) + "".join(nodes) + caption + "</svg>"


def draw_adjacency_matrix() -> str:
    """The 14 by 14 table itself, before and after the row normalisation."""
    matrix, symbols = build_physics_adjacency()
    normalized = normalize_adjacency(matrix, method="row")

    cell, left, top = 21.0, 74.0, 96.0
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
            _mono(offset + i * cell + cell / 2 - 0.75, top - 8, symbols[i],
                  size=8.5, opacity=0.7)
            for i in range(14)
        ) + "".join(
            _mono(offset - 6, top + i * cell + cell / 2 + 3, symbols[i],
                  size=8.5, anchor="end", opacity=0.7)
            for i in range(14)
        )
        panels.append(
            _text(offset + 14 * cell / 2, top - 34, title, size=13, weight="600")
            + "".join(squares)
            + labels
        )

    caption = _text(
        _WIDTH / 2,
        _HEIGHT - 14,
        "Left: 41 edges, a zero diagonal, symmetric. Right: self-loops added, "
        "then each row divided by its own degree, so every row sums to 1.",
        size=12,
        opacity=0.68,
    )
    return _open(880, 470) + "".join(panels) + caption + "</svg>"


def draw_degrees() -> str:
    """How many neighbours each sensor has, as a bar for each of the 14."""
    _, _, degrees = _graph_layout()
    order = list(PAPER_SENSOR_SYMBOLS)
    top, height, base = 70.0, 300.0, 380.0
    width, gap = 40.0, 16.0
    left = (_WIDTH - (len(order) * width + (len(order) - 1) * gap)) / 2
    largest = max(degrees.values())

    bars = []
    for index, symbol in enumerate(order):
        degree = degrees[symbol]
        bar_height = height * degree / largest
        x = left + index * (width + gap)
        colour = _STATION_HUE.get(_STATION_OF[symbol], "var(--fig-accent, #2f7fd1)")
        bars.append(
            f'<rect x="{x:.1f}" y="{base - bar_height:.1f}" width="{width:g}" '
            f'height="{bar_height:.1f}" rx="4" fill="{colour}" fill-opacity="0.85">'
            f"<title>{symbol}: {degree} neighbours</title></rect>"
            + _mono(x + width / 2, base - bar_height - 8, str(degree), size=12,
                    weight="600")
            + _mono(x + width / 2, base + 18, symbol, size=10.5, opacity=0.8)
            + _text(x + width / 2, base + 34, _STATION_OF[symbol], size=8,
                    opacity=0.45)
        )

    caption = _text(
        _WIDTH / 2,
        top - 34,
        "Neighbours per sensor in the paper's graph",
        size=14,
        weight="600",
    ) + _text(
        _WIDTH / 2,
        base + 66,
        "T30, Nc and W31 are the best connected, with 8 neighbours each. "
        "The four bypass-side sensors have 4.",
        size=12,
        opacity=0.68,
    )
    return _open(_WIDTH, 470) + caption + "".join(bars) + "</svg>"


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
    to_x = lambda c: left + width * c / max(total - 1, 1)  # noqa: E731
    to_y = lambda r: top + height * (1 - r / rul_max)  # noqa: E731

    points = " ".join(f"{to_x(i):.1f},{to_y(float(v)):.1f}" for i, v in enumerate(curve))
    knee = int(np.argmax(curve < rul_max))

    naive = (
        f'<polyline points="{to_x(0):.1f},{to_y(min(total, rul_max)):.1f} '
        f'{to_x(total - 1):.1f},{to_y(0):.1f}" fill="none" '
        f'stroke="var(--fig-muted, #9aa3b0)" stroke-width="1.6" '
        f'stroke-dasharray="6 4" />'
    )

    ticks = "".join(
        _mono(left - 10, to_y(value) + 4, str(value), size=10, anchor="end", opacity=0.6)
        + f'<line x1="{left - 5}" y1="{to_y(value):.1f}" x2="{left}" '
        f'y2="{to_y(value):.1f}" stroke="var(--fig-axis, #8a93a0)" />'
        for value in (0, 25, 50, 75, 100, 125)
    ) + "".join(
        _mono(to_x(value), top + height + 20, str(value), size=10, opacity=0.6)
        for value in (0, 50, 100, 150, total - 1)
        if value < total
    )

    return (
        _open(_WIDTH, 470)
        + _axes(left, top, width, height, "flight cycle", "label (cycles left)")
        + ticks
        + naive
        + f'<line x1="{to_x(knee):.1f}" y1="{top}" x2="{to_x(knee):.1f}" '
        f'y2="{top + height}" stroke="var(--fig-warn, #c2521f)" '
        f'stroke-width="1.2" stroke-dasharray="3 3" opacity="0.7" />'
        + f'<polyline points="{points}" fill="none" '
        f'stroke="var(--fig-accent, #2f7fd1)" stroke-width="2.6" '
        f'stroke-linejoin="round" />'
        + _text(to_x(knee / 2), to_y(rul_max) - 14, "held at the ceiling", size=12,
                fill="var(--fig-accent, #2f7fd1)", weight="600")
        + _text(to_x(knee) + 8, top + 22, f"cycle {knee}", size=11,
                anchor="start", fill="var(--fig-warn, #c2521f)")
        + _text(to_x(total * 0.82), to_y(rul_max * 0.62), "falls by one per cycle",
                size=12, fill="var(--fig-accent, #2f7fd1)", weight="600")
        + _text(to_x(total * 0.30), to_y(rul_max * 0.80), "the dashed line is the",
                size=11, opacity=0.6)
        + _text(to_x(total * 0.30), to_y(rul_max * 0.80) + 14,
                "uncapped alternative", size=11, opacity=0.6)
        + _text(
            _WIDTH / 2,
            _HEIGHT - 62,
            f"Engine 1 of FD001, which ran {total} cycles. "
            f"The ceiling is 125 (assumption A-01), so the label is flat for "
            f"the first {knee} cycles.",
            size=12,
            opacity=0.7,
        )
        + "</svg>"
    )


def draw_smoothing() -> str:
    """One real sensor trace, before and after the noise filter of step 5."""
    from ..data.smoothing import ewma_series

    frame, total = _one_engine(1)
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
        _open(_WIDTH, 470)
        + _axes(left, top, width, height, "flight cycle", "T50 (degrees Rankine)")
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
            _HEIGHT - 62,
            f"Engine 1, sensor T50, all {total} cycles. Average step between "
            f"neighbouring cycles: {jitter_before:.3f} before, "
            f"{jitter_after:.3f} after — {removed:.0f} % of the jitter removed.",
            size=12,
            opacity=0.7,
        )
        + _text(
            _WIDTH / 2,
            _HEIGHT - 44,
            "The filter only ever looks backwards, so it cannot leak a future "
            "reading into the present.",
            size=11.5,
            opacity=0.55,
        )
        + "</svg>"
    )


def draw_windowing() -> str:
    """The sliding window of step 8, animated over a real engine's history."""
    frame, total = _one_engine(1)
    column = symbol_to_column("T50")
    values = frame[column].to_numpy(dtype=float)
    window = 30
    n_windows = max(len(values) - window + 1, 0)

    left, top, width, height = 90.0, 78.0, 700.0, 230.0
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
        _open(_WIDTH, 470)
        + _axes(left, top, width, height, "flight cycle", "T50")
        + f'<polyline points="{trace}" fill="none" '
        f'stroke="var(--fig-muted, #9aa3b0)" stroke-width="1.6" />'
        + box
        + marker
        + _text(_WIDTH / 2, 40, "One example is 30 consecutive cycles", size=14,
                weight="600")
        + _text(left + window_width / 2, top + height + 62,
                "the window the model receives", size=11.5,
                fill="var(--fig-accent, #2f7fd1)")
        + _text(
            _WIDTH / 2,
            _HEIGHT - 62,
            f"Engine 1 ran {total} cycles, so a 30-cycle window moving one cycle "
            f"at a time (assumption A-05) yields {n_windows} training examples "
            f"from this engine alone.",
            size=12,
            opacity=0.7,
        )
        + _text(
            _WIDTH / 2,
            _HEIGHT - 44,
            "The orange line is the last cycle of the window — the one number "
            "each sensor node actually carries (assumption A-07).",
            size=11.5,
            opacity=0.55,
        )
        + "</svg>"
    )


# --------------------------------------------------------------------------
# The network and the explanation
# --------------------------------------------------------------------------


def draw_architecture() -> str:
    """Every layer of the network, with the parameter counts it really has."""
    from ..models.estgnn import ESTGNN

    matrix, _ = build_physics_adjacency()
    model = ESTGNN(normalize_adjacency(matrix, method="row"), in_features=1)
    rows = model.layer_summary()
    total = model.count_parameters()

    box_width, box_height, gap = 96.0, 96.0, 24.0
    left = (_WIDTH - (len(rows) * box_width + (len(rows) - 1) * gap)) / 2
    top = 150.0

    blocks = []
    for index, row in enumerate(rows):
        x = left + index * (box_width + gap)
        is_graph = str(row["type"]) == "GCN"
        colour = (
            "var(--fig-accent, #2f7fd1)" if is_graph else "var(--fig-lpt, #1c7a5e)"
        )
        if str(row["layer"]) in ("Input", "Flatten"):
            colour = "var(--fig-muted, #9aa3b0)"
        arrow = (
            f'<path d="M{x - gap + 4:.1f} {top + box_height / 2:.1f} '
            f'h{gap - 12:.1f}" stroke="var(--fig-axis, #8a93a0)" stroke-width="1.6" '
            f'marker-end="url(#arrow)" />'
            if index
            else ""
        )
        blocks.append(
            arrow
            + f'<rect x="{x:.1f}" y="{top:.1f}" width="{box_width:g}" '
            f'height="{box_height:g}" rx="8" fill="{colour}" fill-opacity="0.14" '
            f'stroke="{colour}" stroke-width="1.8" />'
            + _text(x + box_width / 2, top + 26, str(row["layer"]), size=13,
                    weight="600")
            + _text(x + box_width / 2, top + 45, str(row["activation"]), size=10.5,
                    opacity=0.65)
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
        _open(_WIDTH, 430, defs)
        + _text(_WIDTH / 2, 48, "The network of the paper's Table 4", size=16,
                weight="600")
        + _text(_WIDTH / 2, 72, "Each box shows what comes out of it, and how "
                "many numbers it learns", size=12, opacity=0.65)
        + "".join(blocks)
        + _text(
            _WIDTH / 2,
            352,
            f"{total:,} trainable parameters in total".replace(",", " ")
            + " — the count the paper's Table 4 prints, checked by a test.",
            size=13,
            weight="600",
        )
        + _text(
            _WIDTH / 2,
            376,
            "Built by this page just now, from "
            "src/thesis_gnn_rul_xai/models/estgnn.py. The per-sensor LSTM the "
            "paper puts in front is deliberately absent (assumption A-11).",
            size=11.5,
            opacity=0.6,
        )
        + "</svg>"
    )


def draw_shapley() -> str:
    """What a Shapley value averages over, drawn for a three-player example."""
    players = ("A", "B", "C")
    orders = list(itertools.permutations(players))

    left, top, row_height = 150.0, 118.0, 44.0
    cell, gap = 56.0, 12.0

    rows = []
    for index, order in enumerate(orders):
        y = top + index * row_height
        rows.append(_mono(left - 22, y + 26, f"#{index + 1}", size=11,
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
        _open(_WIDTH, 460)
        + _text(_WIDTH / 2, 46, "A Shapley value is an average over every "
                "joining order", size=16, weight="600")
        + _text(_WIDTH / 2, 72, "Three players have 6 orders. B's value is how "
                "much it adds, averaged over all of them.", size=12, opacity=0.65)
        + _text(_WIDTH / 2, 96, "first · second · third", size=10.5, opacity=0.45)
        + "".join(rows)
        + _text(
            _WIDTH / 2,
            400,
            "This project has 14 sensors, so 14! = 87 178 291 200 orders — but "
            "only 2¹⁴ = 16 384 distinct groups.",
            size=12.5,
            opacity=0.75,
        )
        + _text(
            _WIDTH / 2,
            424,
            "explain/shap.py evaluates all 16 384 and weights them, so the "
            "attributions are exact rather than sampled.",
            size=11.5,
            opacity=0.55,
        )
        + "</svg>"
    )


def draw_engine() -> str:
    """A turbofan drawn to place the sensors, with each station named."""
    stations: tuple[tuple[float, str, tuple[str, ...]], ...] = (
        (120, "Fan inlet", ("T2", "P2")),
        (232, "LPC outlet", ("T24",)),
        (348, "HPC outlet", ("T30", "P30", "Ps30")),
        (452, "Combustor", ("phi", "farB")),
        (556, "HPT cooling", ("W31",)),
        (664, "LPT outlet", ("T50",)),
        (760, "LPT cooling", ("W32",)),
    )
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
        _open(_WIDTH, 470, defs)
        + _text(_WIDTH / 2, 34, "Where the sensors sit", size=16, weight="600")
        + _text(_WIDTH / 2, 56, "Air enters at the left, is compressed, is burned, "
                "and drives the turbines on its way out", size=12, opacity=0.65)
        + body
        + flow
        + "".join(marks)
        + _text(_WIDTH / 2, 440, "Bold names are among the 14 the paper keeps; "
                "faded ones are dropped. Shaft-speed and bypass sensors "
                "(Nf, Nc, NRf, BPR, P15, htBleed) are not drawn — they measure "
                "the whole machine rather than one station.", size=11,
                opacity=0.55)
        + _text(_WIDTH / 2, 458, "Schematic, drawn for this page. It is not a "
                "figure from the paper and is not to scale.", size=10.5,
                opacity=0.42)
        + "</svg>"
    )


def draw_pipeline() -> str:
    """The eight preprocessing steps and the shape the data has after each."""
    steps: tuple[tuple[str, str, str], ...] = (
        ("1", "Load", "three text files → one table"),
        ("2", "Diagnose", "which sensors never move"),
        ("3", "Select", "keep 14, in Table 3's order"),
        ("4", "Cap", "hold the label at 125"),
        ("5", "Smooth", "backward-looking filter"),
        ("6", "Split", "by engine, before any statistic"),
        ("7", "Rescale", "training engines only"),
        ("8", "Window", "30 cycles → one example"),
    )
    box_w, box_h, gap = 88.0, 82.0, 18.0
    left = (_WIDTH - (len(steps) * box_w + (len(steps) - 1) * gap)) / 2
    top = 150.0

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
            + _text(x + box_w / 2, top + box_h + 20, what, size=9.5, opacity=0.6)
        )

    defs = (
        '<defs><marker id="arrow3" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="var(--fig-axis, #8a93a0)" />'
        "</marker></defs>"
    )

    return (
        _open(_WIDTH, 400, defs)
        + _text(_WIDTH / 2, 48, "The eight preprocessing steps", size=16,
                weight="600")
        + _text(_WIDTH / 2, 72, "Their order is not free: three of them decide "
                "whether the experiment is valid at all", size=12, opacity=0.65)
        + "".join(blocks)
        + _text(_WIDTH / 2, 314, "The three in orange are the leakage-critical "
                "ones", size=12, weight="600", fill="var(--fig-warn, #c2521f)")
        + _text(_WIDTH / 2, 336, "The filter may only look backwards; the split "
                "happens before any statistic is learned; the scaling is fitted "
                "on training engines alone.", size=11.5, opacity=0.6)
        + _text(_WIDTH / 2, 358, "Each is guarded by its own test — see "
                "tests/test_pipeline.py.", size=11, opacity=0.45)
        + "</svg>"
    )


def draw_reach() -> str:
    """How far three layers reach, as the real distance histogram of the graph."""
    matrix, symbols = build_physics_adjacency()
    report = adjacency_report(matrix, symbols)
    histogram: dict[int, int] = dict(report["distance_histogram"])  # type: ignore[arg-type]
    beyond = pairs_beyond_reach(matrix, symbols, 3)

    total = sum(histogram.values())
    left, base, height = 180.0, 340.0, 220.0
    bar_w, gap = 92.0, 40.0

    bars = []
    for index, (distance, count) in enumerate(sorted(histogram.items())):
        x = left + index * (bar_w + gap)
        bar_h = height * count / max(histogram.values())
        reachable = distance <= 3
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
                    "reachable" if reachable else "out of reach", size=10,
                    weight="600", fill=colour)
        )

    names = ", ".join(f"{a} and {b}" for a, b, _ in beyond)
    return (
        _open(_WIDTH, 460)
        + _text(_WIDTH / 2, 50, "How far apart the 91 sensor pairs are", size=16,
                weight="600")
        + _text(_WIDTH / 2, 74, "Three graph layers carry information three "
                "edges. Anything further never mixes.", size=12, opacity=0.65)
        + "".join(bars)
        + _text(_WIDTH / 2, 430, f"The four pairs beyond reach: {names}.",
                size=12, opacity=0.72)
        + _text(_WIDTH / 2, 450, "Computed here by "
                "graph/adjacency.py::pairs_beyond_reach.", size=10.5, opacity=0.45)
        + "</svg>"
    )


FIGURES: tuple[Figure, ...] = (
    Figure(
        "engine",
        "Where the sensors sit",
        "A turbofan, and the station each measurement is taken at.",
        draw_engine,
    ),
    Figure(
        "rul-curve",
        "The piecewise-linear label",
        "What the model is asked to predict, on a real engine's true length.",
        draw_rul_curve,
        needs_data=True,
    ),
    Figure(
        "smoothing",
        "The noise filter",
        "One real sensor trace before and after the filter of step 5.",
        draw_smoothing,
        needs_data=True,
    ),
    Figure(
        "windowing",
        "The sliding window",
        "How one engine's history becomes many fixed-size examples.",
        draw_windowing,
        needs_data=True,
    ),
    Figure(
        "pipeline",
        "The eight preprocessing steps",
        "The order of the chain, and the three steps that decide its validity.",
        draw_pipeline,
    ),
    Figure(
        "sensor-graph",
        "The sensor graph",
        "The paper's Table 3, drawn as the network it describes.",
        draw_sensor_graph,
    ),
    Figure(
        "adjacency",
        "The adjacency matrix",
        "The same graph as a table, raw and after normalisation.",
        draw_adjacency_matrix,
    ),
    Figure(
        "degrees",
        "Neighbours per sensor",
        "Which sensors the paper's graph considers central.",
        draw_degrees,
    ),
    Figure(
        "message-passing",
        "Message passing, three layers deep",
        "Information spreading from one sensor, and where it stops.",
        draw_message_passing,
    ),
    Figure(
        "reach",
        "What three layers can reach",
        "The distance between every pair of sensors, and the four out of reach.",
        draw_reach,
    ),
    Figure(
        "architecture",
        "The network",
        "Every layer of Table 4, with the parameter counts it really has.",
        draw_architecture,
        needs_torch=True,
    ),
    Figure(
        "shapley",
        "What a Shapley value averages over",
        "Every order three players could join in, and where B lands in each.",
        draw_shapley,
    ),
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
    except ImportError:
        return _notice(
            "torch",
            "PyTorch is needed for this figure",
            "This figure builds the real network to read its layer shapes and "
            "parameter counts, which needs PyTorch. Install it with "
            "'pip install torch' and reload this page. Nothing else on this "
            "site needs it.",
        )
    except Exception as error:  # pragma: no cover - defensive
        return _notice(
            "error",
            "This figure could not be drawn",
            f"{error.__class__.__name__}: {error}",
        )


def figure_root() -> Path:
    """The project directory these figures read their data from."""
    return project_root()
