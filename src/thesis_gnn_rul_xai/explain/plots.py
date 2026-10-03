"""Draws how each sensor's contribution to the prediction moves over one engine's life.

Read README.md in this folder. It explains how to read the figure.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

__all__ = ["plot_contributions_over_time"]

# The colours of the project's charts: one surface, three inks, hairlines, a
# blue and red pair for "pushed the prediction up" and "pushed it down", and two
# series colours for comparing methods.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
HAIRLINE = "#e1e0d9"
AXIS = "#c3c2b7"
UP = "#2a78d6"
DOWN = "#e34948"
SERIES = ("#2a78d6", "#eb6834")

_COLUMNS = 4


def _style(axis: plt.Axes) -> None:
    """Hairline axes and grid, muted tick labels, nothing heavier."""
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(AXIS)
        axis.spines[side].set_linewidth(0.8)
    axis.tick_params(colors=INK_MUTED, labelsize=8, length=2)
    axis.grid(axis="y", color=HAIRLINE, linewidth=0.6)
    axis.set_axisbelow(True)


def plot_contributions_over_time(
    table: pd.DataFrame,
    sensors: Sequence[str],
    *,
    compare: pd.DataFrame | None = None,
    method: str = "Shapley value",
    compare_method: str = "occlusion",
    title: str | None = None,
) -> Figure:
    """One small panel per sensor, largest average contribution first, above the prediction."""
    sensors = list(sensors)
    missing = [s for s in ["end_cycle", "rul", "predicted", "baseline", *sensors] if s not in table]
    if missing:
        raise ValueError(f"table has no column(s) {missing}; pass what dynamic_shap returns.")
    if compare is not None and len(compare) != len(table):
        raise ValueError(
            f"compare has {len(compare)} row(s) but table has {len(table)}; they must "
            "describe the same windows."
        )

    cycles = table["end_cycle"].to_numpy()
    values = table[sensors].to_numpy(dtype=float)
    order = np.argsort(-np.abs(values).mean(axis=0), kind="stable")
    ceiling = float(table["rul"].max())
    falling = table.loc[table["rul"] < ceiling, "end_cycle"]
    knee = float(falling.min()) if len(falling) else None

    rows = int(np.ceil((len(sensors) + 1) / _COLUMNS))
    figure = plt.figure(figsize=(13, 3.0 + 2.1 * rows), facecolor=SURFACE)
    grid = figure.add_gridspec(rows + 1, _COLUMNS, height_ratios=[1.5] + [1] * rows,
                               hspace=0.55, wspace=0.12)

    top = figure.add_subplot(grid[0, :])
    _style(top)
    # The window's label: the remaining life capped at the ceiling, 125 cycles by default (assumption A-01).
    top.plot(cycles, table["rul"], color=INK_MUTED, linewidth=1.6,
             label="label: remaining life, capped")
    top.plot(cycles, table["predicted"], color=UP, linewidth=2.0, label="predicted")
    base = float(table["baseline"].iloc[0])
    top.axhline(base, color=INK_SECONDARY, linewidth=0.8)
    # Both labels sit where an engine near failure leaves the plot empty: the
    # right-hand end, above the average-window line, and the foot of the knee.
    top.annotate(f"prediction for an average window: {base:.1f}", (cycles[-1], base),
                 xytext=(0, 4), textcoords="offset points", ha="right", fontsize=8,
                 color=INK_SECONDARY)
    if knee is not None:
        top.axvline(knee, color=AXIS, linewidth=0.8)
        top.annotate(f"label starts falling, cycle {knee:.0f}", (knee, 0.0),
                     xytext=(4, 2), textcoords="offset points", fontsize=8,
                     color=INK_SECONDARY)
    top.set_ylabel("cycles left", fontsize=9, color=INK_SECONDARY)
    top.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=INK_SECONDARY)
    top.set_title(title or "The prediction, and where it came from", loc="left",
                  fontsize=11, color=INK)

    limit = float(np.abs(values).max())
    if compare is not None:
        limit = max(limit, float(np.abs(compare[sensors].to_numpy(dtype=float)).max()))
    limit *= 1.08

    first = None
    for slot, index in enumerate(order):
        sensor = sensors[index]
        axis = figure.add_subplot(grid[1 + slot // _COLUMNS, slot % _COLUMNS],
                                  sharex=first, sharey=first)
        first = first or axis
        _style(axis)
        series = values[:, index]
        axis.axhline(0.0, color=AXIS, linewidth=0.8)
        if knee is not None:
            axis.axvline(knee, color=HAIRLINE, linewidth=0.8)
        if compare is None:
            axis.fill_between(cycles, series, 0.0, where=series >= 0, color=UP, alpha=0.28,
                              linewidth=0, interpolate=True)
            axis.fill_between(cycles, series, 0.0, where=series < 0, color=DOWN, alpha=0.28,
                              linewidth=0, interpolate=True)
            axis.plot(cycles, series, color=INK_SECONDARY, linewidth=1.0)
        else:
            axis.plot(cycles, series, color=SERIES[0], linewidth=1.3, label=method)
            axis.plot(cycles, compare[sensor].to_numpy(dtype=float), color=SERIES[1],
                      linewidth=1.3, label=compare_method)
        axis.set_ylim(-limit, limit)
        axis.set_title(f"{sensor}   average size {np.abs(series).mean():.1f}", loc="left",
                       fontsize=9, color=INK)
        if slot % _COLUMNS:
            axis.tick_params(labelleft=False)
        if slot + _COLUMNS < len(sensors):
            axis.tick_params(labelbottom=False)
        else:
            axis.set_xlabel("flight cycle", fontsize=8, color=INK_MUTED)

    key = figure.add_subplot(grid[1 + len(sensors) // _COLUMNS, len(sensors) % _COLUMNS:])
    key.axis("off")
    if compare is None:
        lines = [
            f"Each panel: one sensor's {method}, in cycles, at every window.",
            "Blue area: that sensor pushed the predicted life up.",
            "Red area: it pushed the predicted life down.",
            "Every panel shares one vertical scale, so heights compare directly.",
        ]
    else:
        handles = [plt.Line2D([], [], color=c, linewidth=2) for c in SERIES]
        key.legend(handles, [method, compare_method], loc="upper left", fontsize=9,
                   frameon=False, labelcolor=INK_SECONDARY)
        lines = [
            "Both in cycles added to the predicted life.",
            f"Panel titles give the {method}'s average size.",
            "Every panel shares one vertical scale.",
        ]
    key.text(0.0, 0.95 if compare is None else 0.6, "\n".join(lines), va="top",
             fontsize=9, color=INK_SECONDARY, linespacing=1.6, transform=key.transAxes)
    return figure
