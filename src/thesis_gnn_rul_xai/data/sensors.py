"""
C-MAPSS sensor definitions and sensor selection
===============================================

What the dataset contains
-------------------------
C-MAPSS stands for *Commercial Modular Aero-Propulsion System Simulation*. It
is a simulation program written at NASA that models a large turbofan aircraft
engine. NASA used it to generate a public benchmark dataset in which simulated
engines are run until they fail, while 21 virtual sensors record the engine
state once per flight cycle (Saxena et al., 2008).

Each row of a raw C-MAPSS file describes one engine at one flight cycle and has
26 columns:

===========  ==========================================================
Column       Meaning
===========  ==========================================================
1            Engine unit number (an identifier, not a measurement)
2            Time, counted in flight cycles since the engine was new
3, 4, 5      Three operational settings (altitude, Mach number, and
             throttle resolver angle)
6 ... 26     Twenty-one sensor measurements
===========  ==========================================================

A *flight cycle* is one complete take-off, flight and landing. It is the unit
of time used throughout this project; there are no seconds or hours anywhere.

Why only 14 of the 21 sensors are used
--------------------------------------
Some of the 21 sensors never change during the whole lifetime of an engine, or
change only between a few fixed levels. A signal that never changes carries no
information about degradation and cannot help a model predict failure. Removing
such signals is standard practice on this dataset.

Wen et al. (2026) keep 14 sensors. Their Table 3 prints the adjacency matrix of
the sensor graph over exactly these 14 sensors, in a specific order. **That
order matters**: the adjacency matrix is indexed by position, so if the sensor
columns were reordered, the graph would connect the wrong sensors. The order is
therefore fixed in this module and checked by an assertion.

Two findings about the paper's sensor selection
-----------------------------------------------
Both were established by running :func:`detect_uninformative_sensors` on the
real FD001 training data, and both belong in the thesis rather than in a
comment.

**The unnamed seventh sensor is NRc.** Section 3.2.2 of Wen et al. (2026)
states that "seven sensors are removed", but then names only six: T2, P2, epr,
farB, Nfdmd and PCNRdmd. Measured on the data, exactly those six are perfectly
constant. The seventh sensor missing from their Table 3 is NRc, the corrected
core speed, and it is **not** constant: it takes 6078 distinct values across
20 631 training rows. It must therefore have been dropped for a different
reason, most plausibly redundancy, since the corrected core speed is derived
from the physical core speed Nc, which they keep.

**P15 is kept but carries almost no information.** The bypass-duct pressure
P15 is one of the 14 sensors in their graph, yet on FD001 it takes only **two**
distinct values in the entire training set (21.60 and 21.61). It cannot
represent gradual degradation. It is kept here anyway, because the adjacency
matrix is defined over the paper's fixed 14-sensor set and removing a node
would change the graph. The observation is reported rather than silently
corrected.

This module follows Table 3, because Table 3 is the list the model actually
consumes. Both findings are recorded in ``docs/ASSUMPTIONS.md``.

References
----------
Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation
modeling for aircraft engine run-to-failure simulation. *2008 International
Conference on Prognostics and Health Management*, Denver, CO, pp. 1-9. IEEE.
https://doi.org/10.1109/PHM.2008.4711414

Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
(2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
enhancement for aeroengine RUL prediction. *Aerospace Science and Technology*,
168, 110875. https://doi.org/10.1016/j.ast.2025.110875
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = [
    "CYCLE_COL",
    "OP_SETTING_COLS",
    "PAPER_EXCLUDED_SYMBOLS",
    "PAPER_SENSOR_COLUMNS",
    "PAPER_SENSOR_SYMBOLS",
    "RAW_COLUMNS",
    "SENSORS",
    "SENSOR_COLS",
    "UNIT_COL",
    "SensorInfo",
    "column_to_symbol",
    "detect_uninformative_sensors",
    "symbol_to_column",
]

#: Name of the column that identifies an engine.
UNIT_COL = "unit"

#: Name of the column that counts flight cycles.
CYCLE_COL = "cycle"

#: Names of the three operational-setting columns.
OP_SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]

#: Names of the 21 raw sensor columns, in file order.
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]

#: The complete list of column names for a raw C-MAPSS file, in file order.
RAW_COLUMNS = [UNIT_COL, CYCLE_COL, *OP_SETTING_COLS, *SENSOR_COLS]


@dataclass(frozen=True)
class SensorInfo:
    """Description of a single C-MAPSS sensor.

    Attributes
    ----------
    index : int
        Position of the sensor in the file, from 1 to 21.
    symbol : str
        Short engineering symbol used in the aerospace literature, for example
        ``"T24"`` for the total temperature at the low-pressure compressor
        outlet.
    description : str
        Plain-language description of what the sensor measures.
    unit : str
        Physical unit. ``"degR"`` is degrees Rankine, an absolute temperature
        scale used in American aerospace engineering; ``"psia"`` is pounds per
        square inch, absolute; ``"rpm"`` is revolutions per minute;
        ``"lbm/s"`` is pounds of mass per second; ``"-"`` means the quantity is
        a dimensionless ratio.
    station : str
        The part of the engine where the sensor sits. This is used only for
        documentation and figures, never by the model.
    """

    index: int
    symbol: str
    description: str
    unit: str
    station: str

    @property
    def column(self) -> str:
        """The name of this sensor's column in the loaded data frame."""
        return f"sensor_{self.index}"


#: All 21 C-MAPSS sensors. Symbols, descriptions and units follow Table 2 of
#: Wen et al. (2026), which in turn follows Saxena et al. (2008).
SENSORS: tuple[SensorInfo, ...] = (
    SensorInfo(1, "T2", "Total temperature at fan inlet", "degR", "Fan inlet"),
    SensorInfo(2, "T24", "Total temperature at LPC outlet", "degR", "LPC outlet"),
    SensorInfo(3, "T30", "Total temperature at HPC outlet", "degR", "HPC outlet"),
    SensorInfo(4, "T50", "Total temperature at LPT outlet", "degR", "LPT outlet"),
    SensorInfo(5, "P2", "Pressure at fan inlet", "psia", "Fan inlet"),
    SensorInfo(6, "P15", "Total pressure in bypass duct", "psia", "Bypass duct"),
    SensorInfo(7, "P30", "Total pressure at HPC outlet", "psia", "HPC outlet"),
    SensorInfo(8, "Nf", "Physical fan speed", "rpm", "Fan shaft"),
    SensorInfo(9, "Nc", "Physical core speed", "rpm", "Core shaft"),
    SensorInfo(10, "epr", "Engine pressure ratio (P50/P2)", "-", "Engine overall"),
    SensorInfo(11, "Ps30", "Static pressure at HPC outlet", "psia", "HPC outlet"),
    SensorInfo(12, "phi", "Ratio of fuel flow to Ps30", "pps/psi", "Combustor"),
    SensorInfo(13, "NRf", "Corrected fan speed", "rpm", "Fan shaft"),
    SensorInfo(14, "NRc", "Corrected core speed", "rpm", "Core shaft"),
    SensorInfo(15, "BPR", "Bypass ratio", "-", "Bypass duct"),
    SensorInfo(16, "farB", "Burner fuel-air ratio", "-", "Combustor"),
    SensorInfo(17, "htBleed", "Bleed enthalpy", "-", "Bleed system"),
    SensorInfo(18, "Nf_dmd", "Demanded fan speed", "rpm", "Control"),
    SensorInfo(19, "PCNfR_dmd", "Demanded corrected fan speed", "rpm", "Control"),
    SensorInfo(20, "W31", "HPT coolant bleed", "lbm/s", "HPT cooling"),
    SensorInfo(21, "W32", "LPT coolant bleed", "lbm/s", "LPT cooling"),
)

#: The 14 sensors kept by Wen et al. (2026), in the exact row and column order
#: of their Table 3. Do not reorder: the adjacency matrix is indexed by this
#: order, so a change here silently rewires the sensor graph.
PAPER_SENSOR_SYMBOLS: tuple[str, ...] = (
    "T24", "T30", "T50", "P15", "P30", "Nf", "Nc",
    "Ps30", "phi", "NRf", "BPR", "htBleed", "W31", "W32",
)

#: The 7 sensors dropped by Wen et al. (2026). See the module docstring for the
#: discrepancy in how the paper describes this set.
#:
#: ASSUMPTION A-10: Section 3.2.2 says seven sensors are removed and then names
#: only six. ``NRc`` is the seventh, identified as the one sensor of Table 2
#: that is absent from Table 3 without being named in that sentence.
PAPER_EXCLUDED_SYMBOLS: tuple[str, ...] = (
    "T2", "P2", "epr", "NRc", "farB", "Nf_dmd", "PCNfR_dmd",
)

_SYMBOL_TO_SENSOR = {s.symbol: s for s in SENSORS}
_COLUMN_TO_SENSOR = {s.column: s for s in SENSORS}


def symbol_to_column(symbol: str) -> str:
    """Translate an engineering symbol such as ``"T24"`` into ``"sensor_2"``."""
    try:
        return _SYMBOL_TO_SENSOR[symbol].column
    except KeyError as exc:
        raise KeyError(
            f"Unknown sensor symbol {symbol!r}. "
            f"Valid symbols are: {sorted(_SYMBOL_TO_SENSOR)}"
        ) from exc


def column_to_symbol(column: str) -> str:
    """Translate a column name such as ``"sensor_2"`` into ``"T24"``."""
    try:
        return _COLUMN_TO_SENSOR[column].symbol
    except KeyError as exc:
        raise KeyError(
            f"Unknown sensor column {column!r}. "
            f"Valid columns are: {sorted(_COLUMN_TO_SENSOR)}"
        ) from exc


#: The 14 retained sensors expressed as data frame column names.
PAPER_SENSOR_COLUMNS: tuple[str, ...] = tuple(
    symbol_to_column(s) for s in PAPER_SENSOR_SYMBOLS
)

# Consistency checks that run when the module is first imported. If any of
# these fail the package is internally inconsistent and must not be used.
assert len(PAPER_SENSOR_SYMBOLS) == 14, "Table 3 defines exactly 14 sensors."
assert len(set(PAPER_SENSOR_SYMBOLS)) == 14, "Duplicate sensor in the kept set."
assert set(PAPER_SENSOR_SYMBOLS).isdisjoint(PAPER_EXCLUDED_SYMBOLS), (
    "A sensor cannot be both kept and excluded."
)
assert len(set(PAPER_SENSOR_SYMBOLS) | set(PAPER_EXCLUDED_SYMBOLS)) == 21, (
    "The kept and excluded sets together must cover all 21 sensors."
)


def detect_uninformative_sensors(
    frame: pd.DataFrame,
    *,
    relative_tolerance: float = 1e-9,
    min_distinct_values: int = 3,
) -> dict[str, list[str]]:
    """Find sensors that carry no usable degradation information.

    This is a *diagnostic*, not a filter. The pipeline always uses the fixed
    14-sensor list from the paper, because the adjacency matrix depends on it.
    The purpose of this function is to let us check, from the data itself,
    whether the paper's choice agrees with what the measurements show. Any
    disagreement is a finding worth reporting, not something to hide.

    Two simple criteria are applied to each of the 21 sensor columns:

    1. **Constant**: the difference between the largest and the smallest value
       is essentially zero. Such a signal is a flat line.
    2. **Few distinct values**: the sensor only ever takes a small number of
       discrete levels. This happens when a quantity is rounded coarsely, and
       the resulting signal is usually too crude to show gradual wear.

    Parameters
    ----------
    frame : pandas.DataFrame
        A loaded C-MAPSS data frame containing the 21 raw sensor columns.
    relative_tolerance : float
        A sensor counts as constant when its range (maximum minus minimum) is
        at most this fraction of the absolute mean value. A relative test is
        used because the sensors have very different magnitudes: a range of
        0.01 is negligible for a temperature near 1400 degR but large for a
        ratio near 0.03.
    min_distinct_values : int
        A sensor with fewer distinct values than this is reported as
        low-resolution. The default of 3 means "flags any signal that takes at
        most two different values", which cannot represent gradual wear.

        Note that *distinct count* is the right criterion here, not the size of
        the range. On the real FD001 training data the fan speed ``Nf`` has a
        relative range of 2.8e-4 across 53 distinct levels and is genuinely
        informative, while the bypass-duct pressure ``P15`` has a slightly
        larger relative range of 4.6e-4 spread over only 2 levels and is not.
        A range test would keep the wrong one of the two.

    Returns
    -------
    dict
        A dictionary with three keys: ``"constant"``, ``"low_resolution"`` and
        ``"informative"``. Each maps to a sorted list of engineering symbols.

    Examples
    --------
    A frame in which sensor 1 is a flat line and sensor 2 varies:

    >>> import pandas as pd
    >>> frame = pd.DataFrame({c: [1.0, 1.0, 1.0] for c in SENSOR_COLS})
    >>> frame["sensor_2"] = [1.0, 2.0, 3.0]
    >>> report = detect_uninformative_sensors(frame)
    >>> report["informative"]
    ['T24']
    >>> "T2" in report["constant"]
    True
    """
    constant: list[str] = []
    low_resolution: list[str] = []
    informative: list[str] = []

    for sensor in SENSORS:
        values = frame[sensor.column]
        value_range = float(values.max() - values.min())
        scale = max(abs(float(values.mean())), 1e-12)
        n_distinct = int(values.nunique())

        if value_range <= relative_tolerance * scale:
            constant.append(sensor.symbol)
        elif n_distinct < min_distinct_values:
            low_resolution.append(sensor.symbol)
        else:
            informative.append(sensor.symbol)

    return {
        "constant": sorted(constant),
        "low_resolution": sorted(low_resolution),
        "informative": sorted(informative),
    }
