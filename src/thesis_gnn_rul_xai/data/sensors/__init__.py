"""Names the 21 sensors, and holds the list of the 14 the paper keeps.

The order of that list must not change. The sensor graph is stored as a plain
grid of numbers, so its rows are matched to sensors by position alone;
reordering the list would quietly connect the wrong sensors together.

Read README.md in this folder. It explains everything this file does, and the
two things we found out about the paper's own choice of sensors.
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

# The column that says which engine a row belongs to.
UNIT_COL = "unit"

# The column that counts flight cycles.
CYCLE_COL = "cycle"

# The three columns describing how the aircraft was being flown.
OP_SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]

# The 21 sensor columns, in the order they appear in the file.
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]

# All 26 column names of a raw file, in file order.
RAW_COLUMNS = [UNIT_COL, CYCLE_COL, *OP_SETTING_COLS, *SENSOR_COLS]


@dataclass(frozen=True)
class SensorInfo:
    """What one sensor is. README.md in this folder explains the units."""

    index: int
    symbol: str
    description: str
    unit: str
    station: str

    @property
    def column(self) -> str:
        """The name of this sensor's column in a loaded table."""
        return f"sensor_{self.index}"


# All 21 sensors. The names, descriptions and units come from Table 2 of
# Wen et al. (2026), which in turn follows Saxena et al. (2008).
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

# The 14 sensors Wen et al. (2026) keep, in the exact order of their Table 3.
# Do not reorder. The sensor graph is matched to this list by position, so a
# change here would quietly connect the wrong sensors together.
PAPER_SENSOR_SYMBOLS: tuple[str, ...] = (
    "T24", "T30", "T50", "P15", "P30", "Nf", "Nc",
    "Ps30", "phi", "NRf", "BPR", "htBleed", "W31", "W32",
)

# The 7 sensors Wen et al. (2026) drop.
#
# ASSUMPTION A-10: NRc is the seventh of them. Their Section 3.2.2 says seven
# sensors are removed and then names only six; NRc is the one sensor in their
# Table 2 that is missing from their Table 3 without being named. Explained in
# README.md in this folder; written up in docs/assumptions/A-10.md.
PAPER_EXCLUDED_SYMBOLS: tuple[str, ...] = (
    "T2", "P2", "epr", "NRc", "farB", "Nf_dmd", "PCNfR_dmd",
)

_SYMBOL_TO_SENSOR = {s.symbol: s for s in SENSORS}
_COLUMN_TO_SENSOR = {s.column: s for s in SENSORS}


def symbol_to_column(symbol: str) -> str:
    """Translate a sensor name such as "T24" into a column name, "sensor_2"."""
    try:
        return _SYMBOL_TO_SENSOR[symbol].column
    except KeyError as exc:
        raise KeyError(
            f"Unknown sensor symbol {symbol!r}. "
            f"Valid symbols are: {sorted(_SYMBOL_TO_SENSOR)}"
        ) from exc


def column_to_symbol(column: str) -> str:
    """Translate a column name such as "sensor_2" into a sensor name, "T24"."""
    try:
        return _COLUMN_TO_SENSOR[column].symbol
    except KeyError as exc:
        raise KeyError(
            f"Unknown sensor column {column!r}. "
            f"Valid columns are: {sorted(_COLUMN_TO_SENSOR)}"
        ) from exc


# The same 14 sensors, written as column names.
PAPER_SENSOR_COLUMNS: tuple[str, ...] = tuple(
    symbol_to_column(s) for s in PAPER_SENSOR_SYMBOLS
)

# Checks that run the moment this file is imported. If one of them fails, the
# package contradicts itself and must not be used.
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
    """Find the sensors whose readings never really move.

    This only reports; it never changes which sensors are used. It exists so
    the paper's choice of sensors can be checked against the data itself.
    README.md in this folder explains the two tests it applies.
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
