"""What the Design tab shows before anything runs: the data, the cautions, the network and each box's flowchart."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from ...evaluation.published import DATASET_DESCRIPTION, SUBSETS
from .commands import preprocessing_settings
from .settings import (
    _BY_KEY,
    RESULT_SEEDS,
    SettingsError,
    fixed_graph_refusal,
    read_more,
)


def caveats(settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    """What a reader must be told before setting this configuration beside the paper."""
    notes: list[dict[str, Any]] = []

    # "run": it qualifies a stored run's numbers, so the run's page lists it.
    # "design": it flags a conflict between settings the reader may not have
    # open, so the Design tab shows it beside the Train button as well.
    def add(level: str, ref: str, text: str, where: tuple[str, ...] = ("run",)) -> None:
        notes.append({"level": level, "ref": ref, "text": text, "where": list(where)})

    seeds = int(settings.get("seeds") or 0)
    if seeds < RESULT_SEEDS:
        add("warn", "", f"{seeds} seed{'' if seeds == 1 else 's'} is exploratory: quote "
            "nothing from it. A result is a mean over at least five seeds, with its spread.",
            ("design", "run"))
    if settings.get("temporal") in ("lstm", "gru"):
        add("info", "A-26", "The memory size and conversion from temporal output to graph input are project "
            "choices. Table 4 specifies one graph-input number per sensor but not how it is obtained.")
    if settings.get("temporal") == "last_cycle":
        add("info", "D-6", "Only each window's last cycle reaches the graph layers, which "
            "the paper's w/o LSTM model may not have done.")
    if settings.get("shared_across_sensors") and settings.get("temporal") in ("lstm", "gru"):
        add("info", "Section 2.2", "One shared network is not the paper's stage, which "
            "gives each sensor its own.")
    if list(settings.get("gcn_hidden") or [32, 64, 32]) != [32, 64, 32] or \
            int(settings.get("fc_hidden") or 128) != 128:
        add("info", "Table 4", "The layer widths differ from Table 4.")
    if float(settings.get("validation_fraction") or 0.0) > 0:
        add("warn", "F-16", "The paper does not describe selecting an earlier pass by validation error. "
            "This project's measured LSTM comparison depends on whether that selection is used.")
    elif settings.get("lr_schedule") == "plateau":
        add("warn", "A-17", "The plateau schedule needs held-back engines; with none the "
            "rate never leaves its first value.", ("design", "run"))
    if settings.get("clip_test_labels") is True:
        add("info", "A-02", "Test answers are capped at the prediction ceiling. This cannot increase "
            "RMSE or the NASA score for these bounded predictions, but R² and ASUE need not improve.")
    if settings.get("graph") == "complete":
        # Every sensor of the complete graph has the same number of neighbours,
        # so both rescalings give the same matrix, 1/14 everywhere.
        add("info", "D-2", "The complete graph averages every sensor equally, so it "
            "measures the cost of averaging as much as the value of the physics.")
    # Only with engines held back, each seed drawing its own, do seeds get different graphs.
    varies = float(settings.get("validation_fraction") or 0.0) > 0 and not settings.get("fixed_split")
    if settings.get("graph") == "correlation" and varies:
        add("info", "A-27", "The graph is rebuilt from each seed's own training engines, "
            "so two seeds can be given slightly different graphs.")
    # A random or rewired graph is drawn from the seed the data is prepared
    # under: one for every seed only when the same engines are held back for
    # all of them, as in scripts/train.py, which refuses the same held-back
    # engines with none held back. The Design tab never shows that refusal as
    # a caution: validate_settings refuses the request before it is drawn.
    if settings.get("graph") in ("random", "rewired"):
        refusal = fixed_graph_refusal(settings)
        if refusal:
            add("warn", "A-41", refusal)
        else:
            add("info", "A-41", "The graph is drawn at random from "
                + ("one seed and serves every seed." if settings.get("fixed_split")
                   else "each seed, so every seed is given a different graph."))
    subset, window = settings.get("subset"), settings.get("window_size")
    if subset in SUBSETS and isinstance(window, int) and not isinstance(window, bool):
        counted = dataset_facts(str(subset), window)["counted"]
        if counted and counted["left_out"] > 0:
            one = counted["left_out"] == 1
            add("warn", "A-23", f"{counted['left_out']} of {counted['test_engines']} test "
                f"engines {'is' if one else 'are'} shorter than the window and "
                f"{'was' if one else 'were'} left out, so the score cannot be set beside "
                "Table 7.")
    if settings.get("fixed_split"):
        add("info", "", "Keeping the same held-back engines across seeds leaves out variation "
            "due to choosing a different group of engines.")
    return notes


def _test_lengths(subset: str) -> tuple[int, ...]:
    """How many cycles each test engine of a sub-dataset recorded, or nothing while its file is missing."""
    from ...utils.paths import raw_data_dir

    return _read_test_lengths(subset) if (raw_data_dir() / f"test_{subset}.txt").is_file() else ()


@lru_cache(maxsize=8)
def _read_test_lengths(subset: str) -> tuple[int, ...]:
    """How many cycles each test engine of a sub-dataset recorded, from the real file."""
    from ...data.loading import read_cmapss_frame
    from ...data.sensors import CYCLE_COL, UNIT_COL
    from ...utils.paths import raw_data_dir

    frame = read_cmapss_frame(raw_data_dir() / f"test_{subset}.txt")
    return tuple(int(n) for n in frame.groupby(UNIT_COL)[CYCLE_COL].max().tolist())


def dataset_facts(subset: str, window_size: int) -> dict[str, Any]:
    """What Table 5 says about a sub-dataset, and how many test engines this window leaves out."""
    facts: dict[str, Any] = {
        "table5": DATASET_DESCRIPTION[subset],
        "counted": None,
        "files": _file_facts(subset),
    }
    lengths = _test_lengths(subset)
    if lengths:
        # An engine shorter than the window yields no window at all (A-23).
        facts["counted"] = {
            "test_engines": len(lengths),
            "left_out": sum(1 for n in lengths if n < window_size),
        }
    return facts


def preview(settings: Mapping[str, Any]) -> dict[str, Any]:
    """What the Design tab draws for these settings: the data, the cautions, the network built for real."""
    answer: dict[str, Any] = {
        "dataset": dataset_facts(settings["subset"], int(settings["window_size"])),
        "design_cautions": [note for note in caveats(settings) if "design" in note["where"]],
        "torch": True,
    }
    try:
        model = _network(settings)
    except _NoTorch:
        answer["torch"] = False
        return answer
    except ValueError as refused:
        answer["network_error"] = str(refused)
        return answer
    answer["layers"] = model.layer_summary()
    answer["parameters"] = model.count_parameters()
    return answer


class _NoTorch(Exception):
    """PyTorch is missing, so no network can be built."""


def _network(settings: Mapping[str, Any], matrix: Any = None) -> Any:
    """The network these settings build, for real, on the given sensor graph or one that stands in for it."""
    try:
        from ...graph.adjacency import (
            build_fully_connected_adjacency,
            build_physics_adjacency,
            normalize_adjacency,
        )
        from ...models import ModelConfig, build_model
    except ImportError as missing:
        # Only a missing PyTorch is expected here; any other failed import is a fault.
        if not (missing.name or "").startswith("torch"):
            raise
        raise _NoTorch() from None

    # The correlation graph needs the training engines, which a preview does not
    # load. The layers and their parameter counts do not depend on which edges
    # the graph has, so the paper's graph stands in for it when none is given.
    if matrix is None:
        matrix, symbols = build_physics_adjacency()
        if settings["graph"] == "complete":
            matrix, _ = build_fully_connected_adjacency(len(symbols))
    return build_model(
        normalize_adjacency(matrix, method=settings["adjacency_normalization"]),
        _model_config(settings, ModelConfig),
        window_size=int(settings["window_size"]),
        rul_max=int(settings["rul_max"]),
    )


def _model_config(settings: Mapping[str, Any], factory: Any) -> Any:
    """The ModelConfig the settings describe."""
    return factory(
        temporal=settings["temporal"],
        temporal_hidden=int(settings["temporal_hidden"]),
        temporal_out_features=int(settings["temporal_out_features"]),
        shared_across_sensors=bool(settings["shared_across_sensors"]),
        gcn_hidden=tuple(settings["gcn_hidden"]),
        fc_hidden=int(settings["fc_hidden"]),
        gcn_bias=bool(settings["gcn_bias"]),
    )


# -- one box of the pipeline, as its own flowchart -----------------------------
# README.md in this folder says what each box's flowchart shows and why. Every
# number in them is read from the files, from a real preprocessing run or from
# the network built for real; the words say what the code does, in its order.
# A step's tag says where its operation comes from; a value on screen that is
# not the default says so in words, with where the default came from, so a
# value the paper never printed never carries the paper's tag.

STAGES = ("data", "preprocessing", "temporal", "graph")

_SOURCE_WORDS = {"paper": "the paper's {ref}", "assumption": "assumption {ref}",
                 "project": "a rule of this repository", "extension": "an option the paper does not have"}

# One preprocessing run at a time: a second request for the same settings then
# finds the first one's answer kept, instead of running it again beside it.
_PREPARE_LOCK = threading.Lock()


class _Undescribable(Exception):
    """A box whose flowchart cannot be drawn for these settings, with the reason in words."""


def stage_detail(stage: str, settings: Mapping[str, Any]) -> dict[str, Any]:
    """One box of the page's pipeline as its own flowchart: its steps, in the order the code runs them."""
    builders = {"data": _data_steps, "preprocessing": _preprocessing_steps,
                "temporal": _temporal_steps, "graph": _graph_steps}
    if stage not in builders:
        raise SettingsError(f"stage must be one of {list(STAGES)}, got {stage!r}.")
    from ...data.loading import DOWNLOAD_URL

    answer: dict[str, Any] = {"stage": stage, "steps": []}
    needs_files = stage in ("data", "preprocessing") or (stage == "graph" and settings["graph"] == "correlation")
    if needs_files and not _files_present(settings["subset"]):
        answer["note"] = (f"The NASA files for {settings['subset']} are not in data/raw/, so nothing here is "
                          f"counted. They can be downloaded from {DOWNLOAD_URL}")
    try:
        answer["steps"] = builders[stage](settings)
    except _Undescribable as reason:
        answer["error"] = str(reason)
    pending = list(answer["steps"])
    while pending:
        step = pending.pop()
        pending.extend(step.get("steps", []))
        pending.extend(step.get("lanes", []))
        if step.get("picture"):
            step["more"] = read_more(f"operation={step['name']}")
        if not step.get("shape"):
            continue
        picture, name = step.get("picture"), step["name"]
        axes, label, note = "rows, columns", "Data table", ""
        if stage == "temporal":
            axes, label = "windows, sensors, values per sensor", "Sensor values"
            if picture in ("split", "recurrent"):
                values = "readings per cycle" if picture == "split" else "memory values per cycle"
                label = "Sensor histories" if picture == "split" else "Memory after every cycle"
                if settings["shared_across_sensors"]:
                    axes = f"window-sensor pairs, cycles, {values}"
                    note = "The first dimension combines the window and sensor counts; each pair is processed separately."
                else:
                    axes = f"sensor histories x (windows, cycles, {values})"
                    note = "The leading multiplier counts separate sensor histories, each with one sequence per window."
            elif name == "Keep the last hidden state":
                label = "Final memory values"
                axes = ("window-sensor pairs, memory values" if settings["shared_across_sensors"]
                        else "windows, sensors, memory values")
                note = "These are the memory values after the final cycle of each sensor's window."
                if settings["shared_across_sensors"]:
                    note += " The first dimension combines the window and sensor counts."
            elif name in ("Last reading", "Average", "Change over the window"):
                axes, label = "windows, sensors", name
                note = "There is one summary value for each sensor in each window."
            elif (step.get("icon") == "preprocessing" or picture == "window"
                  or (step.get("kind") == "edge" and settings["temporal"] == "raw_window")):
                axes, label = "windows, sensors, cycles", "Sensor readings"
            elif picture == "project":
                note = "Each sensor passes these computed values to the graph layers."
        elif stage == "graph":
            axes, label = "windows, sensors, values per sensor", "Sensor values"
            if picture in ("graph", "table"):
                axes, label = "receiving sensors (rows), sending sensors (columns)", "Sensor graph"
                note = ("Each entry says whether the column's sensor can send to the row's sensor."
                        if picture == "graph" else
                        "Each entry is the weight used when the column's sensor sends to the row's sensor.")
            elif picture in ("flatten", "dense"):
                axes, label = "windows, values per window", "Combined sensor values"
                note = "Each window now has one list of values, with no separate sensor dimension."
            elif picture in ("sigmoid", "scale") or step.get("icon") == "output":
                axes = "windows"
                label = "Life fractions" if picture == "sigmoid" else "Remaining life in cycles"
                note = "There is one prediction per window. The comma in (B,) denotes a single dimension."
        if not step.get("outputs"):
            step["outputs"] = [{"label": label, "shape": step["shape"], "axes": axes}]
        if "B" in step["shape"]:
            note = (note + f" B is the number of windows processed together, at most {int(settings['batch_size'])}.").strip()
        if note:
            step["output_note"] = note
    return answer


def _step(name: str, does: str, *, parts: list[tuple[Any, ...]] | None = None, shape: str = "",
          learned: int | None = None, source: Mapping[str, str] | None = None,
          picture: str = "", summary: str = "", outputs: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """One step of a box's flowchart: its name, the values that decide it, what it hands on and its picture."""
    return {"name": name, "does": " ".join(does.split()), "shape": shape, "learned": learned,
            "parts": [{"text": part[0], "setting": part[1], "index": part[2] if len(part) > 2 else None}
                      for part in (parts or [])],
            "source": dict(source) if source else None, "picture": picture, "summary": summary,
            "outputs": outputs or []}


def _edge(name: str, shape: str = "", icon: str = "") -> dict[str, Any]:
    """What enters or leaves a box, drawn as words beside the picture of the box it comes from or goes to."""
    return {"kind": "edge", "name": name, "shape": shape, "icon": icon}


def _together(*steps: dict[str, Any], shape: str = "") -> dict[str, Any]:
    """Steps that work side by side on the same input, drawn in one row, and what the row hands on."""
    return {"kind": "group", "steps": list(steps), "shape": shape}


def _branch(*steps: dict[str, Any]) -> dict[str, Any]:
    """Steps that make a second input for the step after them, drawn beside the main line."""
    return {"kind": "branch", "steps": list(steps)}


def _networks(stage: Any, kind: str, window: int, shared: bool) -> dict[str, Any]:
    """The recurrent networks of a temporal stage, for drawing them one by one and one from inside."""
    from ...data.sensors import PAPER_SENSOR_SYMBOLS

    cell = stage.cells[0]
    gates = 4 if kind == "lstm" else 3
    learned = sum(p.numel() for p in cell.parameters())
    return {"kind": kind, "shared": shared, "sensors": list(PAPER_SENSOR_SYMBOLS), "window": window,
            "hidden": int(cell.hidden_size), "inputs": int(cell.input_size), "learned_each": int(learned),
            "per_gate": int(learned // gates), "gates": gates}


def _cite(ref: str) -> dict[str, str]:
    """A step's tag: the part of the paper its operation comes from."""
    return {"kind": "paper", "ref": ref, "text": ""}


def _count(n: int) -> str:
    """A whole number with a space between thousands, as the page prints it: "20 631"."""
    return f"{int(n):,}".replace(",", " ")


def _plural(n: int, word: str) -> str:
    """A count and its noun, singular or plural: "1 cycle", "30 cycles"."""
    return f"{_count(n)} {word}{'' if n == 1 else 's'}"


def _default_words(key: str) -> str:
    """A setting's default, in the form's words: "30 cycles", "on", "20%", "Each row sums to one"."""
    spec = _BY_KEY[key]
    value = spec["default"]
    if spec["kind"] == "choice":
        return f'"{_label_of(key, value)}"'
    if key == "validation_fraction":
        return f"{value:.0%}"
    if spec["kind"] == "bool":
        return "on" if value else "off"
    if spec["kind"] == "widths":
        return ", ".join(str(w) for w in value[:-1]) + f" and {value[-1]}"
    unit = _unit_of(spec["label"])
    return f"{value:g} {unit}".strip() if isinstance(value, float) else f"{value} {unit}".strip()


def _unit_of(label: str) -> str:
    """The unit a setting's label ends with, "(cycles)", or nothing."""
    return label[label.rindex("(") + 1:-1] if label.endswith(")") and "(" in label else ""


def _decided(key: str, value: Any) -> tuple[dict[str, str] | None, str]:
    """Where a value on screen comes from: its tag, or none and a sentence saying it is not the default."""
    spec = _BY_KEY[key]
    for choice in spec.get("choices", []):
        if choice["value"] == value and choice["source"]:
            return choice["source"], ""
    source = spec["source"]
    # A paper's list of options, or an option the paper does not have, describes every value alike.
    if value == spec["default"] or (source and (source["kind"] == "extension" or (
            spec["kind"] == "choice" and source["kind"] == "paper"))):
        return source, ""
    where = f" ({_SOURCE_WORDS[source['kind']].format(ref=source['ref'])})" if source else ""
    return None, f'"{spec["label"].split(" (")[0]}" was changed from its default, {_default_words(key)}{where}.'


def _label_of(key: str, value: Any) -> str:
    """An option's words, as the form shows them."""
    for choice in _BY_KEY[key].get("choices", []):
        if choice["value"] == value:
            return str(choice["label"])
    return str(value)


def _files_present(subset: str) -> bool:
    """Whether all three NASA files of a sub-dataset are in data/raw/."""
    from ...utils.paths import raw_data_dir

    return all((raw_data_dir() / f"{kind}_{subset}.txt").is_file() for kind in ("train", "test", "RUL"))


@lru_cache(maxsize=4)
def _file_counts(subset: str) -> dict[str, Any]:
    """Rows and engines in one sub-dataset's three files, and the kept sensors that barely vary, counted from them."""
    from ...data.loading import read_cmapss_frame, read_rul_file
    from ...data.sensors import (
        CYCLE_COL,
        OP_SETTING_COLS,
        PAPER_SENSOR_SYMBOLS,
        SENSORS,
        UNIT_COL,
        detect_uninformative_sensors,
    )
    from ...utils.paths import raw_data_dir

    folder = raw_data_dir()
    train, test = read_cmapss_frame(folder / f"train_{subset}.txt"), read_cmapss_frame(folder / f"test_{subset}.txt")
    check = detect_uninformative_sensors(train)
    train_lengths = train.groupby(UNIT_COL).size()
    test_lengths = test.groupby(UNIT_COL).size()
    sample_columns = [
        {"key": UNIT_COL, "label": "Engine", "detail": UNIT_COL},
        {"key": CYCLE_COL, "label": "Cycle", "detail": CYCLE_COL},
        *[{"key": key, "label": f"Setting {i}", "detail": key}
          for i, key in enumerate(OP_SETTING_COLS, start=1)],
        *[{"key": sensor.column, "label": sensor.symbol,
          "detail": sensor.column} for sensor in SENSORS],
    ]
    sample_keys = [column["key"] for column in sample_columns]
    first_unit = int(train[UNIT_COL].iloc[0])
    sample = train.loc[train[UNIT_COL] == first_unit, sample_keys].head(10)
    answers = read_rul_file(folder / f"RUL_{subset}.txt")
    first_test = int(test[UNIT_COL].iloc[0])
    return {"train_rows": len(train), "train_engines": int(train[UNIT_COL].nunique()),
            "test_rows": len(test), "test_engines": int(test[UNIT_COL].nunique()),
            "labels": len(answers),
            "label_example": {"train_unit": first_unit,
                              "train_last": int(train.loc[train[UNIT_COL] == first_unit, CYCLE_COL].max()),
                              "test_unit": first_test,
                              "test_last": int(test.loc[test[UNIT_COL] == first_test, CYCLE_COL].max()),
                              "test_answer": float(answers.iloc[0])},
            "train_history": {"shortest": int(train_lengths.min()),
                              "median": float(train_lengths.median()),
                              "longest": int(train_lengths.max())},
            "test_history": {"shortest": int(test_lengths.min()),
                             "median": float(test_lengths.median()),
                             "longest": int(test_lengths.max())},
            "sample": {"source": f"train_{subset}.txt", "columns": sample_columns,
                       "rows": [{key: (int(row[key]) if key in (UNIT_COL, CYCLE_COL)
                                       else float(row[key])) for key in sample_keys}
                                for _, row in sample.iterrows()]},
            "flat": [s for s in PAPER_SENSOR_SYMBOLS if s in check["constant"] + check["low_resolution"]]}


def _file_facts(subset: str) -> dict[str, Any] | None:
    """The counts above, or None while the files are missing; a missing file is never remembered."""
    return _file_counts(subset) if _files_present(subset) else None


@lru_cache(maxsize=16)
def _prepared(key: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    """A real run's counts, small worked examples and sensor graph, without retaining its full tables or windows."""
    import numpy as np

    from ...config import PreprocessingConfig
    from ...data.pipeline import run_preprocessing
    from ...data.smoothing import apply_ewma
    from ...data.windowing import split_units_by_id

    config = PreprocessingConfig(**dict(key))
    run = run_preprocessing(config, keep_frames=True)
    raw = run.raw.train
    train_units, held_units = split_units_by_id(raw.unit.unique(), validation_fraction=config.validation_fraction,
                                               seed=config.seed)
    sample = raw.head(3)
    smooth = apply_ewma(sample, ["sensor_2"], alpha=config.smoothing_alpha)
    examples = [{"engine": int(row.unit), "cycle": int(row.cycle), "reading": float(row.sensor_2),
                 "life": int(row.rul)} for row in sample.itertuples()]
    available = [data for data in (run.train, run.validation) if len(data)]
    first = min(available, key=lambda data: int(data.units.min()))
    engine = int(first.units.min())
    positions = np.flatnonzero(first.units == engine)[:3]
    windows = [{"start": int(first.end_cycles[i]) - config.window_size + 1,
                "end": int(first.end_cycles[i]), "life": float(first.targets[i])} for i in positions]
    return {"train": len(run.train), "validation": len(run.validation), "test": len(run.test),
            "kept": len(train_units), "held": len(held_units),
            "train_rows": int(raw.unit.isin(train_units).sum()), "held_rows": int(raw.unit.isin(held_units).sum()),
            "conditions": int(run.condition_encoder.n_detected), "adjacency": run.adjacency_raw.copy(),
            "example_rows": examples, "smoothed": smooth.sensor_2.tolist(),
            "rescaled": run.frames["train"].loc[sample.index, "sensor_2"].tolist(),
            "window_example": {"engine": engine, "windows": windows}}


def _prepared_for(settings: Mapping[str, Any]) -> dict[str, Any] | None:
    """The preprocessing run these settings describe, or None when the NASA files are missing."""
    if not _files_present(settings["subset"]):
        return None
    key = preprocessing_settings(settings)
    # The rescaling of the graph, and the graph unless it is built from the
    # data, change nothing a flowchart reads from the run.
    del key["adjacency_normalization"]
    if key["graph"] != "correlation":
        del key["graph"]
    with _PREPARE_LOCK:
        return _prepared(tuple(sorted(key.items())))


def _table_outputs(facts: dict[str, Any] | None, columns: int) -> list[dict[str, str]]:
    """Name both working tables and the meaning of their dimensions."""
    return [{"label": label, "shape": f"({_count(facts[key]) if facts else 'rows'}, {columns})",
             "axes": "rows, columns"}
            for label, key in (("Training file", "train_rows"), ("Test file", "test_rows"))]


def _window_outputs(run: dict[str, Any] | None, window: int, with_validation: bool) -> list[dict[str, str]]:
    """Name each collection's sensor-input shape and its separate remaining-life-answer shape."""
    groups = [("Training", run["train"] if run else None),
              *([("Validation", run["validation"] if run else None)] if with_validation else []),
              ("Test", run["test"] if run else None)]
    purposes = {"Training": "Used to update the model during training", "Validation": "Used to check progress during training",
                "Test": "Used for final evaluation after training"}
    return [{"label": label, "shape": f"({_count(n) if n is not None else 'windows'}, 14, {window})",
             "axes": "windows, sensors, cycles", "targets": f"({_count(n) if n is not None else 'windows'},)"}
            | {"use": purposes[label]}
            for label, n in groups]


def _data_steps(settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Separate the training-file path from the test histories and their answer file."""
    subset = settings["subset"]
    facts = _file_facts(subset)
    raw, labelled = _table_outputs(facts, 26), _table_outputs(facts, 27)
    tag, _ = _decided("subset", subset)
    training = _step(
        "Training engines",
        "Each row contains one engine's readings at one cycle, together with its engine number and operating "
        "settings. Each engine is followed to its final recorded cycle, so its remaining life can be calculated. "
        "Whole engines may be set aside to check training progress.",
        parts=[(f"train_{subset}.txt", None)], shape=raw[0]["shape"], picture="file", source=tag,
        outputs=[raw[0]], summary="Readings continue until each engine's final cycle.")
    test = _step(
        "Test engines",
        "These are different engines from those in the training file. Recording stops before failure. "
        "The model predicts life left from each engine's last complete window, a sequence of consecutive "
        "readings. A separate answer file checks those predictions after training; its values are never sensor inputs.",
        parts=[(f"test_{subset}.txt", None)], shape=raw[1]["shape"], picture="file", source=tag,
        outputs=[raw[1]], summary="Readings stop before the engine fails.")
    train_labels = _step(
        "Calculate training labels",
        "For each row, subtract its cycle from the same engine's final recorded cycle. The result is its "
        "label: the known remaining life used to teach the model. Preprocessing then limits these answers "
        "to the selected ceiling. They never become sensor inputs.",
        picture="life", outputs=[labelled[0]], summary="Subtract the current cycle from the engine's final cycle.")
    selection = _step(
        "Select 14 sensor columns for model inputs",
        "Choose which sensor readings the model receives. Engine number, cycle, operating settings and "
        "remaining-life answers stay available for preparing the data, but do not become sensor inputs.",
        picture="select", source=_cite("Table 3"), outputs=labelled,
        summary="The selected sensors supply the model's readings.")
    if facts:
        ex = facts["label_example"]
        for step, kind, engine in ((training, "train", ex["train_unit"]), (test, "test", ex["test_unit"])):
            step["example"] = {"kind": "change", "before": f"{kind}_{subset}.txt",
                               "note": "Read whitespace-separated numbers into named columns; keep rows in engine/cycle order.",
                               "after": f"Engine {engine}, cycle 1: engine + cycle + 3 settings + 21 sensor values (26 columns)."}
        train_labels["example"] = {"kind": "change", "before": f"Engine {ex['train_unit']}, cycle 1; final cycle {ex['train_last']}",
                                   "note": f"{ex['train_last']} - 1", "after": f"{ex['train_last'] - 1} raw cycles left"}
        selection["example"] = {"kind": "change", "before": "All 21 named sensor columns",
                                "note": "Choose the model's sensor-column list without changing the readings.",
                                "after": "T24, T30, T50, P15, P30, Nf, Nc, Ps30, phi, NRf, BPR, htBleed, W31, W32"}
    return [
        _edge("Read the engine histories", icon="data"),
        # Keep the steps flat for consumers that inspect a flowchart, while
        # retaining the two paths for the browser renderer to show as lanes.
        {"kind": "group", "steps": [training, train_labels, test],
         "lanes": [
             {"name": "Training data", "steps": [training, train_labels]},
             {"name": "Test data", "steps": [test], "shape": raw[1]["shape"]}]},
        selection,
        _edge("To preprocessing", icon="preprocessing"),
    ]


def _preprocessing_steps(settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Box 2: from labelled rows to windows, in the order the pipeline runs its steps."""
    run = _prepared_for(settings)
    subset, seed = settings["subset"], settings["seed"]
    ceiling, window, stride = int(settings["rul_max"]), int(settings["window_size"]), int(settings["window_stride"])
    fraction = float(settings["validation_fraction"])
    several = DATASET_DESCRIPTION[subset]["conditions"] > 1
    # Until the windows are cut, what passes from step to step is the training
    # file's table: its rows, and 27 columns, one more once the operating
    # condition is added. The test file's table goes the same way beside it.
    facts = _file_facts(subset)
    rows = _count(facts["train_rows"]) if facts else "rows"
    table, table_more = f"({rows}, 27)", f"({rows}, 28)"
    steps = [_edge(f"{subset}: recorded engine histories", table, icon="data")]

    _, ceiling_note = _decided("rul_max", ceiling)
    capped = bool(settings["clip_test_labels"])
    _, test_note = _decided("clip_test_labels", capped)
    steps.append(_step(
        "Cap the labels",
        f"Labels above {ceiling} cycles are replaced by {ceiling}; smaller labels stay unchanged. This makes "
        "the learning target constant early in life, although the actual time until failure keeps decreasing. "
        + ("Test answers are capped too, so scores assess this capped target (assumption A-02)." if capped else
           "Test answers retain their uncapped values.")
        + (f" {ceiling_note}" if ceiling_note else " The default ceiling is a figure-based interpretation "
           "recorded in assumption A-01, not a printed value.") + (f" {test_note}" if test_note else ""),
        parts=[(f"at {_plural(ceiling, 'cycle')}", "rul_max"),
               ("test labels too" if capped else "not the test labels", "clip_test_labels")],
        shape=table, source=_cite("Eq. 19"), picture="cap"))

    # Switched off, smoothing is no step at all, and the flowchart leaves it out.
    if settings["smoothing"]:
        alpha = float(settings["smoothing_alpha"])
        _, alpha_note = _decided("smoothing_alpha", alpha)
        blends = alpha < 1
        steps.append(_step(
            "Smooth each sensor",
            (f"For each sensor and engine, combine {alpha:g} times the current reading with {1 - alpha:g} times "
             "the previous smoothed value. The first reading starts the sequence. This softens rapid "
             "fluctuations using only the present and past, before rescaling." if blends else
             "With all the weight on the newest reading, each reading passes through unchanged.")
            + " " + (alpha_note or "The default weight is this project's interpretation of the paper's "
                                    "ambiguous wording (assumption A-04).")
            + (f" In {subset}, successive readings can come from different operating conditions, and this "
               "step blends across them." if several and blends else ""),
            parts=[(f"weight {alpha:g} on the newest reading", "smoothing_alpha")], shape=table,
            source=_cite("Eq. 18"), picture="smooth"))

    if fraction > 0:
        tag, note = _decided("validation_fraction", fraction)
        count = f"{run['held']} of {run['held'] + run['kept']} engines" if run else f"{fraction:.0%} of the engines"
        steps.append(_step(
            "Hold engines back",
            f"Set aside {fraction:.0%} of the training-file engines before calculating rescaling statistics "
            f"(random seed {seed}). These validation engines check predictions after each full pass through the "
            "training windows and select which learned weights to keep. Whole engines are separated so "
            "overlapping windows of one engine cannot appear in both groups. The Training procedure controls "
            "this choice; it is applied here before rescaling. The test file stays separate. " + note,
            parts=[(count, "validation_fraction")], shape=table, source=tag, picture="hold"))
    else:
        _, note = _decided("validation_fraction", fraction)
        steps.append(_step(
            "Hold engines back",
            "Every training-file engine is used to learn the model's weights. Training runs for the chosen "
            "number of passes and keeps the final weights, without a separate group for selecting an earlier "
            "version. The Training procedure controls this choice; it is applied here before rescaling. "
            "The test file stays separate. " + note,
            parts=[("none", "validation_fraction")], shape=table, picture="hold"))

    method = settings["normalization"]
    conditions = run["conditions"] if run else None
    steps.append(_step(
        "Find operating conditions",
        "The three operating settings describe how an engine is run. Their rounded combinations determine "
        "how many groups to use, up to six. When there is more than one, k-means (a method that groups nearby "
        "rows) finds representative centres from training engines and assigns each row to its nearest centre. "
        + ("Rescaling below uses these groups." if method == "minmax_per_condition" else
           "The chosen rescaling does not use these groups."),
        parts=[(f"{_plural(conditions, 'condition')} found" if conditions else "", None)], shape=table_more,
        picture="conditions"))

    tag, note = _decided("normalization", method)
    formula = ("Subtract each sensor's training mean and divide by its standard deviation, a measure of spread"
               if method == "zscore" else
               "Subtract each sensor's training minimum and divide by its training range (maximum minus minimum)")
    where = {"zscore": "one mean and one spread per sensor",
             "minmax_per_condition": "one range per sensor within each operating condition",
             "minmax_global": "one range per sensor over all rows"}[method]
    applied = "every engine, the held-back and test engines included" if fraction > 0 else "every engine"
    pooled = (" The operating conditions share one scale; differences in operating conditions can therefore "
              "remain in the inputs."
              if several and method != "minmax_per_condition" else "")
    steps.append(_step(
        "Rescale each sensor",
        f"{formula}, using {where}. Apply those same numbers to {applied}. "
        + ("Values are not limited to a fixed interval." if method == "zscore" else
           "Training minima become 0 and maxima become 1; later values can lie outside that range.")
        + " A sensor with negligible training variation is set to 0. " + note + pooled,
        parts=[(_label_of("normalization", method).lower(), "normalization")], shape=table_more, source=tag,
        picture="rescale"))

    tag, note = _decided("window_size", window)
    _, stride_note = _decided("window_stride", stride)
    every = "every cycle" if stride == 1 else f"every {stride} cycles"
    left_out = (dataset_facts(subset, window)["counted"] or {}).get("left_out", 0)
    steps.append(_step(
        "Cut into windows",
        f"A window is {_plural(window, 'consecutive cycle')} from one engine, arranged as 14 sensor histories. "
        f"Training-file windows start {every}; each is paired with the remaining life at its final cycle. "
        "A test engine supplies only its last complete window. Every window keeps its engine's group "
        "from Hold engines back; there is no new split here. " + note + " " + stride_note
        + (f" {_plural(left_out, 'test engine')} shorter than {_plural(window, 'cycle')} "
                    f"{'is' if left_out == 1 else 'are'} left out (assumption A-23)." if left_out else ""),
        parts=[(_plural(window, "cycle"), "window_size"), (f"a new one {every}", "window_stride")],
        shape=f"({_count(run['train'] + run['validation'])}, 14, {window})" if run else f"(windows, 14, {window})",
        source=tag, picture="window"))

    steps.append(_edge("To the temporal stage, B windows at a time", f"(B, 14, {window})", icon="temporal"))
    _preprocessing_story(steps, settings, run, facts)
    return steps


def _preprocessing_story(steps: list[dict[str, Any]], settings: Mapping[str, Any],
                         run: dict[str, Any] | None, facts: dict[str, Any] | None) -> None:
    """Restore both tables' labelled shapes and keep the single-row walkthrough separate from the flow."""
    window = int(settings["window_size"])
    opening = steps[0]
    opening["name"] = "Engine histories and remaining-life answers"
    opening["outputs"] = _table_outputs(facts, 27)
    opening["handoff_outputs"] = opening["outputs"]
    opening["summary"] = ""
    opening["columns"] = "1 engine number + 1 cycle + 3 operating settings + 21 sensor readings + 1 uncapped remaining-life label."
    opening["handoff"] = (
        f"{facts['train_rows']:,} rows · 27 columns" if facts else "Rows · 27 columns"
    )
    for step in steps[1:-1]:
        picture = step["picture"]
        step["outputs"] = _table_outputs(facts, 28 if picture in ("conditions", "rescale") else 27)
        if picture in ("hold", "conditions", "rescale"):
            groups = [("Training", run["train_rows"] if run else None),
                      *([("Validation", run["held_rows"] if run else None)]
                        if settings["validation_fraction"] else []),
                      ("Test", facts["test_rows"] if facts else None)]
            columns = 27 if picture == "hold" else 28
            step["outputs"] = [{"label": label, "shape": f"({_count(n) if n is not None else 'rows'}, {columns})",
                                "axes": "rows, columns"} for label, n in groups]
        if picture == "cap":
            step["summary"] = "Limit remaining-life answers to the selected ceiling."
        elif picture == "smooth":
            step["summary"] = "Blend each reading with earlier readings from the same engine."
        elif picture == "hold":
            step["summary"] = (("Assign whole engines to training or validation. "
                                if settings["validation_fraction"] else "Keep all training-file engines for learning. ")
                               + "Chosen in Training → Procedure; applied here before rescaling.")
        elif picture == "conditions":
            step["summary"] = "Group rows with similar operating settings."
        elif picture == "rescale":
            step["summary"] = "Use the training engines to set each sensor's scale."
        elif picture == "window":
            step["summary"] = "Group consecutive readings; each window keeps its engine's assigned group."
            step["outputs"] = _window_outputs(run, window, with_validation=bool(settings["validation_fraction"]))
            step["aside"] = "Training-file engines supply overlapping windows. Each test engine supplies only its last complete window."
        step["handoff_outputs"] = step["outputs"]
    steps[-1]["name"] = "To the temporal stage"
    steps[-1]["outputs"] = [{"label": "Model inputs", "shape": f"(B, 14, {window})", "axes": "windows, sensors, cycles"}]
    steps[-1]["summary"] = f"B counts windows processed together (at most {int(settings['batch_size']):,})."
    if run:
        row = run["example_rows"][1]
        capped = min(row["life"], int(settings["rul_max"]))
        records = [{"step": "Original row", "reading": row["reading"], "life": row["life"]},
                   {"step": "Cap the label", "reading": row["reading"], "life": capped}]
        if settings["smoothing"]:
            records.append({"step": "Smooth the sensor", "reading": run["smoothed"][1], "life": capped})
        records.append({"step": "Rescale the sensor", "reading": run["rescaled"][1], "life": capped})
        opening["walkthrough"] = {"file": f"train_{settings['subset']}.txt", "engine": row["engine"],
                                  "cycle": row["cycle"], "records": records,
                                  "windows": run["window_example"]}

        # Keep the real examples available separately from the compact diagram.
        opening["example"] = {"kind": "rows", "source": f"train_{settings['subset']}.txt",
                              "label_state": "uncapped", "rows": [
            {"engine": item["engine"], "cycle": item["cycle"],
             "reading": item["reading"], "life": item["life"]}
            for item in run["example_rows"]
        ]}
        cap = next(step for step in steps if step.get("picture") == "cap")
        cap["example"] = {"kind": "change", "label": "Example remaining-life label",
                           "before": f"{row['life']} cycles",
                           "after": f"{capped} cycles"}
        smooth = next((step for step in steps if step.get("picture") == "smooth"), None)
        if smooth:
            alpha = float(settings["smoothing_alpha"])
            smooth["example"] = {"kind": "change", "label": "Sensor T24 after smoothing",
                                  "before": str(row["reading"]),
                                  "after": str(run["smoothed"][1]),
                                  "note": f"{alpha:g} x current reading + {1 - alpha:g} x previous smoothed reading"}
        rescale = next(step for step in steps if step.get("picture") == "rescale")
        before_scale = (f"{run['smoothed'][1]:.3f}" if settings["smoothing"]
                        else str(row["reading"]))
        rescale["example"] = {"kind": "change", "label": "Sensor T24 after scaling",
                               "before": before_scale,
                               "after": f"{run['rescaled'][1]:.4f}",
                               "note": "The scale is fitted on training engines and then reused."}
        hold = next(step for step in steps if step.get("picture") == "hold")
        hold["example"] = {"kind": "groups", "items": [
            {"name": "Training", "value": f"{run['train_rows']:,} rows",
             "detail": f"{run['train_rows']:,} rows used to learn"},
            *([{"name": "Held back", "value": f"{run['held_rows']:,} rows",
                "detail": f"{run['held_rows']:,} rows used to check progress"}] if run["held_rows"] else []),
        ]}
        window_step = next(step for step in steps if step.get("picture") == "window")
        window_step["example"] = {"kind": "windows", "engine": run["window_example"]["engine"],
                                   "windows": run["window_example"]["windows"]}
        for step in steps:
            if step.get("example", {}).get("kind") == "change":
                step["example"]["context"] = f"train_{settings['subset']}.txt · engine {row['engine']} · cycle {row['cycle']}"
    else:
        opening["handoff"] = "Rows · 27 columns"


def _built(settings: Mapping[str, Any], matrix: Any = None) -> Any:
    """The network for a box's flowchart, or the reason it cannot be built."""
    if settings["temporal"] == "window_summary" and int(settings["window_size"]) < 2:
        raise _Undescribable("The window summary measures the change over the window, so the window length must "
                             "be at least 2 cycles.")
    try:
        return _network(settings, matrix)
    except _NoTorch:
        raise _Undescribable("PyTorch is not installed, so the network cannot be built to describe it.") from None
    except ValueError as refused:
        raise _Undescribable(str(refused)) from None


def _temporal_steps(settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Box 3: how each sensor's window becomes the numbers the graph layers receive."""
    model = _built(settings)
    stage, kind = model.temporal, settings["temporal"]
    window, nodes, k = int(settings["window_size"]), model.n_nodes, int(model.in_features)
    readings = _plural(window, "reading")
    chosen, _ = _decided("temporal", kind)
    steps = [_edge("Windows from preprocessing, B at a time", f"(B, {nodes}, {window})", icon="preprocessing")]
    if kind in ("lstm", "gru"):
        hidden, shared = int(settings["temporal_hidden"]), bool(settings["shared_across_sensors"])
        _, hidden_note = _decided("temporal_hidden", hidden)
        cells = sum(p.numel() for p in stage.cells.parameters())
        name = "LSTM" if kind == "lstm" else "GRU"
        rows = f"B x {nodes}" if shared else "B"
        inside = (
            f"For each window, this LSTM starts with two lists of {_plural(hidden, 'number')}, all zero. "
            "The cell state is the list it updates by keeping part of the previous values and adding new "
            "values. The hidden state is an output calculated from that updated list. The current reading "
            "and previous hidden state determine the new values and the multipliers called gates. "
            "Each gate is between 0 and 1: 0 keeps none of a value, 1 keeps all of it. Both lists pass to "
            "the next cycle. " if kind == "lstm" else
            f"For each window, this GRU starts with a list of {_plural(hidden, 'number')}, all zero. "
            "This list is called its hidden state. It combines each new reading with that list to calculate "
            "an updated list. Multipliers between 0 and 1, called gates, control how much of the old and "
            "new information is kept. The updated list passes to the next cycle. ")
        steps.append(_step(
            "Split by sensor",
            (f"Each window becomes {nodes} separate sensor histories, one reading per cycle. The same learned "
             "rule processes every sensor, but their histories and memories remain separate." if shared else
             f"Each sensor's {readings} {'becomes' if window == 1 else 'become'} a sequence of its own, one "
             "number per cycle. A separate learned rule processes each sensor. Sensors do not exchange "
             "readings within this stage."),
            parts=[("one sequence per window and sensor" if shared else f"{nodes} sequences of {readings}", None)],
            shape=f"({rows}, {window}, 1)" if shared else f"{nodes} x (B, {window}, 1)",
            source=_decided("shared_across_sensors", True)[0] if shared else chosen,
            picture="split"))
        steps.append(_step(
            f"One {name} for all sensors" if shared else f"One {name} per sensor",
            f"Reads the {_plural(window, 'cycle')} in order, from oldest to newest. " + inside
            + "The same weights calculate these updates at every cycle. Training adjusts those weights "
            "using the final RUL prediction error, through the graph layers that follow. The lists reset "
            "for each new window; the learned weights remain. "
            + (hidden_note or "The memory width is our choice (assumption A-26)."),
            parts=[(f"hidden {hidden}", "temporal_hidden"),
                   ("shared" if shared else f"{len(stage.cells)} networks", "shared_across_sensors")],
            shape=f"({rows}, {window}, {hidden})" if shared else f"{nodes} x (B, {window}, {hidden})", learned=cells,
            source=_decided("shared_across_sensors", True)[0] if shared else chosen, picture="recurrent"))
        steps[-1]["networks"] = _networks(stage, kind, window, shared)
        steps.append(_step(
            "Keep the last hidden state",
            f"Each sensor's network produces {_plural(hidden, 'number')} after each reading. "
            f"After reading all {_plural(window, 'cycle')}, keep the last set of {_plural(hidden, 'number')} "
            "and send it to the conversion below. This last set was calculated using the earlier sets, "
            "so it can still contain information from earlier readings. During training, the prediction "
            "error is traced back through those earlier calculations too. "
            + ("The cell state has already helped calculate this last hidden state; it is not an additional "
               "input to the conversion below." if kind == "lstm" else
               "This GRU option is an extension of the paper's architecture."),
            parts=[(f"{_plural(hidden, 'number')} per sensor", None)],
            shape=f"(B x {nodes}, {hidden})" if shared else f"(B, {nodes}, {hidden})",
            source=_cite("Eq. 7") if kind == "lstm" else chosen, picture="last"))
        projection = sum(p.numel() for p in stage.project.parameters())
        _, out_note = _decided("temporal_out_features", k)
        steps.append(_step(
            "Turn into numbers per sensor",
            f"Multiply the summary's {_plural(hidden, 'number')} by learned weights, add them, and add a "
            f"learned offset to produce each of {_plural(k, 'output number')}. All sensors use the same weights "
            "and offsets. This conversion is our choice; the paper specifies one number entering the graph "
            "per sensor but not how to produce it (assumption A-26)."
            + (f" {out_note}" if out_note else ""),
            parts=[(f"{k} per sensor", "temporal_out_features")], shape=f"(B, {nodes}, {k})",
            learned=projection, source=_decided("temporal_out_features", 1)[0], picture="project"))
    elif kind == "last_cycle":
        steps.append(_step(
            "Keep the last reading",
            "Each sensor keeps only its most recent preprocessed reading"
            + (f"; the other {_plural(window - 1, 'reading')} {'is' if window == 2 else 'are'} dropped"
               if window > 1 else "")
            + ". This step learns no weights. Earlier cycles can still contribute through smoothing, if "
            "enabled. Using the last reading in place of the paper's LSTM memory stage is our choice "
            "(assumption A-07).",
            parts=[(f"cycle {window} of {window}", "temporal")], shape=f"(B, {nodes}, 1)", source=chosen,
            picture="last"))
    elif kind == "mean_window":
        steps.append(_step(
            "Average the window",
            (f"Each sensor's {readings} are replaced by their plain average." if window > 1 else
             "Each sensor's single reading is its own average.") + " This step learns no weights and loses "
            "the order of readings within the window. It is an extension, not a stage described in the paper.",
            parts=[(f"{readings} to 1", "temporal")], shape=f"(B, {nodes}, 1)", source=chosen,
            picture="average"))
    elif kind == "window_summary":
        steps += [
            _together(
                _step("Last reading", "Each sensor's reading at the most recent cycle.",
                      parts=[("1 number", None)], shape=f"(B, {nodes})", picture="last"),
                _step("Average", f"The plain average of the {readings}.", parts=[("1 number", None)],
                      shape=f"(B, {nodes})", picture="average"),
                _step("Change over the window",
                      f"Fit a straight line to the {readings} by minimising the squared differences between "
                      "each reading and the line's value at that cycle. Keep its final value minus its first: positive "
                      "means rising, negative means falling. This measures the overall trend, not just "
                      "the difference between the first and last readings.",
                      parts=[("1 number", None)], shape=f"(B, {nodes})",
                      picture="slope"),
            ),
            _step("Three numbers per sensor",
                  "Put each sensor's last reading, average and fitted change side by side. These fixed "
                  "calculations give the graph layers information about both level and trend; the graph "
                  "layers learn how to combine them. This is our extension (assumption A-28).",
                  parts=[("last · average · change", "temporal")], shape=f"(B, {nodes}, 3)", source=chosen,
                  picture="stack"),
        ]
    else:
        first = int(settings["gcn_hidden"][0])
        steps.append(_step(
            "Pass every cycle",
            (f"All {readings} go on unchanged" if window > 1 else "The single reading goes on unchanged")
            + f". The first graph layer learns {_plural(first, 'weighted sum')} across these cycle positions "
            "while also combining connected sensors. The weights are shared across sensors. This is an "
            "extension of the paper's architecture, which processes time separately before the graph layers.",
            parts=[(f"all {readings}", "temporal")], shape=f"(B, {nodes}, {window})", source=chosen,
            picture="window"))
    steps.append(_edge("To the graph layers", f"(B, {nodes}, {k})", icon="graph"))
    return steps


def _graph_steps(settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Box 4: the sensor graph, the three graph layers and the layers that turn them into remaining life."""
    import numpy as np

    from ...data.sensors import PAPER_SENSOR_SYMBOLS
    from ...graph.adjacency import (
        build_fully_connected_adjacency,
        build_physics_adjacency,
        normalize_adjacency,
        pairs_beyond_reach,
    )

    graph, rescaling, seed = settings["graph"], settings["adjacency_normalization"], settings["seed"]
    symbols = list(PAPER_SENSOR_SYMBOLS)
    if graph == "correlation":
        run = _prepared_for(settings)
        matrix = run["adjacency"] if run else None
    elif graph == "complete":
        matrix, _ = build_fully_connected_adjacency(len(symbols))
    else:
        matrix, _ = build_physics_adjacency()
    model = _built(settings, matrix)
    layers = {row["layer"]: row for row in model.layer_summary()}
    nodes, k = model.n_nodes, int(model.in_features)

    graph_words = {
        "physics": "the pairs in the paper's Table 3, proposed from engine structure and physical relationships",
        "complete": "every sensor joined to every other",
        "correlation": "pairs with the strongest linear association in training readings, whether they rise "
                       "together or in opposite directions",
    }[graph]
    extra, pairs, sums, beyond, apart = "", "", "", [], 0
    if matrix is not None:
        joined = np.asarray(matrix) > 0
        np.fill_diagonal(joined, False)
        if graph != "correlation":  # its name already says how many pairs it keeps
            pairs = f" · {_count(int(joined.sum() // 2))} pairs"
        rescaled = normalize_adjacency(np.asarray(matrix), method=rescaling)
        alone = [s for s, row in zip(symbols, joined, strict=True) if not row.any()]
        twins: dict[bytes, list[str]] = {}
        for symbol, row in zip(symbols, np.round(rescaled, 6), strict=True):
            twins.setdefault(row.tobytes(), []).append(symbol)
        groups = [group for group in twins.values() if len(group) > 1]
        if graph == "correlation":
            if float(settings["validation_fraction"]) == 0:
                extra += " It uses all training-file engines; none are held back."
            elif settings["fixed_split"]:
                extra += " Each repeat uses the same selection of training engines."
            else:
                extra += (f" This is random seed {seed}'s graph; each repeat rebuilds it using its own "
                          "training engines.")
        if alone:
            extra += (f" {' and '.join(alone)} {'has' if len(alone) == 1 else 'have'} no neighbours, so "
                      "the graph layers receive only their own information at those positions.")
        if graph == "complete":
            extra += (" Rescaling then gives every sensor the same average, so all sensor positions have "
                      "identical numbers after the first graph layer.")
        elif groups:
            named = "; ".join(", ".join(g[:-1]) + " and " + g[-1] for g in groups)
            extra += (f" Identical rescaled rows ({named}) make these sensor positions produce identical "
                      "outputs in the first graph layer.")
        totals = rescaled.sum(axis=1)
        sums = (" Every row sums to 1." if np.allclose(totals, 1, atol=1e-5) else
                f" Its rows sum to between {totals.min():.4f} and {totals.max():.4f}, not to 1.")
        if graph == "complete":
            sums = " On this graph both rescalings give the same table, 1/14 everywhere, so this setting changes nothing."
        beyond = sorted((min(a, b, key=symbols.index), max(a, b, key=symbols.index))
                        for a, b, _ in pairs_beyond_reach(np.asarray(matrix), symbols, 3))
        # Pairs no path joins at all: pairs_beyond_reach counts only pairs a path joins.
        part = list(range(len(symbols)))
        for i, j in zip(*np.nonzero(joined), strict=True):
            a, b = part[i], part[j]
            part = [a if x == b else x for x in part]
        apart = sum(1 for i in range(len(symbols)) for j in range(i + 1, len(symbols)) if part[i] != part[j])
    else:
        extra = " It is built from the training engines, which are not in data/raw/, so it cannot be shown here."

    graph_tag, graph_note = _decided("graph", graph)
    graph_steps = [
        _step("Sensor graph",
              f"Each node is a sensor; a connection allows two sensors to exchange information. Here we connect "
              f"{graph_words}. The table uses 1 for a connection and 0 otherwise, including 0 on its diagonal. "
              f"Training does not change the connections.{extra} {graph_note}",
              parts=[(_label_of("graph", graph) + pairs, "graph")], shape=f"({nodes}, {nodes})", source=graph_tag,
              picture="graph"),
    ]
    tag, note = _decided("adjacency_normalization", rescaling)
    graph_steps.append(_step(
        "Rescale the graph",
        ("Include each sensor itself among its neighbours. Divide each row by its number of included sensors. "
         "That row then gives equal weights to itself and its neighbours, so the next layer takes their average."
         if rescaling == "row" else
         "Include each sensor itself among its neighbours. Divide each connection's 1 by the square root "
         "of the product of the two sensors' neighbour counts, each counting itself. Both directions receive "
         "the same weight, so the table is symmetric. This need not produce an average.")
        + sums + (f" {note}" if note else ""),
        parts=[(_label_of("adjacency_normalization", rescaling).lower(), "adjacency_normalization")],
        shape=f"({nodes}, {nodes})", source=tag, picture="table"))
    # The graph is a second input of the first graph layer, beside the windows.
    steps = [_edge("From the temporal stage, B windows at a time", f"(B, {nodes}, {k})", icon="temporal"),
             _branch(*graph_steps)]

    bias = bool(settings["gcn_bias"])
    defaults = _BY_KEY["gcn_hidden"]["default"]
    width_out = str(k)
    index = 1
    while f"GCN-{index}" in layers:
        row = layers[f"GCN-{index}"]
        width_in = row["in"].split(",")[-1].strip(" )")
        width_out = row["out"].split(",")[-1].strip(" )")
        reach = {1: " An output can depend on the sensor itself and its immediate neighbours.",
                 2: " Through the preceding layer, an output can depend on sensors up to two connections away.",
                 3: " Through the preceding layers, an output can depend on sensors up to three connections away."}.get(index, "")
        if index == 3 and (beyond or apart):
            far = [f"{_plural(len(beyond), 'pair')} {'is' if len(beyond) == 1 else 'are'} further apart than that ("
                   + ", ".join(f"{a}-{b}" for a, b in beyond) + ")"] if beyond else []
            far += [f"{_plural(apart, 'pair')} {'is' if apart == 1 else 'are'} joined by no path at all"] if apart else []
            reach += (" On this graph " + " and ".join(far) + "; one sensor's input cannot reach the other's "
                      "position in these three layers.")
        width_note = (f' "Graph layer widths" was changed from its default for this layer, {defaults[index - 1]} '
                      "(the paper's Table 4)." if index <= len(defaults) and int(width_out) != defaults[index - 1]
                      else "")
        steps.append(_step(
            f"Graph layer {index}",
            f"First combine connected sensors' {_plural(int(width_in), 'number')} using the graph's fixed "
            f"weights. Learned multipliers then make {_plural(int(width_out), 'weighted sum')} per sensor"
            + (", with learned offsets (our extension, assumption A-12)" if bias else ", with no added offset")
            + ". Negative results become zero, an operation called ReLU. All sensors share the learned "
            "multipliers." + reach
            + width_note,
            parts=[(f"{_plural(int(width_out), 'number')} per sensor", "gcn_hidden", index - 1),
                   ("offset" if bias else "no offset", "gcn_bias")],
            shape=row["out"], learned=int(row["parameters"]), source=_cite(f"Eq. {7 + index}"),
            picture="gcn"))
        index += 1
    flat, fc1, fc2 = layers["Flatten"], layers["FC-1"], layers["FC-2"]
    flat_width = flat["out"].split(",")[-1].strip(" )")
    fc = int(settings["fc_hidden"])
    _, fc_note = _decided("fc_hidden", fc)
    _, ceiling_note = _decided("rul_max", int(settings["rul_max"]))
    steps += [
        _step("Flatten",
              f"Place the {nodes} sensor positions' computed numbers end to end in a single list for each "
              "window. No values are averaged or changed. The next layer can assign a separate learned "
              "weight to every position in this list.",
              parts=[(f"{nodes} x {width_out} = {flat_width}", None)], shape=flat["out"], source=_cite("Eq. 11"),
              picture="flatten"),
        _step("Fully connected",
              (f"Each of the {fc} outputs is" if fc > 1 else "Its one output is")
              + f" a sum of all {flat_width} input numbers multiplied by learned weights, plus a learned "
              "offset. Negative results become zero (ReLU). This combines information across all sensor "
              "positions." + (f" {fc_note}" if fc_note else ""),
              parts=[(_plural(fc, "number"), "fc_hidden")], shape=fc1["out"], learned=int(fc1["parameters"]),
              source=_cite("Eq. 12"), picture="dense"),
        _step("Predicted life fraction",
              f"Combine the {_plural(fc, 'number')} using learned multipliers and an added offset. "
              "The sigmoid, a smooth S-shaped function, maps this result to a fraction between 0 and 1. "
              "It is a fraction of the life ceiling, not a probability of failure. "
              "B counts the windows processed together. The final layer first produces shape (B, 1); "
              "squeeze(-1) removes the one-value axis, leaving shape (B,). The comma marks a shape "
              "with one dimension; no prediction is removed.",
              parts=[("0 to 1 of the life ceiling", None)], shape="(B,)", learned=int(fc2["parameters"]),
              summary="One fraction per window. (B,) means B values in one dimension.",
              source=_cite("Eq. 13"), picture="sigmoid"),
        _step("Scale to cycles",
              "The fraction times the ceiling gives remaining life in flight cycles, so no prediction can exceed "
              f"{settings['rul_max']}. Training divides its labels by the same number." + (
                  f" {ceiling_note}" if ceiling_note else ""),
              parts=[(f"x {settings['rul_max']}", "rul_max")], shape="(B,)",
              source={"kind": "assumption", "ref": "A-13", "text": ""}, picture="scale"),
        _edge("Remaining life, one number per window", "(B,)", icon="output"),
    ]
    return steps
