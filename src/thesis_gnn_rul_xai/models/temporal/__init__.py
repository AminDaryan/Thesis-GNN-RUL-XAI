"""Turns each sensor's window of cycles into the numbers its graph node carries.

Read README.md in this folder. It explains what a temporal stage is, why the
reduction happens here rather than in the preprocessing, and what each option
does.
"""

from __future__ import annotations

from typing import Literal, get_args

import torch
from torch import Tensor, nn

from ...utils.logging_setup import get_logger

__all__ = [
    "LSTMStage",
    "LastCycle",
    "MeanWindow",
    "RawWindow",
    "TemporalStage",
    "TemporalStageName",
    "WindowSummary",
    "build_temporal_stage",
]

logger = get_logger(__name__)

# Every stage the configuration may name. README.md describes each one.
TemporalStageName = Literal[
    "last_cycle", "mean_window", "window_summary", "raw_window", "lstm", "gru"
]


class TemporalStage(nn.Module):
    """One sensor's window of cycles in, the numbers its node carries out."""

    # Left without defaults on purpose: a stage that forgets to set one fails
    # the first time it is used, rather than inheriting a value that happens to
    # be right here and wrong for the next stage.
    window_size: int
    out_features: int

    def forward(self, windows: Tensor) -> Tensor:
        """(batch, nodes, window_size) -> (batch, nodes, out_features)."""
        raise NotImplementedError

    def _check(self, windows: Tensor) -> None:
        """Refuse a batch that is not the window this stage was built for."""
        if windows.dim() != 3:
            raise ValueError(
                "windows must be shaped (batch, nodes, window_size), got "
                f"{tuple(windows.shape)}."
            )
        if windows.shape[2] != self.window_size:
            raise ValueError(
                f"{type(self).__name__} was built for a window of "
                f"{self.window_size} cycle(s), got {windows.shape[2]}."
            )


class LastCycle(TemporalStage):
    """Keep the most recent cycle of each window and drop the rest."""

    def __init__(self, window_size: int) -> None:
        super().__init__()
        if window_size < 1:
            raise ValueError(f"window_size must be at least 1, got {window_size}.")
        self.window_size = int(window_size)
        self.out_features = 1

    # ASSUMPTION A-07: give each sensor node one number, the last cycle of its
    # window. Table 4 prints one number per sensor entering the first graph
    # layer; in the paper that number leaves a per-sensor recurrent stage this
    # project can also build now (see LSTMStage), and this stage is the
    # stand-in used when it is not built. The choice is made here, in the
    # model, rather than in data/windowing/ where it used to be made and
    # looked like a property of the data. Written up in docs/ASSUMPTIONS.md.
    def forward(self, windows: Tensor) -> Tensor:
        self._check(windows)
        return windows[:, :, -1:]


class MeanWindow(TemporalStage):
    """Average each sensor's window into one number."""

    def __init__(self, window_size: int) -> None:
        super().__init__()
        if window_size < 1:
            raise ValueError(f"window_size must be at least 1, got {window_size}.")
        self.window_size = int(window_size)
        self.out_features = 1

    # ASSUMPTION A-07: the other reduction that keeps Table 4's printed shape.
    # Never measured against the last cycle; docs/EXPERIMENT_A-07_mean_window.md
    # is the standing instruction for measuring it.
    def forward(self, windows: Tensor) -> Tensor:
        self._check(windows)
        return windows.mean(dim=2, keepdim=True)


class WindowSummary(TemporalStage):
    """Give each sensor three numbers: its last reading, its average, and its change."""

    def __init__(self, window_size: int) -> None:
        super().__init__()
        if window_size < 2:
            raise ValueError(
                f"window_size must be at least 2 to have a change, got {window_size}."
            )
        self.window_size = int(window_size)
        self.out_features = 3

    # ASSUMPTION A-28: not in the paper. Each sensor node carries its last
    # reading, the average of its window, and how far the best straight line
    # through the window rises from the first cycle to the last, in the same
    # units as the readings rather than per cycle, so all three start on one
    # scale. The paper's nodes carry one number, from a per-sensor LSTM; this
    # is a hand-made stand-in used to measure what the trend is worth.
    def forward(self, windows: Tensor) -> Tensor:
        self._check(windows)
        steps = torch.arange(self.window_size, dtype=windows.dtype, device=windows.device)
        steps = steps - steps.mean()
        slope = (windows * steps).sum(dim=2) / (steps * steps).sum()
        change = slope * (self.window_size - 1)
        return torch.stack([windows[:, :, -1], windows.mean(dim=2), change], dim=2)


class RawWindow(TemporalStage):
    """Hand the graph layer every cycle. This is NOT a temporal encoder.

    README.md in this folder explains why this is kept, gated, and named.
    """

    def __init__(self, window_size: int) -> None:
        super().__init__()
        if window_size < 1:
            raise ValueError(f"window_size must be at least 1, got {window_size}.")
        self.window_size = int(window_size)
        self.out_features = int(window_size)

    def forward(self, windows: Tensor) -> Tensor:
        self._check(windows)
        return windows


class LSTMStage(TemporalStage):
    """One recurrent network per sensor, as the paper's Section 2.2 describes."""

    def __init__(
        self,
        *,
        window_size: int,
        n_nodes: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        out_features: int = 1,
        shared_across_sensors: bool = False,
        cell: Literal["lstm", "gru"] = "lstm",
    ) -> None:
        super().__init__()
        if window_size < 1:
            raise ValueError(f"window_size must be at least 1, got {window_size}.")
        if n_nodes < 1:
            raise ValueError(f"n_nodes must be at least 1, got {n_nodes}.")
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be at least 1, got {hidden_size}.")
        if out_features < 1:
            raise ValueError(f"out_features must be at least 1, got {out_features}.")

        self.window_size = int(window_size)
        self.out_features = int(out_features)
        self.n_nodes = int(n_nodes)
        self.hidden_size = int(hidden_size)
        self.shared_across_sensors = bool(shared_across_sensors)
        self.cell_kind = cell

        # The paper's Section 2.2 gives every sensor its own network: "Each
        # sensor s in {1, 2, ..., n} is associated with a dedicated LSTM
        # network", and Equation 7 indexes the hidden state by sensor. That is
        # what the paper describes, so it is a citation and not a choice.
        # shared_across_sensors=True is a cheaper departure from it, and a
        # number produced with it must say so.
        maker = nn.LSTM if cell == "lstm" else nn.GRU
        count = 1 if self.shared_across_sensors else self.n_nodes
        self.cells = nn.ModuleList(
            maker(
                input_size=1,
                hidden_size=self.hidden_size,
                num_layers=int(num_layers),
                batch_first=True,
            )
            for _ in range(count)
        )
        # ASSUMPTION A-26: take the hidden state of the last cycle and put one
        # linear layer on it to reach the width Table 4 prints. The paper gives
        # no hidden size, no depth and no projection: Table 4 has no recurrent
        # row at all, and Table 6 lists only the learning rate, epochs, batch
        # size, time steps, optimiser and loss. Written up in
        # docs/ASSUMPTIONS.md.
        self.project = nn.Linear(self.hidden_size, self.out_features)

        logger.info(
            "Built %s temporal stage: %d cell(s) of hidden size %d over a "
            "%d-cycle window, projected to %d number(s) per sensor. %d "
            "parameters.",
            cell.upper(),
            count,
            self.hidden_size,
            self.window_size,
            self.out_features,
            sum(p.numel() for p in self.parameters()),
        )

    def forward(self, windows: Tensor) -> Tensor:
        self._check(windows)
        if windows.shape[1] != self.n_nodes:
            raise ValueError(
                f"this stage was built for {self.n_nodes} sensor(s), got "
                f"{windows.shape[1]}."
            )
        if self.shared_across_sensors:
            batch, nodes, width = windows.shape
            flat = windows.reshape(batch * nodes, width, 1)
            last = self.cells[0](flat)[0][:, -1, :]
            return self.project(last).reshape(batch, nodes, self.out_features)

        ends = [
            self.cells[node](windows[:, node, :, None])[0][:, -1, :]
            for node in range(self.n_nodes)
        ]
        return self.project(torch.stack(ends, dim=1))


def build_temporal_stage(
    name: str,
    *,
    window_size: int,
    n_nodes: int,
    hidden_size: int = 32,
    out_features: int = 1,
    shared_across_sensors: bool = False,
) -> TemporalStage:
    """Build the stage named. Every option returns the same kind of object."""
    if name == "last_cycle":
        return LastCycle(window_size)
    if name == "mean_window":
        return MeanWindow(window_size)
    if name == "window_summary":
        return WindowSummary(window_size)
    if name == "raw_window":
        return RawWindow(window_size)
    if name in ("lstm", "gru"):
        return LSTMStage(
            window_size=window_size,
            n_nodes=n_nodes,
            hidden_size=hidden_size,
            out_features=out_features,
            shared_across_sensors=shared_across_sensors,
            cell=name,  # type: ignore[arg-type]
        )
    raise ValueError(
        f"temporal stage must be one of {list(get_args(TemporalStageName))}, "
        f"got {name!r}."
    )
