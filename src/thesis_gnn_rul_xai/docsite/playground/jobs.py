"""The queue that runs one playground script at a time, follows its progress and stops it."""

from __future__ import annotations

import calendar
import contextlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ...utils.logging_setup import get_logger
from .commands import explain_arguments, preprocessing_settings, train_arguments
from .runs import _clean
from .settings import _METHOD_NAMES, _STAGE_NAMES, SettingsError

logger = get_logger(__name__)


# Where the playground keeps the files of the jobs it starts: the settings it
# wrote, the command's printed output and its progress. Inside reports/, which
# is not committed, and hidden so the run listing never mistakes it for a run.
JOBS_FOLDER = Path("reports") / ".playground"

# -- jobs -------------------------------------------------------------------


# The states a job can no longer leave.
_ENDED = ("done", "failed", "cancelled", "interrupted")

# What a job is called when its folder holds nothing to name it by.
_UNNAMED = {"train": "Training run", "explain": "Explanation"}


def job_label(settings: Mapping[str, Any]) -> str:
    """The name a training job is listed under: what it trains, on what, how many times."""
    parts = [str(settings.get("subset") or "")]
    temporal = settings.get("temporal")
    if temporal is not None:
        stage = _STAGE_NAMES.get(str(temporal), str(temporal))
        if temporal in ("lstm", "gru") and settings.get("temporal_hidden") is not None:
            stage += f" {settings['temporal_hidden']}"
        parts.append(stage)
    graph = str(settings.get("graph") or "physics")
    rescaling = settings.get("adjacency_normalization")
    parts.append(f"{graph} graph" if rescaling in (None, "row") else f"{graph} · {rescaling} graph")
    parts.append("best pass" if float(settings.get("validation_fraction") or 0.0) > 0
                 else "final pass")
    seeds = int(settings.get("seeds") or 0)
    parts.append(f"{seeds} seed{'' if seeds == 1 else 's'}")
    return " · ".join(part for part in parts if part)


def explanation_label(request: Mapping[str, Any]) -> str:
    """The name an explanation job is listed under: which method, on what, with which seed's weights."""
    method = request.get("method")
    parts = [_METHOD_NAMES.get(str(method), str(method or "Explanation"))]
    if request.get("scope") == "test":
        parts.append("test engines")
    elif request.get("scope") == "engine":
        engine = request.get("engine")
        parts.append("longest engine" if engine is None else f"engine {engine}")
    if request.get("seed") is not None:
        parts.append(f"seed {request['seed']}")
    return " · ".join(parts)


def _plain_name(name: Any) -> bool:
    """Whether a name read from a file is one plain folder name, with no way out of its folder."""
    return (isinstance(name, str) and bool(name) and not name.startswith(".")
            and ":" not in name and Path(name).name == name)


def _explain_request(folder: Path, command: str, progress: Mapping[str, Any]) -> dict[str, Any]:
    """What an explanation job asked for, from its command, and the engine it chose once it finished."""
    request: dict[str, Any] = {"method": progress.get("method"), "scope": progress.get("scope")}
    words = command.split()
    for flag, key in (("--method", "method"), ("--scope", "scope"),
                      ("--seed", "seed"), ("--engine", "engine")):
        if flag in words[:-1]:
            value = words[words.index(flag) + 1]
            request[key] = int(value) if value.isdigit() else value
    # With no engine asked for, the script picks the longest, and only its
    # stored summary says which one that was. The jobs folder sits inside
    # reports/, beside the run the explanation was written into.
    result = progress.get("result") or {}
    if _plain_name(result.get("run")) and _plain_name(result.get("directory")):
        stored = (folder.parent.parent / result["run"] / "explanations"
                  / result["directory"] / "summary.json")
        try:
            summary = json.loads(stored.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = {}
        if isinstance(summary, dict):
            request.update({key: summary[key] for key in ("method", "scope", "seed")
                            if summary.get(key) is not None})
            if summary.get("unit") is not None:
                request["engine"] = summary["unit"]
    return request


def _job_name(folder: Path, kind: str, command: str, progress: Mapping[str, Any],
              fallback: str) -> str:
    """The name a job is listed under, built the same way for this server's jobs and for folders read back."""
    if kind == "train":
        try:
            settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            settings = None
        if isinstance(settings, dict) and settings:
            return job_label(settings)
    elif kind == "explain":
        request = _explain_request(folder, command, progress)
        if request.get("method"):
            return explanation_label(request)
    return fallback


def read_progress(folder: Path) -> dict[str, Any]:
    """What a job's progress file says so far, summarised for the page."""
    path = folder / "progress.jsonl"
    if not path.is_file():
        return {}
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # the last line may be half-written
    summary: dict[str, Any] = {"seeds_done": [], "curve": []}
    for event in events:
        kind = event.get("event")
        if kind == "start":
            summary.update({k: v for k, v in event.items() if k != "event"})
        elif kind == "seed":
            summary["seed"], summary["seed_index"], summary["curve"] = (
                event.get("seed"), event.get("index"), [])
        elif kind == "epoch":
            summary["epoch"] = event.get("epoch")
            summary["curve"].append({
                "epoch": event.get("epoch"),
                "train_loss": event.get("train_loss"),
                "validation_rmse": event.get("validation_rmse_cycles"),
            })
        elif kind == "seed_done":
            summary["seeds_done"].append({k: v for k, v in event.items() if k != "event"})
        elif kind == "window":
            summary.update(done=event.get("done"), total=event.get("total"),
                           seconds=event.get("seconds"))
        elif kind == "done":
            summary["result"] = {k: v for k, v in event.items() if k != "event"}
    return summary


def _process_alive(pid: Any) -> bool:
    """Whether a process with this number is still running on this machine."""
    # The number is read back from a file, so it may be anything at all. A
    # damaged one means "nothing is running", never an error raised out of
    # the listing that every job on the page is drawn from.
    try:
        number = int(pid)
    except (TypeError, ValueError):
        return False
    if number <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        # 0x1000 asks only "does this exist and how did it end", which needs no
        # right over the process; 259 is what Windows reports while it runs.
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, number)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            asked = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(asked) and code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(number, 0)
    except OSError:
        return False
    return True


def _log_tail(folder: Path, lines: int = 25) -> str:
    """The last lines a command printed, which is where a failure explains itself."""
    path = folder / "output.txt"
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


@dataclass
class Job:
    """One command the playground started, or will start when the one before it ends."""

    id: str
    kind: str
    label: str
    arguments: list[str]
    display: str
    folder: Path
    created: float = field(default_factory=time.time)
    status: str = "queued"
    started: float | None = None
    finished: float | None = None
    returncode: int | None = None
    process: subprocess.Popen[bytes] | None = field(default=None, repr=False)

    def progress(self) -> dict[str, Any]:
        """What the job's progress file says so far, summarised for the page."""
        return read_progress(self.folder)

    def log_tail(self, lines: int = 25) -> str:
        """The last lines the command printed, which is where a failure explains itself."""
        return _log_tail(self.folder, lines)

    def record(self) -> None:
        """Write what the page shows about this job beside its files, so a later server can list it."""
        summary = {
            "id": self.id, "kind": self.kind, "label": self.label, "command": self.display,
            "created": self.created, "status": self.status, "started": self.started,
            "finished": self.finished, "returncode": self.returncode,
            # So that a later server can tell a job still running under the
            # server that started it from one whose server died with it.
            "pid": self.process.pid if self.process is not None else None,
        }
        # If the folder was removed meanwhile, the listing is the loss, not the run.
        with contextlib.suppress(OSError):
            (self.folder / "job.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        """The job as the page shows it."""
        now = time.time()
        progress = self.progress()
        return _clean({
            "id": self.id, "kind": self.kind,
            "label": _job_name(self.folder, self.kind, self.display, progress, self.label),
            "status": self.status,
            "command": self.display, "created": self.created, "started": self.started,
            "finished": self.finished, "returncode": self.returncode,
            "elapsed": (self.finished or now) - self.started if self.started else None,
            "progress": progress,
            "log": self.log_tail() if self.status in ("failed", "cancelled") else "",
            "from_disk": False,
        })


def _stamp_to_time(stamp: str) -> float | None:
    """The moment a job folder's name records, as seconds since the epoch."""
    try:
        return float(calendar.timegm(time.strptime(stamp, "%Y%m%dT%H%M%S")))
    except ValueError:
        return None


def _describe_job_folder(path: Path) -> dict[str, Any] | None:
    """One job an earlier server left behind, read from its folder."""
    parts = path.name.split("_")
    if len(parts) < 3:
        return None
    recorded: dict[str, Any] = {}
    if (path / "job.json").is_file():
        try:
            recorded = json.loads((path / "job.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            recorded = {}
    progress = read_progress(path)
    kind = str(recorded.get("kind") or parts[1])
    status = str(recorded.get("status") or "")
    if status not in _ENDED:
        if "result" in progress:
            status = "done"
        elif _process_alive(recorded.get("pid")):
            # The server that started it is still running it. Calling this
            # ended would offer it to "remove all ended", which would delete
            # the folder the command is at that moment writing into.
            status = "running"
        else:
            # A job still running when its server stopped never wrote its end.
            status = "interrupted"
    started, finished = recorded.get("started"), recorded.get("finished")
    command = str(recorded.get("command") or "")
    return _clean({
        "id": path.name,
        "kind": kind,
        # Rebuilt rather than read from job.json, so a folder an older server
        # wrote is named in the same words as a job this server is running.
        "label": _job_name(path, kind, command, progress, str(
            recorded.get("label") or _UNNAMED.get(kind, f"{kind} job"))),
        "status": status,
        "command": command,
        "created": recorded.get("created") or _stamp_to_time(parts[0]) or path.stat().st_mtime,
        "started": started,
        "finished": finished,
        "elapsed": (finished - started) if started and finished else None,
        "progress": progress,
        "log": _log_tail(path) if status == "failed" else "",
        "from_disk": True,
    })


def job_history(root: Path) -> list[dict[str, Any]]:
    """Every job any server left in reports/.playground/, read from the folders."""
    folder = root / JOBS_FOLDER
    if not folder.is_dir():
        return []
    return [described for path in sorted(folder.iterdir())
            if path.is_dir() and (described := _describe_job_folder(path)) is not None]


class JobManager:
    """Runs the playground's commands one at a time, in the order they were asked for."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.jobs: dict[str, Job] = {}
        self.queue: list[str] = []
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stopping = False
        self.worker: threading.Thread | None = None

    def _ensure_worker(self) -> None:
        if self.worker is None or not self.worker.is_alive():
            self.worker = threading.Thread(target=self._work, name="playground-jobs", daemon=True)
            self.worker.start()

    def _new_folder(self, kind: str) -> tuple[str, Path]:
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        identifier = f"{stamp}_{kind}_{secrets.token_hex(3)}"
        folder = self.root / JOBS_FOLDER / identifier
        folder.mkdir(parents=True, exist_ok=True)
        return identifier, folder

    def submit_training(self, settings: Mapping[str, Any]) -> Job:
        """Queue one training run of these settings, under the name job_label gives it."""
        identifier, folder = self._new_folder("train")
        config_path = folder / "preprocessing_config.yaml"
        config_path.write_text(
            yaml.safe_dump(preprocessing_settings(settings), sort_keys=False), encoding="utf-8"
        )
        (folder / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        relative = config_path.relative_to(self.root)
        job = Job(
            id=identifier, kind="train", label=job_label(settings), folder=folder,
            arguments=train_arguments(settings, config_path, folder / "progress.jsonl"),
            display=" ".join(train_arguments(settings, relative, None, display=True)),
        )
        return self._enqueue(job)

    def submit_explanation(self, request: Mapping[str, Any]) -> Job:
        """Queue one explanation of a stored run, under the name explanation_label gives it."""
        identifier, folder = self._new_folder("explain")
        job = Job(
            id=identifier, kind="explain", label=explanation_label(request), folder=folder,
            arguments=explain_arguments(request, folder / "progress.jsonl"),
            display=" ".join(explain_arguments(request, None, display=True)),
        )
        return self._enqueue(job)

    def submit_command(self, kind: str, label: str, arguments: list[str]) -> Job:
        """Queue any command. Only the tests use this directly."""
        identifier, folder = self._new_folder(kind)
        job = Job(id=identifier, kind=kind, label=label, folder=folder,
                  arguments=list(arguments), display=" ".join(arguments))
        return self._enqueue(job)

    def _enqueue(self, job: Job) -> Job:
        with self.lock:
            self.jobs[job.id] = job
            self.queue.append(job.id)
            # Written under the lock: the worker can take this job the moment
            # the lock is released, and its own record must land after this one.
            job.record()
        self._ensure_worker()
        self.wake.set()
        return job

    def cancel(self, identifier: str) -> Job:
        """Remove a queued job, or stop a running one. A stopped run writes nothing."""
        with self.lock:
            job = self.jobs.get(identifier)
            if job is None:
                raise SettingsError(f"There is no job called {identifier!r}.")
            if job.status == "queued":
                self.queue.remove(identifier)
                job.status, job.finished = "cancelled", time.time()
                job.record()
            elif job.status == "running":
                job.status = "cancelling"
                # A job is marked running a moment before its process exists.
                # Asked to stop in that moment, there is nothing to end yet, so
                # _run sees "cancelling" and never starts it.
                if job.process is not None:
                    job.process.terminate()
        return job

    def forget(self, identifier: str = "", *, all_finished: bool = False) -> list[str]:
        """Drop finished jobs from the list and delete their folders. A running or queued job stays."""
        if all_finished:
            targets = [item["id"] for item in self.snapshot() if item["status"] in _ENDED]
        else:
            targets = [identifier]
        forgotten = []
        jobs_folder = (self.root / JOBS_FOLDER).resolve()
        for target in targets:
            # The name has to be one plain folder name inside the jobs folder.
            # Looking for separators is not enough: a Windows name carrying a
            # drive, "C:x", has none, and joining it throws the base away.
            folder = (self.root / JOBS_FOLDER / target).resolve()
            if not target or target.startswith(".") or folder.parent != jobs_folder:
                raise SettingsError(f"{target!r} is not the name of a job.")
            with self.lock:
                job = self.jobs.get(target)
                if job is not None:
                    if job.status not in _ENDED:
                        raise SettingsError(
                            f"Job {target!r} is {job.status}; stop it before removing it."
                        )
                    del self.jobs[target]
            if job is None:
                # Not this server's job. It may still belong to a running one.
                described = _describe_job_folder(folder)
                if described is not None and described["status"] not in _ENDED:
                    raise SettingsError(
                        f"Job {target!r} is {described['status']}; stop it before removing it."
                    )
            if folder.is_dir():
                # The job's own folder only: what it printed and its progress.
                # The run it produced lives elsewhere in reports/ and stays.
                shutil.rmtree(folder, ignore_errors=True)
            elif job is None:
                raise SettingsError(f"There is no job called {target!r}.")
            forgotten.append(target)
        return forgotten

    def snapshot(self) -> list[dict[str, Any]]:
        """Every job this server was asked for, then those earlier servers left on disk, newest first."""
        with self.lock:
            jobs = list(self.jobs.values())
        listed = [job.to_dict() for job in jobs]
        known = {job.id for job in jobs}
        listed += [item for item in job_history(self.root) if item["id"] not in known]
        listed.sort(key=lambda item: item["created"], reverse=True)
        return listed

    def get(self, identifier: str) -> Job | None:
        with self.lock:
            return self.jobs.get(identifier)

    def stop(self) -> None:
        """Stop the running job and forget the queue. Called when the server stops."""
        self.stopping = True
        with self.lock:
            self.queue.clear()
            running = [job for job in self.jobs.values() if job.process is not None
                       and job.process.poll() is None]
        for job in running:
            job.process.terminate()  # type: ignore[union-attr]
        self.wake.set()

    def _work(self) -> None:
        while not self.stopping:
            with self.lock:
                identifier = self.queue.pop(0) if self.queue else None
                job = self.jobs.get(identifier) if identifier else None
                if job is not None:
                    job.status, job.started = "running", time.time()
            if job is None:
                self.wake.clear()
                self.wake.wait(timeout=1.0)
                continue
            self._run(job)

    def _run(self, job: Job) -> None:
        logger.info("Playground job %s: %s", job.id, job.display)
        with (job.folder / "output.txt").open("wb") as output:
            try:
                # Started and published under the lock, so a stop asked for
                # before this point is seen here, and one asked for after it
                # finds the process to end.
                with self.lock:
                    if job.status == "cancelling":
                        job.returncode = None
                    else:
                        job.process = subprocess.Popen(
                            job.arguments, cwd=self.root, stdout=output,
                            stderr=subprocess.STDOUT,
                            env={**_environment(), "PYTHONIOENCODING": "utf-8"},
                        )
                if job.process is not None:
                    # Recorded now the process exists, so its number reaches
                    # the file while the command is still running.
                    job.record()
                    job.returncode = job.process.wait()
            except OSError as failure:
                output.write(f"Could not start the command: {failure}\n".encode())
                job.returncode = -1
        with self.lock:
            job.finished = time.time()
            if job.status == "cancelling":
                job.status = "cancelled"
            else:
                job.status = "done" if job.returncode == 0 else "failed"
            job.process = None
        job.record()


def _environment() -> dict[str, str]:
    import os

    return dict(os.environ)
