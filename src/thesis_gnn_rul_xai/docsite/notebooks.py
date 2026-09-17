"""Shows a notebook in the browser, and starts JupyterLab when asked to.

Read README.md in this folder. It explains everything this file does, and the
difference between reading a notebook here and running one in JupyterLab.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ..utils.paths import project_root

__all__ = [
    "JupyterProcess",
    "NotebookUnavailable",
    "jupyter_command",
    "render_notebook",
    "start_jupyter",
]

# How long to wait for JupyterLab to print the address it is serving on.
_STARTUP_TIMEOUT = 40.0

# The line JupyterLab prints once it is ready, which carries its one-time token.
_ADDRESS_MARKERS = ("http://localhost:", "http://127.0.0.1:")


class NotebookUnavailable(Exception):
    """Raised when a notebook cannot be shown, with what to do about it."""


@dataclass
class JupyterProcess:
    """A running JupyterLab: where it is serving and how it was started."""

    url: str
    port: int
    command: list[str]
    process: subprocess.Popen[str]

    @property
    def running(self) -> bool:
        """Whether the server is still up."""
        return self.process.poll() is None

    def stop(self) -> None:
        """Ask the server to shut down, and give it a moment to do so."""
        if self.running:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - stubborn child
                self.process.kill()


def render_notebook(path: str, root: Path | None = None) -> str:
    """Turn one notebook into the HTML of its cells, with the outputs it was saved with."""
    root = (root or project_root()).resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise NotebookUnavailable(f"{path!r} points outside the project directory.")
    if not target.is_file():
        raise NotebookUnavailable(f"There is no notebook at {path!r}.")

    try:
        import nbformat
        from nbconvert import HTMLExporter
    except ImportError as missing:
        raise NotebookUnavailable(
            "Showing a notebook needs nbconvert and nbformat, which are not "
            "installed. Install them with 'pip install -e .[docs]' and reload "
            "this page."
        ) from missing

    notebook = nbformat.read(target, as_version=4)
    exporter = HTMLExporter(template_name="basic")
    exporter.exclude_input_prompt = False
    # nbconvert also hands back a stylesheet, but it colours code in light
    # colours only; the site's own stylesheet colours notebooks in both themes.
    body, _ = exporter.from_notebook_node(notebook)
    return body


def jupyter_command() -> list[str] | None:
    """The command that would start JupyterLab, or nothing if it is absent."""
    for candidate in (
        [sys.executable, "-m", "jupyterlab"],
        [sys.executable, "-m", "jupyter", "lab"],
    ):
        module = candidate[2] if candidate[1] == "-m" else ""
        if _module_present(module if module != "jupyter" else "jupyter_server"):
            return candidate
    executable = shutil.which("jupyter-lab") or shutil.which("jupyter")
    if executable:
        return [executable, "lab"] if executable.endswith("jupyter") else [executable]
    return None


def _module_present(name: str) -> bool:
    """Whether a module can be imported, without importing it."""
    import importlib.util

    if not name:
        return False
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installs
        return False


def start_jupyter(port: int = 8899, root: Path | None = None) -> JupyterProcess:
    """Start JupyterLab on the notebooks folder and return where it is serving."""
    command = jupyter_command()
    if command is None:
        # No restart is asked for: jupyter_command() looks again on every
        # request, and a package installed while the site runs is found.
        raise NotebookUnavailable(
            "JupyterLab is not installed in this environment. To run the notebooks, "
            "install it with 'pip install -e .[notebook]'."
        )

    root = root or project_root()
    arguments = [
        *command,
        f"--notebook-dir={root}",
        f"--port={port}",
        "--no-browser",
        "--ServerApp.open_browser=False",
    ]
    process = subprocess.Popen(
        arguments,
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    url = _wait_for_address(process, port)
    return JupyterProcess(url=url, port=port, command=arguments, process=process)


def _wait_for_address(process: subprocess.Popen[str], port: int) -> str:
    """Read the server's output until it prints the address it is serving on."""
    found: list[str] = []
    lines: list[str] = []

    def read() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line)
            if found:
                continue
            for marker in _ADDRESS_MARKERS:
                position = line.find(marker)
                if position >= 0:
                    found.append(line[position:].strip())
                    break

    reader = threading.Thread(target=read, daemon=True)
    reader.start()

    deadline = time.monotonic() + _STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if found:
            return found[0]
        if process.poll() is not None:
            raise NotebookUnavailable(
                "JupyterLab stopped immediately. It printed:\n\n"
                + "".join(lines[-25:]).strip()
            )
        time.sleep(0.2)

    process.terminate()
    raise NotebookUnavailable(
        f"JupyterLab did not report an address within {_STARTUP_TIMEOUT:.0f} "
        f"seconds. It may still be starting on port {port}. It printed:\n\n"
        + "".join(lines[-25:]).strip()
    )
