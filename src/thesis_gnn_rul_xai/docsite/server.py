"""The local web server behind the documentation website.

Read README.md in this folder. It explains everything this file does, every
address it answers on, and why it refuses to serve anything outside the
project directory.
"""

from __future__ import annotations

import contextlib
import ipaddress
import json
import subprocess
import threading
import webbrowser
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..graph.adjacency import adjacency_report
from ..utils.logging_setup import get_logger
from ..utils.paths import project_root
from . import playground
from .catalog import build_catalog, read_document, search
from .figures import figure_names, render_figure
from .markup import pygments_stylesheet, render_markdown
from .notebooks import (
    JupyterProcess,
    NotebookUnavailable,
    jupyter_command,
    start_jupyter,
)
from .notebooks import render_notebook as convert_notebook
from .results import latest_verification, load_entries, load_runs

__all__ = ["DEFAULT_PORT", "DocsHandler", "serve", "site_state"]

logger = get_logger(__name__)

# The port the site listens on. Chosen well away from Jupyter's 8888.
DEFAULT_PORT = 8010

# The port JupyterLab is started on when the site is asked to launch it.
JUPYTER_PORT = 8899

# File extensions the source browser will show. Anything else is refused,
# so the server cannot be turned into a way of reading the whole disk.
_SOURCE_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".toml", ".txt", ".cfg", ".json"})

# The largest file the source viewer will render. Every source file in the
# project is far below this; the raw NASA files are far above it.
_SOURCE_SIZE_LIMIT = 400_000

# The language each of those is highlighted as.
_SOURCE_LANGUAGE = {
    ".py": "python",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".cfg": "ini",
    ".txt": "text",
}

_ASSETS = Path(__file__).parent / "assets"

_CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".svg": "image/svg+xml",
}

# Held between requests so the site can report an already-running JupyterLab
# rather than starting a second one.
_jupyter: dict[str, JupyterProcess] = {}
_jupyter_lock = threading.Lock()

# One queue of playground jobs per project root, created on first use, so two
# runs never train at once and compete for the same processor.
_job_managers: dict[Path, playground.JobManager] = {}

# The largest request body the playground accepts. Its settings are a few
# hundred bytes; anything near this is not a request from the page.
_BODY_LIMIT = 64_000


def job_manager(root: Path) -> playground.JobManager:
    """The playground's job queue for this project."""
    with _jupyter_lock:
        if root not in _job_managers:
            _job_managers[root] = playground.JobManager(root)
        return _job_managers[root]


def _commit(root: Path) -> str:
    """The commit the working tree is on, with -dirty when it has changes."""
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if head.returncode != 0:
            return "not a git repository"
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return head.stdout.strip() + ("-dirty" if status.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git
        return "unknown"


def _module_present(name: str) -> bool:
    """Whether a module could be imported, without importing it."""
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installs
        return False


def site_state(root: Path | None = None) -> dict[str, object]:
    """What this machine can and cannot do, so the site can say so plainly."""
    root = root or project_root()
    raw = root / "data" / "raw"
    present = sorted(p.name for p in raw.glob("*_FD00*.txt")) if raw.is_dir() else []
    with _jupyter_lock:
        running = _jupyter.get("process")
        jupyter_url = running.url if running and running.running else ""
    return {
        "root": str(root),
        "commit": _commit(root),
        "torch": _module_present("torch"),
        "nbconvert": _module_present("nbconvert"),
        "jupyter": jupyter_command() is not None,
        "jupyterUrl": jupyter_url,
        "dataFiles": present,
        "hasFD001": all(
            (raw / f"{part}_FD001.txt").is_file()
            for part in ("train", "test", "RUL")
        ),
        "figures": list(figure_names()),
    }


def _graph_facts() -> dict[str, object]:
    """Everything the graph explorer shows, measured from the transcribed table."""
    from ..data.sensors import SENSORS
    from ..graph.adjacency import build_physics_adjacency, to_networkx

    matrix, symbols = build_physics_adjacency()
    report = adjacency_report(matrix, symbols)
    graph = to_networkx(matrix, symbols)
    return {
        "report": report,
        "neighbours": {node: sorted(graph.neighbors(node)) for node in graph.nodes()},
        "descriptions": {s.symbol: s.description for s in SENSORS},
        "stations": {s.symbol: s.station for s in SENSORS},
        "units": {s.symbol: s.unit for s in SENSORS},
    }


def _model_facts() -> dict[str, object]:
    """Every layer of the network, read off a model this call just built."""
    from ..graph.adjacency import build_physics_adjacency, normalize_adjacency

    try:
        from ..models.estgnn import ESTGNN
    except ImportError:
        return {
            "error": "This page builds the real network to read its layer shapes "
            "and parameter counts, which needs PyTorch. Install it with "
            "'pip install torch' and reload. Nothing else on this site needs it."
        }

    matrix, _ = build_physics_adjacency()
    model = ESTGNN(normalize_adjacency(matrix, method="row"), window_size=1)
    return {
        "layers": model.layer_summary(),
        "parameters": model.count_parameters(),
        "nodes": model.n_nodes,
        "in_features": model.in_features,
        "rul_max": model.rul_max,
    }


class DocsHandler(BaseHTTPRequestHandler):
    """Answers one request: a page, a rendered document, or a figure."""

    server_version = "ThesisDocs/1.0"
    root: Path = Path()

    # -- plumbing ----------------------------------------------------------

    def log_message(self, format: str, *args: object) -> None:
        """Send request logging to the project logger rather than to stderr."""
        logger.debug("%s - %s", self.address_string(), format % args)

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        """Write one response, with the headers every reply here shares."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(body)

    def _json(self, payload: object, status: int = 200) -> None:
        """Write one JSON response."""
        body = json.dumps(payload, default=str).encode("utf-8")
        self._send(body, "application/json; charset=utf-8", status)

    def _error(self, message: str, status: int = 400) -> None:
        """Write one failure, as a message the page can show the reader."""
        self._json({"error": message}, status)

    def _safe_path(self, given: str) -> Path:
        """Resolve a requested path, refusing anything outside the project."""
        target = (self.root / given).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError(f"{given!r} points outside the project directory.")
        return target

    # -- routes ------------------------------------------------------------

    def do_GET(self) -> None:
        """Answer one GET request."""
        parsed = urlparse(self.path)
        route = parsed.path
        query = parse_qs(parsed.query)
        one = lambda key: (query.get(key) or [""])[0]  # noqa: E731

        try:
            if route in ("/", "/index.html"):
                self._serve_asset("shell.html")
            elif route.startswith("/assets/"):
                self._serve_asset(route[len("/assets/") :])
            elif route == "/api/state":
                self._json(site_state(self.root))
            elif route == "/api/catalog":
                self._json(build_catalog(self.root).to_dict())
            elif route == "/api/doc":
                self._serve_doc(one("path"))
            elif route == "/api/source":
                self._serve_source(one("path"))
            elif route == "/api/notebook":
                self._serve_notebook(one("path"))
            elif route == "/api/figure":
                self._send(
                    render_figure(one("name")).encode("utf-8"),
                    "text/html; charset=utf-8",
                )
            elif route == "/api/graph":
                self._json(_graph_facts())
            elif route == "/api/model":
                self._json(_model_facts())
            elif route == "/api/results":
                self._json(
                    {
                        "runs": [asdict(run) for run in load_runs(self.root)],
                        "entries": [asdict(e) for e in load_entries(self.root)],
                        "verification": (
                            asdict(checked)
                            if (checked := latest_verification(self.root))
                            else None
                        ),
                    }
                )
            elif route == "/api/search":
                self._json({"hits": search(one("q"), build_catalog(self.root), self.root)})
            elif route == "/api/jupyter":
                self._json(self._jupyter_state())
            elif route == "/api/playground/options":
                self._json(playground.options())
            elif route == "/api/playground/runs":
                self._json({"runs": playground.list_runs(self.root)})
            elif route == "/api/playground/run":
                self._json(playground.run_detail(self.root, one("name")))
            elif route == "/api/playground/explanation":
                self._json(playground.explanation_detail(self.root, one("run"), one("name")))
            elif route == "/api/playground/compare":
                self._json(playground.compare_runs(self.root, one("first"), one("second")))
            elif route == "/api/playground/jobs":
                self._json({"jobs": job_manager(self.root).snapshot(),
                            "torch": _module_present("torch")})
            else:
                self._error(f"No route at {route!r}.", HTTPStatus.NOT_FOUND)
        except (ValueError, FileNotFoundError) as refused:
            self._error(str(refused), HTTPStatus.NOT_FOUND)
        except Exception as failure:  # pragma: no cover - defensive
            logger.exception("Request for %s failed.", route)
            self._error(f"{failure.__class__.__name__}: {failure}", 500)

    def do_POST(self) -> None:
        """Answer one POST request: starting JupyterLab, or the playground."""
        route = urlparse(self.path).path
        if route.startswith("/api/playground/"):
            self._playground_post(route)
            return
        if route != "/api/jupyter":
            self._error("Nothing here accepts POST.", HTTPStatus.NOT_FOUND)
            return
        try:
            self._json(self._start_jupyter())
        except NotebookUnavailable as refused:
            self._error(str(refused), HTTPStatus.SERVICE_UNAVAILABLE)
        except Exception as failure:  # pragma: no cover - defensive
            logger.exception("Starting JupyterLab failed.")
            self._error(f"{failure.__class__.__name__}: {failure}", 500)

    def _same_origin(self) -> bool:
        """Whether a request that starts work came from this site's own page.

        A browser attaches Origin to every POST, and any web page the reader has
        open could otherwise ask this server to start a training run.
        """
        origin = self.headers.get("Origin")
        same = origin is None or origin == f"http://{self.headers.get('Host', '')}"
        # Started with --host set to something reachable from the network, the
        # site still starts work only for someone sitting at this computer.
        return same and ipaddress.ip_address(self.client_address[0]).is_loopback

    def _body(self) -> dict[str, object]:
        """The JSON object sent with a POST."""
        length = int(self.headers.get("Content-Length") or 0)
        if length > _BODY_LIMIT:
            raise playground.SettingsError("The request is too large.")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise playground.SettingsError("The request is not valid JSON.") from None
        if not isinstance(body, dict):
            raise playground.SettingsError("The request must be a JSON object.")
        return body

    def _playground_post(self, route: str) -> None:
        """Check settings, preview a network, or start and stop jobs."""
        if not self._same_origin():
            self._error("Refused: the request did not come from this site.",
                        HTTPStatus.FORBIDDEN)
            return
        try:
            body = self._body()
            if route == "/api/playground/preview":
                self._json(playground.preview(playground.validate_settings(body)))
            elif route == "/api/playground/train":
                if not _module_present("torch"):
                    raise playground.SettingsError(
                        "Training needs PyTorch, which is not installed here."
                    )
                settings = playground.validate_settings(body.get("settings", {}))
                label = str(body.get("label") or "Training run")[:120]
                self._json(job_manager(self.root).submit_training(settings, label).to_dict())
            elif route == "/api/playground/explain":
                request = playground.validate_explain(body, self.root)
                label = str(body.get("label") or "Explanation")[:120]
                self._json(
                    job_manager(self.root).submit_explanation(request, label).to_dict()
                )
            elif route == "/api/playground/cancel":
                self._json(job_manager(self.root).cancel(str(body.get("id", ""))).to_dict())
            else:
                self._error(f"No route at {route!r}.", HTTPStatus.NOT_FOUND)
        except playground.SettingsError as refused:
            self._error(str(refused), HTTPStatus.BAD_REQUEST)
        except Exception as failure:  # pragma: no cover - defensive
            logger.exception("Playground request %s failed.", route)
            self._error(f"{failure.__class__.__name__}: {failure}", 500)

    # -- what each route does ----------------------------------------------

    def _serve_asset(self, name: str) -> None:
        """Send one of this package's own files: the page, its style, its script."""
        if "/" in name or "\\" in name or name.startswith("."):
            self._error("Bad asset name.", HTTPStatus.NOT_FOUND)
            return
        path = _ASSETS / name
        if not path.is_file():
            self._error(f"No asset called {name!r}.", HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        if name == "app.css":
            body += b"\n\n/* Pygments, for the code blocks */\n"
            body += pygments_stylesheet().encode("utf-8")
        self._send(body, _CONTENT_TYPES.get(path.suffix, "application/octet-stream"))

    def _serve_doc(self, path: str) -> None:
        """Render one Markdown file and send it with its headings."""
        if not path.lower().endswith(".md"):
            raise ValueError(f"{path!r} is not a Markdown document.")
        source = read_document(path, self.root)
        rendered = render_markdown(source, source_path=path)
        catalog = build_catalog(self.root)
        document = catalog.get(path)
        self._json(
            {
                "path": path,
                "title": rendered.title,
                "html": rendered.html,
                "hasMath": rendered.has_math,
                "headings": [asdict(h) for h in rendered.headings],
                "minutes": document.minutes if document else None,
                "words": document.words if document else None,
            }
        )

    def _serve_source(self, path: str) -> None:
        """Show one source file, coloured, without letting the reader roam."""
        target = self._safe_path(path)
        if target.suffix.lower() not in _SOURCE_SUFFIXES:
            raise ValueError(
                f"{path!r} is not one of the file types this site will show "
                f"({', '.join(sorted(_SOURCE_SUFFIXES))})."
            )
        if not target.is_file():
            raise FileNotFoundError(f"No file at {path!r}.")
        size = target.stat().st_size
        if size > _SOURCE_SIZE_LIMIT:
            raise ValueError(
                f"{path!r} is {size / 1e6:.1f} MB. This viewer shows files up to "
                f"{_SOURCE_SIZE_LIMIT // 1000} kB, which covers every source "
                "file in the project. The raw NASA data is meant to be read by "
                "the pipeline, not in a browser."
            )
        language = _SOURCE_LANGUAGE.get(target.suffix.lower(), "text")
        text = target.read_text(encoding="utf-8", errors="replace")
        rendered = render_markdown(
            f"```{language}\n{text}\n```", source_path=path, link_identifiers=False
        )
        self._json(
            {
                "path": path,
                "title": path.rsplit("/", 1)[-1],
                "html": rendered.html,
                "lines": text.count("\n") + 1,
                "headings": [],
            }
        )

    def _serve_notebook(self, path: str) -> None:
        """Convert one notebook to HTML, with the outputs it was saved with."""
        if not path.lower().endswith(".ipynb"):
            raise ValueError(f"{path!r} is not a notebook.")
        try:
            body, styles = convert_notebook(path, self.root)
        except NotebookUnavailable as refused:
            self._error(str(refused), HTTPStatus.SERVICE_UNAVAILABLE)
            return
        self._json(
            {
                "path": path,
                "title": path.rsplit("/", 1)[-1],
                "html": body,
                "css": styles,
            }
        )

    def _jupyter_state(self) -> dict[str, object]:
        """Whether JupyterLab is installed, and whether this site started it."""
        with _jupyter_lock:
            running = _jupyter.get("process")
            alive = bool(running and running.running)
            return {
                "installed": jupyter_command() is not None,
                "running": alive,
                "url": running.url if alive and running else "",
            }

    def _start_jupyter(self) -> dict[str, object]:
        """Start JupyterLab once, and hand back the address it is serving on."""
        with _jupyter_lock:
            running = _jupyter.get("process")
            if running is not None and running.running:
                return {"installed": True, "running": True, "url": running.url}
            started = start_jupyter(JUPYTER_PORT, self.root)
            _jupyter["process"] = started
            logger.info("JupyterLab started at %s", started.url)
            return {"installed": True, "running": True, "url": started.url}


def serve(
    port: int = DEFAULT_PORT,
    *,
    root: Path | None = None,
    host: str = "127.0.0.1",
    open_browser: bool = True,
) -> None:
    """Run the documentation website until the reader stops it with Ctrl+C."""
    root = (root or project_root()).resolve()

    handler = type("BoundDocsHandler", (DocsHandler,), {"root": root})
    server = ThreadingHTTPServer((host, port), handler)
    address = f"http://{host}:{server.server_address[1]}/"

    state = site_state(root)
    logger.info("Serving the documentation of %s at %s", root, address)
    print(f"\n  The documentation site is at  {address}\n")
    print(f"  project      {root}")
    print(f"  commit       {state['commit']}")
    print(f"  FD001 data   {'present' if state['hasFD001'] else 'not in data/raw/'}")
    print(f"  PyTorch      {'available' if state['torch'] else 'not installed'}")
    print(f"  JupyterLab   {'available' if state['jupyter'] else 'not installed'}")
    print("\n  Press Ctrl+C to stop.\n")

    if open_browser:
        threading.Timer(0.6, webbrowser.open, args=(address,)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopping.")
    finally:
        server.shutdown()
        server.server_close()
        with _jupyter_lock:
            running = _jupyter.pop("process", None)
        if running is not None:
            running.stop()
        if root in _job_managers:
            _job_managers[root].stop()
