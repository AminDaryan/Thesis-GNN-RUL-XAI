#!/usr/bin/env python3
"""
Open the documentation in a browser
===================================

Starts a small web server on this machine that shows every README, every
assumption, both notebooks and a set of figures drawn from this repository as
it stands. Nothing is uploaded and nothing is published; the server listens on
the loopback address only, so it is reachable from this computer alone.

The documentation is read from disk on every request, so a page always shows
what the file says now. There is no build step and no generated copy that could
fall out of date.

Examples
--------
The usual way, which opens a browser window::

    python scripts/docs_ui.py

A different port, without opening a browser::

    python scripts/docs_ui.py --port 8123 --no-browser

The site can start JupyterLab for you, so the notebooks can be run rather than
only read. That needs JupyterLab installed::

    pip install -e .[notebook]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thesis_gnn_rul_xai.docsite.server import DEFAULT_PORT, serve
from thesis_gnn_rul_xai.utils.logging_setup import configure_logging
from thesis_gnn_rul_xai.utils.paths import project_root


def build_parser() -> argparse.ArgumentParser:
    """Every option this script takes."""
    parser = argparse.ArgumentParser(
        description="Serve this project's documentation in a browser.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"which port to listen on (default: {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "which address to listen on. The default reaches this computer "
            "only. Change it only if you understand that any other value "
            "exposes the whole project directory to your network."
        ),
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="do not open a browser window; just print the address.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="log every request.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Start the server and keep it running until interrupted."""
    arguments = build_parser().parse_args(argv)
    configure_logging(logging.DEBUG if arguments.verbose else logging.INFO)

    if arguments.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"\n  Warning: listening on {arguments.host} makes every file in\n"
            f"  {project_root()}\n"
            "  readable by anything that can reach this machine on that "
            "address.\n"
        )

    try:
        serve(
            port=arguments.port,
            host=arguments.host,
            open_browser=not arguments.no_browser,
        )
    except OSError as error:
        print(
            f"\n  Could not listen on {arguments.host}:{arguments.port} — {error}\n"
            "  Something else is probably using that port. Try another one:\n"
            f"      python scripts/docs_ui.py --port {arguments.port + 1}\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
