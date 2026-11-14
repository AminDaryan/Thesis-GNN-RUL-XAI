#!/usr/bin/env python3
"""
Fetch the cited papers
======================

Every paper this project cites has a copy in ``docs/references/`` on the
author's computer. Those copies are not committed, because many of them are
publishers' files that may not be passed on. ``docs/references/manifest.json``,
which is committed, records for each one the file name, where it came from,
which version it is, and its SHA-256 checksum: a 64-character fingerprint that
changes completely if a single byte of the file changes, so a matching
checksum means the file is the very one the project's citations were checked
against.

This script reads that list and, for each paper:

- reports it as present if the file is there and its checksum matches;
- warns if the file is there but its checksum does not match, which means it
  is a different version and its page or section numbers may not be the ones
  the documents cite;
- downloads it if it is missing and free to download, then checks the
  checksum;
- otherwise says where to obtain it, which for a publisher's copy is usually a
  university library.

Examples
--------
Download whatever is missing and free::

    python scripts/fetch_references.py

Only report, download nothing::

    python scripts/fetch_references.py --check

Exit status is 1 if any file is missing or does not match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thesis_gnn_rul_xai.utils.paths import project_root

# Some publishers refuse a request that does not name a browser-like client.
_USER_AGENT = "Mozilla/5.0 (thesis-gnn-rul-xai reference fetcher)"


def references_dir(root: Path | None = None) -> Path:
    """The folder holding the cited papers."""
    return (root or project_root()) / "docs" / "references"


def load_manifest(root: Path | None = None) -> list[dict[str, object]]:
    """Every entry of the manifest, in the order it lists them."""
    path = references_dir(root) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))["papers"]


def sha256(path: Path) -> str:
    """The SHA-256 checksum of a file, as 64 hexadecimal characters."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    """Save the file at an address to a path, refusing anything that is not a PDF."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read()
    if not body.startswith(b"%PDF"):
        raise ValueError(f"{url} did not return a PDF.")
    target.write_bytes(body)


def main(argv: list[str] | None = None) -> int:
    """Check every cited paper and fetch the free ones that are missing."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="report only; download nothing")
    args = parser.parse_args(argv)

    folder = references_dir()
    failures = 0
    for paper in load_manifest():
        target = folder / str(paper["file"])
        if target.is_file():
            if sha256(target) == paper["sha256"]:
                print(f"present    {target.name}")
            else:
                print(f"DIFFERENT  {target.name}: not the version the citations were checked against")
                failures += 1
            continue
        if not paper["free"] or args.check:
            print(f"missing    {target.name}: {paper['obtain']}")
            failures += 1
            continue
        try:
            download(str(paper["source"]), target)
        except (OSError, ValueError) as refused:
            print(f"FAILED     {target.name}: {refused}")
            failures += 1
            continue
        if sha256(target) != paper["sha256"]:
            print(f"DIFFERENT  {target.name}: downloaded, but the source now serves another version")
            failures += 1
        else:
            print(f"downloaded {target.name}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
