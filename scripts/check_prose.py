#!/usr/bin/env python3
"""
Catch borrowed words used in the documentation without being explained
=====================================================================

This project is a thesis. Its readers are examiners and future students who
have never seen the code, so working rule 9 requires every term to be spelled
out the first time it appears. That rule has been in CLAUDE.md from the start
and has been broken twice anyway, because the words that fail are the ones that
stop feeling like jargon once you have been reading the code for an hour:
*buffer*, *broadcast*, *state dictionary*, *self-loop*. One of them,
"an unrescaled table is rejected", was not even a real word and survived in
models/estgnn.py for weeks.

Restating the rule does not work. This script is the floor underneath it: it
holds a list of borrowed terms and reports every place one is used for the
first time in a document without an explanation beside it.

What it cannot do
-----------------
It cannot judge whether writing is clear. It matches a fixed list of words
against a small set of explanatory cues, so it will miss jargon nobody has
added to WORDS, and it will occasionally flag a sentence that is perfectly
clear. It is a gate against one specific, repeated failure — not a substitute
for reading the prose as someone who does not know the project.

Examples
--------
Check everything and print the report::

    python scripts/check_prose.py

Only the count, for use in a script::

    python scripts/check_prose.py --quiet

Silencing a false positive
--------------------------
Put a waiver on the line itself, naming the word::

    The buffer here is an audio buffer.   <!-- prose-ok: buffer -->
    # a genuine ring buffer               # prose-ok: buffer

Exit status is 1 if anything is unexplained, so this can be a gate.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thesis_gnn_rul_xai.utils.paths import project_root

# Terms borrowed from PyTorch, graph theory or statistics that mean nothing to
# a reader who knows Python but has never used them. The note beside each is
# printed in the report so the failure explains itself.
WORDS: dict[str, str] = {
    "unrescaled": "not a word outside this repository",
    "buffer": "PyTorch's meaning, not the everyday one",
    "broadcast": "PyTorch's array-shape meaning",
    "broadcasting": "PyTorch's array-shape meaning",
    "state dictionary": "PyTorch's name for a saved model",
    "state dict": "PyTorch's name for a saved model",
    "logits": "a machine-learning term",
    "self-loop": "graph theory: an edge from a node to itself",
    "degree matrix": "graph theory",
    "non-Euclidean": "mathematics",
    "ablation": "research method: removing a part to see what it was worth",
    "hyperparameter": "a setting chosen before training, not learned",
    "epoch": "one pass over the training data",
    "z-score": "statistics: how many standard deviations from the mean",
    "Glorot": "a person's surname used as the name of a method",
    "Xavier": "the same person's given name, same method",
    "over-smoothing": "a graph-network failure mode",
    "readout": "a graph-network term",
    "message passing": "a graph-network term",
}

# Phrases that mark an explanation standing next to the word. Deliberately
# generous: the goal is to catch bare uses, not to grade the prose.
CUES = (
    " is ", " are ", " means ", " meaning ", " called ", " that is ",
    " which is ", " which means ", " — ", " -- ", ":", "i.e.", "namely",
    "in other words", " refers to ", " stands for ", " named after ",
    "so that ", "without it", "without that", " lets ", " so it ", " so a ",
)

# Emphasis marks break the cue match: "zero *is* the mean" contains no " is ".
EMPHASIS = re.compile(r"[*_`]")

WAIVER = re.compile(r"prose-ok:\s*([\w\- ]+)")


@dataclass(frozen=True)
class Finding:
    """One borrowed word used without an explanation beside it."""

    path: str
    line: int
    word: str
    text: str


def documents(root: Path) -> list[Path]:
    """Every file whose prose this project promises to keep readable."""
    found: list[Path] = []
    found += sorted(root.glob("src/thesis_gnn_rul_xai/**/README.md"))
    found += sorted(root.glob("src/thesis_gnn_rul_xai/**/*.py"))
    found += sorted(root.glob("docs/learn/*.md"))
    found += sorted(root.glob("scripts/*.py"))
    return [p for p in found if "__pycache__" not in p.parts]


def _explained(window: str, word: str) -> bool:
    """Whether an explanation of this word stands within the given lines."""
    lowered = EMPHASIS.sub("", window).lower()
    if word.lower() not in lowered:
        return False
    return any(cue in lowered for cue in CUES)


def _waived(window: str, word: str) -> bool:
    """Whether a `prose-ok` note nearby excuses this use."""
    return any(
        word.lower() in match.group(1).lower() for match in WAIVER.finditer(window)
    )


def check_file(path: Path, root: Path) -> list[Finding]:
    """Report the first unexplained use of each borrowed word in one file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    findings: list[Finding] = []

    for word, _ in WORDS.items():
        pattern = re.compile(rf"\b{re.escape(word)}", re.IGNORECASE)
        for number, line in enumerate(lines, 1):
            if not pattern.search(line):
                continue
            # A definition may sit on the line before or after, so read three.
            window = "\n".join(lines[max(0, number - 2): number + 2])
            if not _waived(window, word) and not _explained(window, word):
                findings.append(
                    Finding(
                        path=str(path.relative_to(root)).replace("\\", "/"),
                        line=number,
                        word=word,
                        text=line.strip()[:88],
                    )
                )
            break  # only the first use in a file has to introduce the word
    return findings


def build_parser() -> argparse.ArgumentParser:
    """Every option this script takes."""
    parser = argparse.ArgumentParser(
        description="Find borrowed words used without an explanation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--quiet", action="store_true", help="print only the final count."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Check every document and report what is unexplained."""
    arguments = build_parser().parse_args(argv)
    # Windows consoles default to a codepage that cannot print an em dash.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = project_root()

    findings: list[Finding] = []
    for path in documents(root):
        findings.extend(check_file(path, root))

    if not arguments.quiet:
        if findings:
            print("Borrowed words used without an explanation beside them:\n")
            for item in sorted(findings, key=lambda f: (f.path, f.line)):
                print(f"  {item.path}:{item.line}")
                print(f"      {item.word} — {WORDS[item.word]}")
                print(f"      {item.text}\n")
            print(
                "Explain each one where it stands, or add a `prose-ok: <word>` "
                "note on the line if it is a false alarm.\n"
            )
        else:
            print("Every borrowed word is explained where it first appears.\n")

    checked = len(documents(root))
    print(f"{checked} document(s) checked, {len(findings)} unexplained use(s).")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
