# The cited papers

This folder holds a copy of every paper this project cites that a copy could
be obtained for: 39 PDFs and NASA's description of the dataset, 40 files in
all. The reading list, with what each paper is used for and which version each
copy is, is [`docs/REFERENCES.md`](../REFERENCES.md).

**On a fresh clone this folder holds only this file and `manifest.json`.** The
copies are kept on the author's computer and are not committed, because many
of them are publishers' files that may not be passed on (`.gitignore` says
the same).

To get them back:

```bash
python scripts/fetch_references.py
```

It reads `manifest.json` and, for each paper, reports it as present, downloads
it if it is missing and free to download, or says where to obtain it. A
publisher's copy usually has to come through a university library. Add
`--check` to report without downloading.

**Why the checksum matters.** Page, section and equation numbers in this
project refer to one particular version of each paper, and different versions
of the same paper number things differently: the C-MAPSS paper's scoring
function is Eq. (11) in the copy here and Eq. (9) in NASA's earlier draft.
`manifest.json` stores each file's SHA-256 checksum, a 64-character
fingerprint that changes completely if a single byte of the file changes. The
fetch script reports any file whose checksum does not match as a different
version, and `tests/test_references.py` does the same for every copy present.

**Adding a paper.** Put the PDF here, named `firstauthor_year_short_title.pdf`;
add its entry to `manifest.json` with the checksum the script's `sha256()`
prints for it; and add its citation, with a "Copy:" line linking the file, to
`docs/REFERENCES.md`. Mark it `"free": true` only after downloading its
`source` twice and getting the same checksum both times, since that is what
lets the fetch script restore exactly this version.
