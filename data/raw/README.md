# Raw data directory

Put the official NASA C-MAPSS files here. For the FD001 sub-dataset you need
exactly three files:

```
data/raw/train_FD001.txt
data/raw/test_FD001.txt
data/raw/RUL_FD001.txt
```

The files are **not** included in this repository. NASA distributes them under
its own terms, and a research repository should point to the original source
rather than redistribute a copy that may drift out of date.

See the "Getting the data" section of the main `README.md` for where to
download them, or run:

```bash
python scripts/prepare_data.py --check
```

which reports which files are present, which are missing, and whether the ones
present have the expected number of rows.

All four sub-datasets (FD001 to FD004) can live here together. Only the three
files for the sub-dataset named in the configuration are read on any one run.

If you would rather keep a single copy of the dataset elsewhere, leave this
directory empty and pass `--data-dir /path/to/CMAPSSData` instead.

## Do not edit these files

Nothing in this project ever writes to this directory. Keeping the raw files
untouched means the original measurements can always be recovered, and any
result can be traced back to them.
