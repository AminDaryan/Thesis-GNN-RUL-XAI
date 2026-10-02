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

## Where to download them

1. Open the NASA Prognostics Center of Excellence data repository:
   <https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>
2. Download **Turbofan Engine Degradation Simulation Data Set**.
3. Copy the files into this directory.
4. Check them:

   ```bash
   python scripts/prepare_data.py --check
   ```

   This reports which files are present, which are missing, and whether the
   ones present have the row and engine counts of the files NASA distributes
   (20 631 training rows over 100 engines, and 100 test engines, for FD001).
   The paper's Table 5 (Wen et al., 2026, page 7) prints the same engine
   counts. NASA's own readme prints FD004's two the other way round (248
   training engines, 249 test engines, where the files hold 249 and 248), and
   none of the papers held in `docs/references/` prints the row counts.
   (Corrected 2026-10-02: this called the counts "published for the official
   release"; found when the cited papers were collected into
   `docs/references/` and checked against the places that cite them.)

All four sub-datasets (FD001 to FD004) can live here together. Only the three
files for the sub-dataset named in the configuration are read on any one run.

## Keeping the dataset somewhere else

If you would rather keep a single copy of the dataset elsewhere, leave this
directory empty and point the script at it instead:

```bash
python scripts/prepare_data.py --run --data-dir /path/to/CMAPSSData
```

The same applies to `run_preprocessing(config, data_dir=...)` in Python.

> **There is no substitute for the real files.** This project does not
> generate stand-in data, on purpose. A pipeline that appears to work on
> invented signals tells you nothing about whether it works on engines, and a
> number computed from invented signals is not a result. If the files are
> missing, every command and every test says so and points at the download.

## Do not edit these files

Nothing in this project ever writes to this directory. Keeping the raw files
untouched means the original measurements can always be recovered, and any
result can be traced back to them.
