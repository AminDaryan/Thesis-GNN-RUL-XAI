# Stored training runs

These folders are the training runs stored on Amin's PC, copied here on
2026-11-14 so that every run and the settings it was made with are kept on
GitHub. Nothing in them was edited after the run wrote it.

## What a run folder holds

- `summary.json`: the run's settings, the command it was started with, the
  commit of the code that made it, and its averaged errors.
- `results.csv`: one row per seed, with that seed's errors.
- `seed_N/training_config.yaml`, `preprocessing_config.yaml` and
  `model_config.yaml`: every setting of training, of the preparation of the
  sensor readings, and of the network.
- `seed_N/model.pt`: that seed's trained network, its learned numbers saved
  to a file.

The `verification_*` folders hold re-checks of figures quoted in the project's
documents, `sensitivity_*` the study that changes one setting at a time,
`baselines_*` the scores of ordinary models on the same engines, and
`.playground` the jobs started from the website's playground.

## Runs made with uncommitted code

A folder whose name contains `-dirty` was made while the code had changes not
yet committed. It is kept as a record, but it cannot be reproduced from any
commit, so its numbers should not be quoted as results (working rule 3a in
`CLAUDE.md`).

## New runs are not added automatically

`reports/` is still listed in `.gitignore`, so these files were added with
`git add -f`. A run trained after 2026-11-14 stays on the computer that made
it until someone adds it the same way.

Left out of the copy: Python's cache folders (`__pycache__`) and the `.log`
files the sensitivity study writes while it runs.
