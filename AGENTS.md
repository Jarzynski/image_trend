# Project Instructions

This repository is a research pipeline for image-based A-share return prediction.

## Documentation Source Of Truth

- `README.md` is the compact project context source for future Codex sessions and context-compacted conversations.
- After changing any pipeline script, update `README.md` in the same turn.
- Script changes include changes to:
  - `config.py`
  - `01_build_panel.py`
  - `02_make_labels_and_baselines.py`
  - `03_make_images.py`
  - `03_make_images_fast.py`
  - `04_train_logistic.py`
  - `05_train_cnn2d.py`
  - `05_train_cnn2d_4090_fast.py`
  - `06_backtest_decile.py`

## README Update Requirements

When a script changes, keep these README sections consistent:

- Current version, if the change is part of a versioned release.
- Project file tree, when files are added, removed, renamed, or demoted to compatibility wrappers.
- Input/output table, when data paths, file formats, schemas, or execution assumptions change.
- Script output list, when new CSV/Parquet/model/log artifacts are added or removed.
- Upgrade notes, when users need to rerun upstream or downstream stages.
- Version record, when the change should be tracked for Git tag or release history.

## Scope Discipline

- Keep code changes targeted to the requested behavior.
- Do not remove existing backtest logic, model logic, or compatibility entrypoints unless explicitly requested.
- Do not upload, stage, or commit generated data, model weights, prediction files, logs, or paper PDFs.
- Preserve the current `uv` workflow for Python commands.
