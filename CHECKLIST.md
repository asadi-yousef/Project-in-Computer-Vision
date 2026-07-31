# Stage 1 Pre-Submission Checklist

Run through this before submitting. Each item maps to a concrete command or file you can check.

## Setup and data

- [ ] `pip install -r requirements.txt` runs cleanly on the target machine.
- [ ] `python scripts/check_device.py` shows the expected device (`cuda` on the Windows PC, `mps`
      or `cpu` on the Mac).
- [ ] DTD and Flowers-102 verified against official splits:
      `python scripts/verify_dataset_splits.py --dataset dtd --data-dir data`
      `python scripts/verify_dataset_splits.py --dataset flowers102 --data-dir data`
      (DTD: 1880/1880/1880, 47 classes; Flowers-102: 1020/1020/6149, 102 classes.)

## Features and experiments

- [ ] All three feature caches exist: `cache/dtd/resnet18/`, `cache/dtd/dinov2_vits14/`,
      `cache/flowers102/resnet18/` (each with `train.pt`, `val.pt`, `test.pt`).
- [ ] Full experiment sweep completed: `python scripts/run_all_experiments.py` reports
      everything as "already done" (no pending runs).
- [ ] `pytest` passes in full, including `tests/test_integrity.py`'s real-data checks (these
      directly verify the run-count protocol against your actual `outputs/` - see below).

## Protocol compliance (verified automatically by `tests/test_integrity.py`)

- [ ] Every linear-probe (dataset, encoder, k-shot) setting has exactly 3 runs (seeds 0, 1, 2).
- [ ] Every prototype (dataset, encoder, k-shot) setting has exactly 3 runs for 5-/10-shot, and
      exactly 1 run for full.
- [ ] No test accuracy was used to select epochs/hyperparameters (checkpoints are always
      selected by validation accuracy - see `src/classifiers/linear_probe.py`).
- [ ] Encoders are frozen and produce deterministic features (BatchNorm stats don't drift,
      DINOv2 output is repeatable).

## Results and report

- [ ] `python scripts/generate_report.py` run after the final experiment sweep, so `RESULTS.md`
      and `RESULTS.pdf` reflect the latest numbers.
- [ ] `RESULTS.md`/`RESULTS.pdf` contain: accuracy table (mean +/- sample std), accuracy-vs-shot
      plots, training/validation loss curves (representative 10-shot runs), row-normalized
      confusion matrices (one per dataset), and t-SNE feature-space plots.
- [ ] Flag any dataset-specific quirks in your writeup - e.g. Flowers-102's official training
      split has exactly 10 images/class, so its 10-shot and full results are identical by
      construction (not a bug).

## Repository

- [ ] `git status` is clean; latest commit is pushed to the remote.
- [ ] `data/`, `cache/`, `outputs/`, and `.claude/` are not tracked in git (see `.gitignore`).
- [ ] README is up to date with the actual selected datasets/branch/encoders.
