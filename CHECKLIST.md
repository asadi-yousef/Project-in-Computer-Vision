# Pre-Submission Checklist

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
- [ ] Stage 1 and Stage 2 sweep completed: `python scripts/run_all_experiments.py` reports
      everything as "already done" (no pending runs).
- [ ] Stage 3 sweep completed: `python scripts/run_stage3_experiments.py --extension` reports
      everything as "already done" (12 frozen-classifier runs + 12 extension runs).
- [ ] Stage 3 hyperparameter search present: `reports/stage3_tuning.json` exists with 68
      configurations. Re-running `scripts/tune_stage3.py` is optional - the selections it
      produced are recorded in `STAGE3_SELECTED_HYPERPARAMS`.
- [ ] Target-recompute ablation present: `reports/stage3_refresh_ablation.json` exists
      (`python scripts/ablate_stage3_refresh.py`).
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
- [ ] Stage 3's frozen classifier is genuinely Stage 1's: every run calls
      `verify_stored_test_accuracy` before training and refuses to continue unless the loaded
      checkpoint still reproduces its published accuracy. A failure here means the cache or the
      feature preparation drifted, and every Stage 3 delta would be meaningless.
- [ ] Stage 3's flow starts as the exact identity: `tests/test_velocity_net.py` asserts the
      untrained rollout returns its input unchanged and that the pipeline's logits equal the
      frozen classifier's, so every reported delta starts from zero.
- [ ] Stage 3 used raw, unnormalized features. Normalizing them Stage-2-style costs 2.5-4
      accuracy points and is rejected by the verification above; `tests/test_frozen_probe.py`
      asserts that rejection.

## Results and report

- [ ] `python scripts/generate_report.py` run after the final experiment sweep, so `RESULTS.md`
      and `RESULTS.pdf` reflect the latest numbers.
- [ ] `RESULTS.md`/`RESULTS.pdf` contain the Stage 1 deliverables: accuracy table
      (mean +/- sample std), accuracy-vs-shot plots, training/validation loss curves
      (representative 10-shot runs), row-normalized confusion matrices (one per dataset), and
      t-SNE feature-space plots.
- [ ] Stage 2 deliverables present: baseline-vs-flow-matching comparison tables, delta-vs-K
      plots, flow-matching training curves, before/after feature spaces, forward and reverse
      trajectories, and per-step metrics along the flow.
- [ ] Stage 3 deliverables present (part_3.pdf's three required outputs): the main comparison
      table with the change relative to the linear-probe baseline, representative training and
      validation curves for **both** methods, and the before/after feature-space visualization
      computed from one joint embedding.
- [ ] The PDF's title says "Stage 1, Stage 2 and Stage 3 Results" and its figure sections
      include the Stage 3 ones. The PDF carries tables and figures only - the written
      observations live in `RESULTS.md`, so submit both.
- [ ] Every counted claim in the report's Observations sections is derived from the summaries
      rather than typed in, so re-running the sweep cannot leave the prose contradicting the
      tables beside it.
- [ ] Flag any dataset-specific quirks in your writeup - e.g. Flowers-102's official training
      split has exactly 10 images/class, so its 10-shot and full results are identical by
      construction (not a bug). For Stage 3 that means its K=10 row is a full-data result.
- [ ] Be ready to state the Stage 3 caveats out loud: the gains (+0.20 to +1.05) are small
      relative to a per-seed spread reaching 0.87; hyperparameters were selected on validation,
      which flatters the reported test numbers slightly; and both classifier-guided selections
      took the smallest step size searched, so that boundary is untested.

## Repository

- [ ] `git status` is clean; latest commit is pushed to the remote.
- [ ] `data/`, `cache/`, `outputs/`, and `.claude/` are not tracked in git (see `.gitignore`).
- [ ] README is up to date with the actual selected datasets/branch/encoders, and covers all
      three stages.
- [ ] The Stage 3 work is on the branch you intend to submit from - it was developed on
      `stage-3-fm-before-linear-classifier`, so merge or open a PR if `main` is what gets
      graded.
- [ ] Known and accepted: `reports/stage2_presentation.html` is a Stage 2 page and does not
      cover Stage 3. It is gitignored and is not one of the specs' deliverables.
