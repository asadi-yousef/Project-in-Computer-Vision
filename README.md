# CVLAB Summer Project — Stages 1 and 2

## Project goal

This repository implements a multi-stage computer-vision project built on **frozen
pretrained encoders**. Stage 1 establishes reproducible classification baselines; Stage 2
adds a flow-matching layer that transports a frozen feature toward its class prototype.
The encoders are never fine-tuned at any stage — only the classifier and, in Stage 2, the
velocity network are trained.

## Stage 1 scope

Two baselines are implemented and compared:

1. **Linear probe** — a linear classifier (`s = Wz + b`) trained with softmax cross-entropy on
   top of frozen image features.
2. **Image-derived class prototypes** — classification by cosine similarity to the average
   (L2-normalized) training feature of each class. No parameters are trained.

Both baselines are evaluated at three training-set sizes per dataset: 5-shot, 10-shot, and full,
using the official train/validation/test splits (validation is used only for model selection,
test only for final reporting).

## Stage 2 scope

Stage 2 adds a small velocity network `v(z, t)` — an MLP with two hidden layers of width 512
and SiLU activations, with the scalar time concatenated to the feature — trained to transport
a frozen feature toward the prototype of its class. Two training objectives are compared:

1. **Standard flow matching** — sample `t ~ U(0,1)`, interpolate `z_t = (1-t)z + t*p_y`, and
   regress `v(z_t, t)` toward the constant target velocity `p_y - z`. The path is never
   discretized, so one trained network serves every T.
2. **Rolled-out flow matching** — unroll the full T-step Euler sequence used at inference and
   supervise only the final transported point against the prototype, backpropagating through
   all T steps. T is baked into the weights, so training and inference must use the same T.

At inference both variants take T Euler steps (`T` in {4, 12}) and classify the transported
feature by cosine similarity to **the same prototypes Stage 1 used**. Every run reuses the
identical K-shot subset, seed and prototypes as the corresponding Stage 1 prototype run —
`src/flow_matching/runner.py` calls the same `sample_balanced_subset_indices` and
`compute_class_prototypes` functions, and each result stores the baseline accuracy it was
compared against so any drift would be caught immediately.

Two Stage 2 conventions worth knowing:

- **Features are L2-normalized before the flow.** Stage 1's classifier already normalizes at
  both ends, and raw encoder features have norms of roughly 24–48 against unit-norm
  prototypes, which would make the regression target almost entirely about shrinking the
  norm — a direction the cosine classifier ignores. Because cosine similarity is
  scale-invariant, this leaves the Stage 1 baseline numbers unchanged.
- **Training runs a fixed epoch budget and keeps the final weights.** Validation loss is
  logged for the stability curves but never used for selection, matching stage_2.pdf's
  request for stable training and a fair comparison rather than a tuned result.

## Selected datasets and branch

- **Datasets:** DTD (official partition 1) and Oxford Flowers-102.
- **Encoders:** ImageNet-1K-pretrained ResNet-18 on both datasets; DINOv2 ViT-S/14 on DTD.
- **Second baseline:** Image-derived class prototypes (Option A).

## Environment setup

This project targets both a Windows PC with a CUDA GPU and a Mac laptop. Code selects the
compute device automatically (`cuda` → `mps` → `cpu`) and avoids OS-specific paths.

```bash
pip install -r requirements.txt
```

> Note: the default `torch`/`torchvision` wheels from `requirements.txt` work on both platforms.
> For a CUDA-accelerated build on Windows, follow the install command generated at
> https://pytorch.org/get-started/locally/ for your specific CUDA version instead.

Verify which device will actually be used before training:

```bash
python scripts/check_device.py
```

## Dataset preparation

Datasets are loaded through torchvision's official train/val/test splits (`src/data/datasets.py`)
and downloaded automatically on first use into `data/`:

```bash
python scripts/verify_dataset_splits.py --dataset dtd --data-dir data --download
python scripts/verify_dataset_splits.py --dataset flowers102 --data-dir data --download
```

Each run prints split sizes and checks them against the documented official protocol
(DTD partition 1: 1880/1880/1880; Flowers-102: 1020/1020/6149), and asserts there is no
image overlap between splits. Omit `--download` on subsequent runs once the data is present.

## Feature extraction

Frozen encoders are run once per (dataset, encoder, split) and cached under `cache/`:

```bash
python scripts/extract_features.py --dataset dtd --encoder resnet18
python scripts/extract_features.py --dataset dtd --encoder dinov2_vits14
python scripts/extract_features.py --dataset flowers102 --encoder resnet18
```

Each cache file (`cache/<dataset>/<encoder>/<split>.pt`) stores features, labels, and
metadata (feature dim, sample/class counts) used to validate the cache later. No classifier
training or evaluation ever re-runs the image encoder.

## Running one experiment

**Linear probe** (trains `W, b` on cached features, selects the best epoch by validation
accuracy, evaluates test accuracy once at the end):

```bash
python scripts/run_linear_probe.py --dataset dtd --encoder resnet18 --k-shot 10 --seed 0
python scripts/run_linear_probe.py --dataset dtd --encoder resnet18 --k-shot full --seed 0
```

`--k-shot` is `5`, `10`, or `full`. For `5`/`10`, `--seed` (0, 1, or 2) selects both the
balanced training subset and the classifier's initialization/shuffling. For `full`, it
selects only the classifier's initialization. Outputs (config, per-epoch history, result,
checkpoint) are saved under `outputs/linear_probe/<dataset>/<encoder>/k<k>/seed<seed>/`.

**Image-derived prototypes** (no training — builds prototypes from cached features and
classifies by cosine similarity):

```bash
python scripts/run_prototype.py --dataset dtd --encoder resnet18 --k-shot 10 --seed 0
python scripts/run_prototype.py --dataset dtd --encoder resnet18 --k-shot full
```

`--seed` is required for `5`/`10` (selects the subset) and omitted for `full` (single run,
no subset to select). Outputs are saved under `outputs/prototype/<dataset>/<encoder>/k<k>/`.

## Running all experiments

To run the entire Stage 1 and Stage 2 protocol in one go (feature extraction for anything
not already cached, then every linear-probe, prototype and flow-matching run):

```bash
python scripts/run_all_experiments.py
```

It's safe to interrupt and re-run: anything already completed (a `result.json` on disk) is
skipped. Pass `--force-rerun` to re-run and overwrite completed experiments (already-cached
features are always reused regardless, since re-extracting them is expensive and unrelated
to re-running training).

The Stage 2 grid is 3 dataset/encoder pairs x K in {5, 10, full} x {standard, rolled-out} x
T in {4, 12}, which is **63 trainings producing 84 run records** — the counts differ because
standard FM trains once per (pair, K, seed) and is evaluated at both T, while rolled-out
trains once per T. This has a consequence for resuming: rolled-out runs skip per T, but a
standard-FM setting is only skipped when *every* T is already present, since a
half-finished setting has no per-T training to resume from. On a laptop GPU the whole
Stage 2 grid takes roughly ten minutes.

### Repetition protocol

Stage 1 uses two different run counts for the full-data setting, and Stage 2 follows the one
belonging to the branch it extends:

| method | 5-shot | 10-shot | full |
|---|---|---|---|
| linear probe | 3 subset seeds | 3 subset seeds | 3 initialization seeds |
| prototype | 3 subset seeds | 3 subset seeds | **1 run** |
| flow matching | 3 subset seeds | 3 subset seeds | **1 run** |

stage_1.pdf specifies 3 initialization seeds for the full linear probe but states that "the
full-data result requires one run" for the image-prototype branch. Stage 2 extends the
prototype branch, so its full-data settings are single runs and carry no error bars.
`seeds_for_k_shot` in `src/full_sweep.py` is the one place that decision lives.

## Regenerating tables and plots

Once you've run experiments (see above), aggregate and visualize the results:

```bash
python scripts/generate_report.py
```

This reads every `result.json` under `outputs/`, computes mean and sample standard deviation
of test accuracy per (dataset, encoder, method, k-shot) setting, saves the aggregated numbers
to `reports/summary.{json,csv}`, generates one accuracy-vs-training-size plot per
(dataset, encoder) pair under `reports/figures/`, and writes it all into both `RESULTS.md` and
`RESULTS.pdf` at the project root. Re-run it any time after new experiments to refresh both.

It also plots training/validation loss (representative 10-shot, seed-0 run) for each
dataset/encoder pair used with the linear probe, a row-normalized confusion matrix per
dataset (representative setting: full training data, linear probe, seed 0), and a t-SNE
feature-space plot per (dataset, encoder) pair. No re-training occurs for confusion matrices
or feature-space plots - both are recomputed from saved checkpoints/cached features.

The same command builds the Stage 2 sections: a baseline-versus-flow-matching comparison
table per pair, accuracy-vs-K and change-vs-baseline plots, flow-matching training curves,
a three-panel feature-space comparison (original / after standard FM / after rolled-out FM,
from one joint t-SNE projection), forward and reverse flow trajectories (PCA, as stage_2.pdf
recommends — a linear projection keeps straight paths straight), and per-step metrics along
the flow. Everything is recomputed from the saved velocity-network checkpoints, so no
training is repeated. Missing runs are skipped with a message rather than failing the
report, so a partially-completed sweep still produces output.

The written **Observations** section of `RESULTS.md` derives its counted claims from the
aggregated summaries rather than hardcoding them, so the prose cannot drift out of step with
the tables if the sweep is re-run.

Feature-space plots reuse the same 10 classes and 150 test samples per dataset across every
encoder trained on it (selection stored in `reports/feature_viz_selection_<dataset>.json` for
reproducibility, generated once and reused on every re-run).

## Output directory structure

```
configs/           # example experiment config (YAML schema reference)
src/               # library code (data, encoders, features, classifiers, flow_matching, evaluation, visualization, utils)
scripts/           # CLI entry points
tests/             # test suite (pytest)
data/              # raw datasets (gitignored - regenerate via scripts/verify_dataset_splits.py --download)
cache/             # cached frozen-encoder features (gitignored - regenerate via scripts/extract_features.py)
outputs/           # per-run configs, checkpoints, history, results (gitignored - regenerate via scripts/run_all_experiments.py)
reports/           # aggregated summary.{json,csv}, figures/, feature_viz_selection_*.json (tracked - small, final numbers)
RESULTS.md         # generated results report (tracked)
RESULTS.pdf        # same report as a PDF (tracked)
```

`data/`, `cache/`, and `outputs/` are gitignored because they're large and fully regenerable from
the code plus a fixed seed. `reports/` and the root `RESULTS.*` files are tracked, since they're
small and are the actual reportable deliverable.

Within `outputs/`, each method gets its own tree. Stage 2 inserts a `T<steps>` level and reuses
the prototype branch's `single_run` folder for the full-data setting:

```
outputs/linear_probe/<dataset>/<encoder>/k<k>/seed<n>/
outputs/prototype/<dataset>/<encoder>/k<k>/{seed<n>|single_run}/
outputs/fm_standard/<dataset>/<encoder>/k<k>/T<steps>/{seed<n>|single_run}/
outputs/fm_rolled/<dataset>/<encoder>/k<k>/T<steps>/{seed<n>|single_run}/
```

Every flow-matching run directory is self-contained — `config.yaml`, `history.json`,
`result.json` and `checkpoint.pt` — so every Stage 2 figure can be rebuilt from the saved
weights without retraining. The two standard-FM directories for a given (pair, K, seed) hold
the same checkpoint, because that objective does not depend on T.

## Reproducibility

- **Seeding**: `src/utils/seeding.py::set_seed()` seeds Python, NumPy, and PyTorch together.
  A single `seed` (0, 1, or 2) drives both the balanced K-shot subset selection and the
  classifier's initialization/shuffling for a given run (see `src/utils/config.py`).
- **Determinism**: same seed -> identical training history and identical K-shot subset,
  verified in `tests/test_linear_probe.py` and `tests/test_few_shot.py`.
- **Run metadata**: every result is saved alongside its full config, git commit hash, and
  library versions (`src/utils/run_metadata.py`), so any number can be traced back to exactly
  what produced it.
- **Integrity checks** (`tests/test_integrity.py`) guard the experiment rules that are easy to
  violate by accident rather than by incorrect implementation: encoders never drift or behave
  stochastically in eval mode (BatchNorm stats, DINOv2 determinism), classifier code never
  imports an encoder module (so training/evaluation can never silently re-run the image
  encoder), and - when real data/cache/outputs are present - the actual completed runs are
  checked against the exact stage_1.pdf run-count protocol (3 seeds per linear-probe setting;
  3 seeds for prototype 5-/10-shot, 1 run for prototype full).
- **Stage 1 / Stage 2 comparability**: `tests/test_flow_runner.py` asserts that the
  prototypes a flow-matching run trains against reproduce the corresponding Stage 1 run's
  stored test accuracy to within 1e-9, using the real cache and outputs when present. If the
  subsets or prototypes ever drift apart, the comparison stops being valid and this fails.

## Common errors

- **`RuntimeError: Dataset not found. You can use download=True to download it`** — DTD/Flowers-102
  data isn't in `data/` yet. Run `scripts/verify_dataset_splits.py` with `--download` first (see
  Dataset preparation above); `download=False` is the default everywhere else on purpose, to
  avoid re-downloading by accident.
- **`ValueError: dinov2_vits14 is only used on 'dtd' in this project`** — intentional: this
  project scoped DINOv2 to DTD only (see `src/utils/config.py`). Not a bug; pass
  `--encoder resnet18` for Flowers-102.
- **`UserWarning: xFormers is not available`** when building `DINOv2Encoder` — harmless. DINOv2
  falls back to a native (non-xFormers) attention implementation; this project doesn't depend on
  xFormers to keep dependencies minimal.
- **`Selected device: cpu` on a machine with a GPU** — the installed `torch` build doesn't have
  CUDA support (the plain `pip install torch` wheel is CPU-only on Windows). Install the
  CUDA-enabled wheel from https://pytorch.org/get-started/locally/, then confirm with
  `scripts/check_device.py`.
- **pytest fails with a `PermissionError` on a `pytest-of-<user>` temp directory** (seen on
  Windows when a stale/locked temp folder exists from another process) — already worked around
  via `--basetemp=.pytest_tmp` in `pyproject.toml`, which keeps pytest's temp files inside the
  project instead of the OS temp directory. If it still happens, delete `.pytest_tmp/` and retry.
- **Non-ASCII characters in file paths** (e.g. a Windows username with accented/non-Latin
  characters) can cause some third-party tools to mis-render paths in error messages or logs.
  This doesn't affect the actual file I/O (Python handles Unicode paths correctly) - only cosmetic
  output.

## Running tests

```bash
pytest
```
