# CVLAB Summer Project — Stages 1, 2 and 3

## Project goal

This repository implements a multi-stage computer-vision project built on **frozen
pretrained encoders**. Stage 1 establishes reproducible classification baselines; Stage 2
adds a flow-matching layer that transports a frozen feature toward its class prototype;
Stage 3 puts a flow-matching layer *before* Stage 1's trained linear classifier and asks
whether it can reshape the frozen features into something that classifier handles better.

The encoders are never fine-tuned at any stage. Stage 1 trains only the classifier, Stage 2
only the velocity network, and Stage 3 only the velocity network — its classifier is Stage 1's
own checkpoint, loaded and frozen.

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

## Stage 3 scope

Stage 3 keeps Stage 2's velocity network and Euler integrator but changes what the flow is
for. The pipeline is

```
z --FM--> z_hat --frozen linear classifier--> s
```

where the classifier is the Stage 1 linear probe for the same dataset, encoder, K and seed —
loaded from its checkpoint, verified against its published test accuracy, and frozen. Two
training objectives are compared, both updating only the velocity network:

1. **End-to-end rolled-out classification** (`fm_cls_rolled`) — run the full T-step Euler
   rollout, score the endpoint with the frozen classifier, and backpropagate cross-entropy
   through all T steps. Structurally Stage 2's rolled-out training with a different target.
2. **Classifier-guided targets** (`fm_cls_guided`) — build a per-example target by taking a
   gradient step on `z_hat` in feature space that reduces the frozen classifier's loss, then
   train with Stage 2's standard FM loss using `z` as source and that target as endpoint.
   Targets are periodically recomputed as the flow changes.

**Optional extension.** part_3.pdf also invites unfreezing the classifier and training it
jointly with the flow. `fm_cls_joint` does that, on Strategy 1's objective and with everything
else held identical to the corresponding frozen run, so the difference is attributable to the
unfreezing. It is accompanied by `cls_finetune`, a control the spec does not ask for: the
Stage 1 classifier trained on alone, with the flow held at its identity initialization. Without
it the extension is unreadable, because any gain from unfreezing could just be the gain from
training the classifier for another 200 epochs — and measurably, part of it is.

part_3.pdf narrows the scope deliberately: **one representative encoder per dataset**
(DINOv2 for DTD, ResNet-18 for Flowers-102), **one training-set size** (K=10), and **a single
number of Euler steps** (T=4) used throughout. Three seeds, as in Stage 1.

Four Stage 3 conventions worth knowing, each of which differs from Stage 2 for a reason:

- **Features are *not* L2-normalized.** Stage 2 normalizes because its classifier is cosine
  similarity, which is scale-invariant. Stage 1's linear probe is not, and was fitted to raw
  features with norms of roughly 24 (ResNet-18) to 48 (DINOv2); normalizing here costs 2.5–4
  accuracy points before training starts. `verify_stored_test_accuracy` fails the run if this
  is ever violated.
- **The flow is initialized to the exact identity.** part_3.pdf asks that the untrained system
  behave like the original linear probe. Zeroing the velocity network's final layer makes it
  predict zero velocity, so the Euler rollout returns `z` unchanged and the pipeline reproduces
  the Stage 1 probe's logits bit for bit — every reported delta starts from precisely zero.
  `build_near_identity_velocity_network` does this; Stage 2's default initialization moves
  DINOv2 features by about 6% of their norm and is left alone.
- **Checkpoints are selected by best validation accuracy**, as in Stage 1, rather than keeping
  the final weights as Stage 2 does. Stage 3's pipeline has a meaningful validation accuracy,
  and the unregularized objective degrades badly late in training without it. Selection runs
  over the trained epochs only: including the untrained identity would clamp every delta at
  `>= 0` and hide a result where the flow genuinely hurts.
- **The classifier is frozen with `requires_grad=False`, not `no_grad` or `detach`.** Strategy 1
  must backpropagate *through* the classifier to reach the flow; the other two would sever
  that path and silently train nothing.

One structural finding shapes the whole stage: because part_3.pdf requires reusing Stage 1's
own training subset, and Stage 1 trained its probe on that subset to convergence, the frozen
classifier already reaches **100% accuracy and a cross-entropy near 0.002** there. The
classification objective therefore starts at its floor, and the cheapest remaining descent
direction is to inflate feature magnitude rather than improve the representation. The
displacement penalty part_3.pdf offers is what prevents that — unregularized, validation
accuracy peaks within a few epochs and then falls below the untrained identity.

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

**Stage 3** has no single-run script — its protocol is twelve runs, so it is driven by its
own sweep (see below). To run one setting only, pass filters:

```bash
python scripts/run_stage3_experiments.py --methods fm_cls_guided --datasets dtd --seeds 0
```

Outputs land under `outputs/<method>/<dataset>/<encoder>/k10/T4/seed<seed>/`. The
corresponding Stage 1 linear-probe run must already exist: Stage 3 loads that checkpoint
rather than retraining an equivalent one.

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

Stage 3 runs separately, after Stage 1 is complete:

```bash
python scripts/tune_stage3.py                          # hyperparameter search (~40 min on a laptop GPU)
python scripts/run_stage3_experiments.py               # the twelve reported runs (~3 min)
python scripts/run_stage3_experiments.py --extension   # plus the optional extension (~4 min)
```

`--extension` adds `fm_cls_joint` and its `cls_finetune` control. They are opt-in because
part_3.pdf marks the extension optional and asks for it only "after completing the
frozen-classifier experiments".

One further ablation answers a question the search could not:

```bash
python scripts/ablate_stage3_refresh.py     # does step 6 earn its keep? (~6 min)
```

part_3.pdf's classifier-guided recipe ends with "recompute the targets as the FM changes
during training". The search showed recomputing *less* often works better but stopped at every
20 epochs, so it could not say whether recomputing at all is necessary. This varies the refresh
interval alone out to `max_epochs`, at which the targets are built once and never recomputed.
It is deliberately an **ablation, not a selection** — the result is reported and does not change
`STAGE3_SELECTED_HYPERPARAMS`, so it adds no further validation-based selection to numbers that
are already slightly optimistic for that reason. Results go to
`reports/stage3_refresh_ablation.json`.

The search is optional to re-run — its outcome is already recorded in
`STAGE3_SELECTED_HYPERPARAMS` in `src/utils/config.py`, and the full results in
`reports/stage3_tuning.json`. It exists because part_3.pdf names specific knobs to
experiment with and asks that changes be "clearly described and justified experimentally",
so the search table is a reportable result rather than a private tuning artifact. It trains
34 configurations x 3 seeds x 2 datasets and ranks them on **mean validation delta**, with
test accuracy computed for the report but never used for ranking.

Both scripts skip completed work; `--force` re-runs it.

### Repetition protocol

Stage 1 uses two different run counts for the full-data setting, and Stage 2 follows the one
belonging to the branch it extends:

| method | 5-shot | 10-shot | full |
|---|---|---|---|
| linear probe | 3 subset seeds | 3 subset seeds | 3 initialization seeds |
| prototype | 3 subset seeds | 3 subset seeds | **1 run** |
| flow matching (Stage 2) | 3 subset seeds | 3 subset seeds | **1 run** |
| Stage 3 | — | 3 subset seeds | — |
| Stage 3 extension | — | 3 subset seeds | — |

Stage 3 runs only at K=10 (part_3.pdf: "one training-set size for the main experiments"), and
follows the linear probe it extends: three seeds, each pairing with the Stage 1 checkpoint
trained on that seed's own subset.

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

The same command also builds the Stage 3 section: the main three-way comparison table
(linear probe vs. both Stage 3 methods, with the change relative to baseline), training and
selection diagnostics, a class-separation table, training curves, a three-panel feature-space
comparison, and the hyperparameter search.

Two Stage 3 figures differ from their Stage 2 counterparts by design. The training-curve
figure gives each strategy's objective its own axis, because the two minimize different
quantities — a classification cross-entropy and a squared velocity error — and marks both the
untrained pipeline's accuracy and the selected epoch, so the gap between them is what Stage 3
contributed. The feature-space projection is fitted on **unnormalized** features and draws no
prototypes, because Stage 3's classifier is a hyperplane rather than a set of reference points
and is not scale-invariant.

Because a two-dimensional projection can only suggest an answer to the question part_3.pdf
poses of that figure — how each strategy changes the class structure — the section also reports
a Fisher-style class-separation ratio measured in the **full** feature space, on the same
samples the figure plots.

The written **Observations** sections of `RESULTS.md` derive their counted claims from the
aggregated summaries and measurements rather than hardcoding them, so the prose cannot drift
out of step with the tables if the sweep is re-run. That includes which method won: the Stage 3
ranking sentence branches on the measured result rather than asserting a fixed reading.

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
reports/           # aggregated summary.{json,csv}, figures/, feature_viz_selection_*.json, stage3_tuning.json (tracked - small, final numbers)
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
outputs/fm_cls_rolled/<dataset>/<encoder>/k10/T4/seed<n>/
outputs/fm_cls_guided/<dataset>/<encoder>/k10/T4/seed<n>/
outputs/fm_cls_joint/<dataset>/<encoder>/k10/T4/seed<n>/
outputs/cls_finetune/<dataset>/<encoder>/k10/T4/seed<n>/
```

The two extension runs additionally write `classifier.pt`, since they are the only ones that
train a classifier of their own. It sits beside `checkpoint.pt` rather than inside it so that
every existing reader — all of which expect `checkpoint.pt` to be a velocity network — keeps
working unchanged.

Stage 3 reuses the same layout helper as Stage 2, so its runs sit alongside them in the same
shape and the report pipeline walks them identically. All four files are written by one shared
`save_run_artifacts`, which is also what the linear probe uses — a new stage cannot
accidentally produce directories the report cannot read.

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
- **Stage 1 / Stage 3 comparability**: every Stage 3 run calls `verify_stored_test_accuracy`
  before training, re-scoring the loaded Stage 1 checkpoint on the test split and refusing to
  continue unless it reproduces the accuracy Stage 1 published. This catches the failure modes
  that would otherwise be invisible and would quietly invalidate the stage: the wrong seed's
  checkpoint, a rebuilt feature cache, or features prepared differently here than they were
  then — notably L2-normalizing them, which `tests/test_frozen_probe.py` asserts is rejected.
- **Stage 3 near-identity initialization**: `tests/test_velocity_net.py` asserts that the
  untrained rollout returns its input exactly, for several T, and that the full pipeline's
  logits equal the frozen classifier's. On the real checkpoints the pipeline reproduces every
  stored Stage 1 accuracy with `max|dlogit| = 0`.

## Common errors

- **`RuntimeError: Dataset not found. You can use download=True to download it`** — DTD/Flowers-102
  data isn't in `data/` yet. Run `scripts/verify_dataset_splits.py` with `--download` first (see
  Dataset preparation above); `download=False` is the default everywhere else on purpose, to
  avoid re-downloading by accident.
- **`ValueError: dinov2_vits14 is only used on 'dtd' in this project`** — intentional: this
  project scoped DINOv2 to DTD only (see `src/utils/config.py`). Not a bug; pass
  `--encoder resnet18` for Flowers-102.
- **`FileNotFoundError: No Stage 1 checkpoint at ...`** when running Stage 3 — Stage 3 reuses
  Stage 1's trained classifier rather than retraining one, so that run must exist first. Run
  `scripts/run_all_experiments.py` (or `scripts/run_linear_probe.py` for the single setting).
- **`ValueError: Frozen probe from ... would be invalid`** — the loaded Stage 1 checkpoint no
  longer reproduces its stored test accuracy. Usually means the feature cache was rebuilt since
  Stage 1 ran, or features are being prepared differently (Stage 3 must use raw, unnormalized
  features). This is an intentional guard, not a bug: continuing would make every Stage 3 delta
  meaningless.
- **`KeyError: No Stage 3 selection recorded for (...)`** — `stage3_hyperparams_for` was asked
  for a (method, dataset) pair the search never covered. Run `scripts/tune_stage3.py`, or check
  the pair against `STAGE3_SELECTED_HYPERPARAMS` in `src/utils/config.py`.
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
