# CVLAB Summer Project — Stage 1: Classification Baselines

## Project goal

This repository implements Stage 1 of a multi-stage computer-vision project: a reproducible
image classification pipeline built on **frozen pretrained encoders**. No flow-matching
component is used at this stage — that is introduced in later stages.

## Stage 1 scope

Two baselines are implemented and compared:

1. **Linear probe** — a linear classifier (`s = Wz + b`) trained with softmax cross-entropy on
   top of frozen image features.
2. **Image-derived class prototypes** — classification by cosine similarity to the average
   (L2-normalized) training feature of each class. No parameters are trained.

Both baselines are evaluated at three training-set sizes per dataset: 5-shot, 10-shot, and full,
using the official train/validation/test splits (validation is used only for model selection,
test only for final reporting).

## Selected datasets and branch

- **Datasets:** DTD (official partition 1) and Oxford Flowers-102.
- **Encoders:** ImageNet-1K-pretrained ResNet-18 on both datasets; DINOv2 ViT-S/14 on DTD.
- **Second baseline:** Image-derived class prototypes (Option A).

## Environment setup

_To be completed in Task 2._

This project targets both a Windows PC with a CUDA GPU and a Mac laptop. Code selects the
compute device automatically (`cuda` → `mps` → `cpu`) and avoids OS-specific paths.

```bash
pip install -r requirements.txt
```

> Note: the default `torch`/`torchvision` wheels from `requirements.txt` work on both platforms.
> For a CUDA-accelerated build on Windows, follow the install command generated at
> https://pytorch.org/get-started/locally/ for your specific CUDA version instead.

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

_To be completed in Task 18._

## Regenerating tables and plots

_To be completed in Tasks 13–17._

## Output directory structure

```
data/     # raw datasets (gitignored)
cache/    # cached frozen-encoder features (gitignored)
outputs/  # results, checkpoints, tables, plots (gitignored)
```

## Reproducibility

_To be completed in Task 2 / Task 19._

## Common errors

_To be completed as they are encountered._

## Running tests

```bash
pytest
```
