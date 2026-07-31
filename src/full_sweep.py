"""Full experiment-sweep orchestration: extract any missing cached features,
then run every linear-probe and prototype experiment stage_1.pdf's protocol
requires, skipping anything already completed.

Factored out of scripts/run_all_experiments.py so the sweep logic can be
unit-tested against a tiny synthetic setup, without invoking the CLI script
or touching the real project directories.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch

from src.classifiers.linear_probe_runner import run_linear_probe_experiment
from src.classifiers.prototype_runner import run_prototype_experiment
from src.features.pipeline import all_splits_cached, extract_and_cache_all_splits
from src.utils.config import ExperimentConfig

# This project's fixed choice of datasets/encoders (see src/utils/config.py).
DATASET_ENCODER_PAIRS: List[Tuple[str, str]] = [
    ("dtd", "resnet18"),
    ("dtd", "dinov2_vits14"),
    ("flowers102", "resnet18"),
]
K_SHOTS: List = [5, 10, "full"]
SEEDS: List[int] = [0, 1, 2]


def linear_probe_result_path(
    output_dir: Union[str, Path], dataset: str, encoder: str, k_shot, seed: int
) -> Path:
    return Path(output_dir) / "linear_probe" / dataset / encoder / f"k{k_shot}" / f"seed{seed}" / "result.json"


def prototype_result_path(
    output_dir: Union[str, Path], dataset: str, encoder: str, k_shot, seed: int
) -> Path:
    seed_folder = f"seed{seed}" if k_shot != "full" else "single_run"
    return Path(output_dir) / "prototype" / dataset / encoder / f"k{k_shot}" / seed_folder / "result.json"


def run_full_sweep(
    data_dir: Union[str, Path],
    cache_dir: Union[str, Path],
    output_dir: Union[str, Path],
    device: torch.device,
    force_rerun: bool = False,
    dataset_encoder_pairs: Optional[List[Tuple[str, str]]] = None,
    k_shots: Optional[List] = None,
    seeds: Optional[List[int]] = None,
) -> None:
    """Run the full stage_1.pdf experiment protocol.

    Anything already completed (a result.json already on disk) is skipped
    unless force_rerun is set, so this is safe to run repeatedly - e.g. to
    finish an interrupted sweep, or add newly-cached encoders later.

    Args:
        data_dir, cache_dir, output_dir: project directories.
        device: device to train/extract on.
        force_rerun: if True, re-run and overwrite completed linear-probe
            and prototype experiments. Feature extraction is unaffected by
            this flag - already-cached features are always reused, since
            re-extracting is expensive and unrelated to re-running training.
        dataset_encoder_pairs, k_shots, seeds: override this project's full
            protocol (used by tests to run a tiny synthetic sweep instead).
    """
    dataset_encoder_pairs = dataset_encoder_pairs or DATASET_ENCODER_PAIRS
    k_shots = k_shots or K_SHOTS
    seeds = seeds or SEEDS

    print("== Step 1: feature extraction ==")
    # force_rerun deliberately does not apply here: it means "re-run the
    # classifier experiments," not "re-extract already-cached features,"
    # since extraction is the expensive step this project caches specifically
    # to avoid repeating.
    for dataset, encoder in dataset_encoder_pairs:
        if all_splits_cached(cache_dir, dataset, encoder):
            print(f"  {dataset}/{encoder}: already cached, skipping")
            continue
        print(f"  {dataset}/{encoder}: extracting ...")
        extract_and_cache_all_splits(dataset, encoder, data_dir, cache_dir, device)

    print("== Step 2: linear-probe experiments ==")
    for dataset, encoder in dataset_encoder_pairs:
        for k_shot in k_shots:
            for seed in seeds:
                result_path = linear_probe_result_path(output_dir, dataset, encoder, k_shot, seed)
                if result_path.exists() and not force_rerun:
                    print(f"  linear_probe {dataset}/{encoder} k={k_shot} seed={seed}: already done, skipping")
                    continue
                print(f"  linear_probe {dataset}/{encoder} k={k_shot} seed={seed}: running ...")
                config = ExperimentConfig(
                    dataset=dataset, encoder=encoder, method="linear_probe", k_shot=k_shot, seed=seed
                )
                result = run_linear_probe_experiment(
                    config, cache_dir, Path(output_dir) / "linear_probe", device
                )
                print(f"    test accuracy: {result['test_accuracy']:.4f}")

    print("== Step 3: prototype experiments ==")
    few_shot_k_shots = [k for k in k_shots if k != "full"]
    for dataset, encoder in dataset_encoder_pairs:
        for k_shot in few_shot_k_shots:
            for seed in seeds:
                result_path = prototype_result_path(output_dir, dataset, encoder, k_shot, seed)
                if result_path.exists() and not force_rerun:
                    print(f"  prototype {dataset}/{encoder} k={k_shot} seed={seed}: already done, skipping")
                    continue
                print(f"  prototype {dataset}/{encoder} k={k_shot} seed={seed}: running ...")
                config = ExperimentConfig(
                    dataset=dataset, encoder=encoder, method="prototype", k_shot=k_shot, seed=seed
                )
                result = run_prototype_experiment(config, cache_dir, Path(output_dir) / "prototype")
                print(f"    test accuracy: {result['test_accuracy']:.4f}")

        if "full" in k_shots:
            result_path = prototype_result_path(output_dir, dataset, encoder, "full", 0)
            if result_path.exists() and not force_rerun:
                print(f"  prototype {dataset}/{encoder} k=full: already done, skipping")
                continue
            print(f"  prototype {dataset}/{encoder} k=full: running ...")
            config = ExperimentConfig(
                dataset=dataset, encoder=encoder, method="prototype", k_shot="full", seed=0
            )
            result = run_prototype_experiment(config, cache_dir, Path(output_dir) / "prototype")
            print(f"    test accuracy: {result['test_accuracy']:.4f}")

    print("All experiments complete.")
