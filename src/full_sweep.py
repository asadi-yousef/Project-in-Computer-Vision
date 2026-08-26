"""Full experiment-sweep orchestration: extract any missing cached features,
then run every linear-probe, prototype, and flow-matching experiment the
Stage 1 and Stage 2 protocols require, skipping anything already completed.

Factored out of scripts/run_all_experiments.py so the sweep logic can be
unit-tested against a tiny synthetic setup, without invoking the CLI script
or touching the real project directories.
"""

import dataclasses
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch

from src.classifiers.linear_probe_runner import run_linear_probe_experiment
from src.classifiers.prototype_runner import run_prototype_experiment
from src.features.pipeline import all_splits_cached, extract_and_cache_all_splits
from src.flow_matching.runner import (
    flow_matching_run_dir,
    run_rolled_out_fm_experiment,
    run_standard_fm_experiment,
)
from src.utils.config import VALID_EULER_STEPS, ExperimentConfig, FlowMatchingHyperparams

# This project's fixed choice of datasets/encoders (see src/utils/config.py).
DATASET_ENCODER_PAIRS: List[Tuple[str, str]] = [
    ("dtd", "resnet18"),
    ("dtd", "dinov2_vits14"),
    ("flowers102", "resnet18"),
]
K_SHOTS: List = [5, 10, "full"]
SEEDS: List[int] = [0, 1, 2]
EULER_STEP_COUNTS: List[int] = list(VALID_EULER_STEPS)


def linear_probe_result_path(
    output_dir: Union[str, Path], dataset: str, encoder: str, k_shot, seed: int
) -> Path:
    return Path(output_dir) / "linear_probe" / dataset / encoder / f"k{k_shot}" / f"seed{seed}" / "result.json"


def prototype_result_path(
    output_dir: Union[str, Path], dataset: str, encoder: str, k_shot, seed: int
) -> Path:
    seed_folder = f"seed{seed}" if k_shot != "full" else "single_run"
    return Path(output_dir) / "prototype" / dataset / encoder / f"k{k_shot}" / seed_folder / "result.json"


def flow_matching_result_path(
    output_dir: Union[str, Path],
    dataset: str,
    encoder: str,
    method: str,
    k_shot,
    num_euler_steps: int,
    seed: int,
) -> Path:
    return (
        flow_matching_run_dir(
            output_dir, dataset, encoder, method, k_shot, num_euler_steps, seed
        )
        / "result.json"
    )


def seeds_for_k_shot(k_shot, seeds: List[int]) -> List[int]:
    """Which seeds a parameter-carrying method runs for a given training-set size.

    The 5-shot and 10-shot settings repeat over the subset seeds. The full
    setting is a single run, following the Stage 1 prototype branch this
    stage extends ("the full-data result requires one run"), rather than the
    linear probe's 3 initialization seeds. A project decision - the two
    Stage 1 branches genuinely differ here, and Stage 2 is an extension of
    the prototype branch.
    """
    return list(seeds) if k_shot != "full" else [0]


def run_full_sweep(
    data_dir: Union[str, Path],
    cache_dir: Union[str, Path],
    output_dir: Union[str, Path],
    device: torch.device,
    force_rerun: bool = False,
    dataset_encoder_pairs: Optional[List[Tuple[str, str]]] = None,
    k_shots: Optional[List] = None,
    seeds: Optional[List[int]] = None,
    euler_step_counts: Optional[List[int]] = None,
    flow_matching_hyperparams: Optional[FlowMatchingHyperparams] = None,
) -> None:
    """Run the full Stage 1 and Stage 2 experiment protocols.

    Anything already completed (a result.json already on disk) is skipped
    unless force_rerun is set, so this is safe to run repeatedly - e.g. to
    finish an interrupted sweep, or add newly-cached encoders later.

    Args:
        data_dir, cache_dir, output_dir: project directories.
        device: device to train/extract on.
        force_rerun: if True, re-run and overwrite completed classifier and
            flow-matching experiments. Feature extraction is unaffected by
            this flag - already-cached features are always reused, since
            re-extracting is expensive and unrelated to re-running training.
        dataset_encoder_pairs, k_shots, seeds, euler_step_counts: override
            this project's full protocol (used by tests to run a tiny
            synthetic sweep instead).
        flow_matching_hyperparams: override the velocity network's
            architecture and optimizer settings (used by tests to train a
            tiny network for a few epochs).
    """
    dataset_encoder_pairs = dataset_encoder_pairs or DATASET_ENCODER_PAIRS
    k_shots = k_shots or K_SHOTS
    seeds = seeds or SEEDS
    euler_step_counts = euler_step_counts or EULER_STEP_COUNTS
    flow_matching_hyperparams = flow_matching_hyperparams or FlowMatchingHyperparams()

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

    print("== Step 4: flow-matching experiments ==")
    for dataset, encoder in dataset_encoder_pairs:
        for k_shot in k_shots:
            for seed in seeds_for_k_shot(k_shot, seeds):
                _run_standard_fm_setting(
                    dataset, encoder, k_shot, seed, cache_dir, output_dir, device,
                    force_rerun, euler_step_counts, flow_matching_hyperparams,
                )
                _run_rolled_out_fm_setting(
                    dataset, encoder, k_shot, seed, cache_dir, output_dir, device,
                    force_rerun, euler_step_counts, flow_matching_hyperparams,
                )

    print("All experiments complete.")


def _run_standard_fm_setting(
    dataset: str,
    encoder: str,
    k_shot,
    seed: int,
    cache_dir: Union[str, Path],
    output_dir: Union[str, Path],
    device: torch.device,
    force_rerun: bool,
    euler_step_counts: List[int],
    hyperparams: FlowMatchingHyperparams,
) -> None:
    """Train one standard-FM network for this setting and evaluate it at every T.

    Standard FM's objective never discretizes the path, so all T share one
    training. The whole setting is therefore skipped only when *every* T
    already has a result - a partially-completed setting is retrained,
    because there is no separate per-T training to resume.
    """
    label = f"fm_standard {dataset}/{encoder} k={k_shot} seed={seed}"
    result_paths = [
        flow_matching_result_path(
            output_dir, dataset, encoder, "fm_standard", k_shot, num_euler_steps, seed
        )
        for num_euler_steps in euler_step_counts
    ]
    if all(path.exists() for path in result_paths) and not force_rerun:
        print(f"  {label}: already done, skipping")
        return

    print(f"  {label}: running (one training, evaluated at T={euler_step_counts}) ...")
    config = ExperimentConfig(
        dataset=dataset, encoder=encoder, method="fm_standard", k_shot=k_shot, seed=seed,
        flow_matching=hyperparams,
    )
    results = run_standard_fm_experiment(
        config, cache_dir, output_dir, device, euler_step_counts=euler_step_counts
    )
    for result in results:
        print(
            f"    T={result['num_euler_steps']}: test accuracy "
            f"{result['test_accuracy']:.4f} (delta {result['delta_accuracy']:+.4f})"
        )


def _run_rolled_out_fm_setting(
    dataset: str,
    encoder: str,
    k_shot,
    seed: int,
    cache_dir: Union[str, Path],
    output_dir: Union[str, Path],
    device: torch.device,
    force_rerun: bool,
    euler_step_counts: List[int],
    hyperparams: FlowMatchingHyperparams,
) -> None:
    """Train one rolled-out network per T for this setting.

    T is baked into the learned weights here, so each T is an independent
    run and is skipped independently.
    """
    for num_euler_steps in euler_step_counts:
        label = f"fm_rolled {dataset}/{encoder} k={k_shot} T={num_euler_steps} seed={seed}"
        result_path = flow_matching_result_path(
            output_dir, dataset, encoder, "fm_rolled", k_shot, num_euler_steps, seed
        )
        if result_path.exists() and not force_rerun:
            print(f"  {label}: already done, skipping")
            continue

        print(f"  {label}: running ...")
        config = ExperimentConfig(
            dataset=dataset, encoder=encoder, method="fm_rolled", k_shot=k_shot, seed=seed,
            flow_matching=dataclasses.replace(hyperparams, num_euler_steps=num_euler_steps),
        )
        result = run_rolled_out_fm_experiment(config, cache_dir, output_dir, device)
        print(
            f"    test accuracy: {result['test_accuracy']:.4f} "
            f"(delta {result['delta_accuracy']:+.4f})"
        )
