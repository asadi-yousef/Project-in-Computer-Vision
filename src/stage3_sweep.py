"""Stage 3 experiment sweep: both strategies, both datasets, all seeds.

part_3.pdf's main comparison is deliberately narrow - one representative
encoder per dataset, one training-set size, one number of Euler steps - so
this sweep is twelve runs rather than Stage 2's grid: two strategies x two
datasets x three seeds.

Each run uses the hyperparameters selected by the search recorded in
`STAGE3_SELECTED_HYPERPARAMS`, and pairs with the Stage 1 linear-probe run of
the same dataset, encoder, K and seed - the same subset, the same trained
classifier, and that run's own published accuracy as the baseline.

Like `src.full_sweep`, anything already completed is skipped unless
`force_rerun` is set, so an interrupted sweep can simply be re-run.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch

from src.flow_matching.stage3_runner import (
    ALL_STAGE3_METHODS,
    run_stage3_experiment,
    stage3_run_dir,
)
from src.utils.config import (
    STAGE3_K_SHOT,
    STAGE3_METHODS,
    STAGE3_SEEDS,
    STAGE3_SETTINGS,
    ExperimentConfig,
    Stage3Hyperparams,
    stage3_hyperparams_for,
)


def stage3_result_path(
    output_dir: Union[str, Path],
    dataset: str,
    encoder: str,
    method: str,
    k_shot,
    num_euler_steps: int,
    seed: int,
) -> Path:
    """Where a completed Stage 3 run records its result."""
    return (
        stage3_run_dir(
            output_dir, dataset, encoder, method, k_shot, num_euler_steps, seed
        )
        / "result.json"
    )


def run_stage3_sweep(
    cache_dir: Union[str, Path],
    output_dir: Union[str, Path],
    device: torch.device,
    force_rerun: bool = False,
    methods: Optional[Sequence[str]] = None,
    datasets: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    k_shot=None,
    base_hyperparams: Optional[Stage3Hyperparams] = None,
    verbose: bool = True,
) -> List[dict]:
    """Run every Stage 3 experiment part_3.pdf's main comparison needs.

    Args:
        cache_dir: directory holding cached features.
        output_dir: the project outputs root. Both the Stage 1 checkpoints
            these runs pair with and the Stage 3 run directories live under it.
        device: device to train and evaluate on.
        force_rerun: re-run and overwrite runs that already have a result.json.
        methods, datasets, seeds, k_shot: override the protocol, for tests and
            partial re-runs. `methods` defaults to the two frozen-classifier
            strategies; part_3.pdf's optional extension is opt-in, and is
            requested by naming `fm_cls_joint` and `cls_finetune` explicitly.
        base_hyperparams: settings the selected overrides are applied on top
            of. Tests use this to train a tiny network for a few epochs.
        verbose: print progress.

    Returns:
        One result dict per run that executed, skipped runs excluded.
    """
    methods = list(methods) if methods is not None else list(STAGE3_METHODS)
    datasets = list(datasets) if datasets is not None else list(STAGE3_SETTINGS)
    seeds = list(seeds) if seeds is not None else list(STAGE3_SEEDS)
    k_shot = k_shot if k_shot is not None else STAGE3_K_SHOT

    results = []
    for method in methods:
        for dataset in datasets:
            encoder = STAGE3_SETTINGS[dataset]
            hyperparams = stage3_hyperparams_for(method, dataset, base_hyperparams)

            for seed in seeds:
                label = f"{method} {dataset}/{encoder} k={k_shot} seed={seed}"
                result_path = stage3_result_path(
                    output_dir, dataset, encoder, method, k_shot,
                    hyperparams.num_euler_steps, seed,
                )
                if result_path.exists() and not force_rerun:
                    if verbose:
                        print(f"  {label}: already done, skipping")
                    continue

                if verbose:
                    print(f"  {label}: running ...", flush=True)
                config = ExperimentConfig(
                    dataset=dataset, encoder=encoder, method=method,
                    k_shot=k_shot, seed=seed, stage3=hyperparams,
                )
                result = run_stage3_experiment(config, cache_dir, output_dir, device)
                results.append({**result, "method": method, "dataset": dataset, "seed": seed})

                if verbose:
                    print(
                        f"    test accuracy {result['test_accuracy'] * 100:.2f}% "
                        f"(delta {result['delta_accuracy'] * 100:+.2f}, "
                        f"baseline {result['baseline_test_accuracy'] * 100:.2f}%, "
                        f"best epoch {result['best_epoch']}, "
                        f"displacement {result['test_mean_displacement']:.2f})",
                        flush=True,
                    )

    return results
