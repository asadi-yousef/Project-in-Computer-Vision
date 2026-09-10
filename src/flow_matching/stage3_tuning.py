"""Validation-driven hyperparameter search for the Stage 3 strategies.

part_3.pdf asks for this directly. For Strategy 2 it names the knobs:
"Experiment with choices such as the feature-space step size, the number of
target-improvement steps, how often targets are recomputed, and whether the
target updates should be normalized or otherwise constrained." For Strategy 1
it offers the regularization: "penalizing the displacement between z and
z_hat, or the magnitude of the predicted velocities." And globally: "Any
substantial changes should be clearly described and justified
experimentally." The table this module produces is that justification, and is
a reportable result in its own right rather than a private tuning artifact.

Three rules make the search honest:

  1. **Configurations are ranked on validation only.** Test accuracy is
     computed once per configuration for reporting and never influences the
     ranking, exactly as the displacement-penalty choice was made.

  2. **Ranking uses the validation *delta*, not raw validation accuracy.**
     Each seed pairs with its own Stage 1 checkpoint, and those differ in
     accuracy by up to 1.6 points on DTD. Ranking raw accuracy would mix that
     baseline variation into the comparison; the delta against each run's own
     near-identity starting point is the paired statistic.

  3. **Ranking averages over all seeds.** A single validation accuracy on
     these splits has a standard error near 1.1 points, and the effects being
     compared are a fraction of that. Per-seed ranking would largely select
     noise.

What is deliberately *not* searched: the velocity-network architecture, which
part_3.pdf fixes to Stage 2's design; the learning rate, measured across
three seeds at the full epoch budget to make no material difference; the
number of Euler steps, which part_3.pdf requires to be a single value used
throughout Stage 3; and `normalize_target_update`, already measured to make
the target step about three orders of magnitude too short when disabled.
"""

import dataclasses
import itertools
import json
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn

from src.flow_matching.stage3_runner import (
    Stage3Data,
    evaluate_stage3_checkpoint,
    load_stage3_classifier,
    prepare_stage3_features,
    train_stage3_method,
)
from src.utils.config import STAGE3_METHODS, ExperimentConfig, Stage3Hyperparams

# The searched values, per strategy. Kept deliberately small and readable:
# these are grids a reader can check against the spec's list of knobs, not an
# automated search space.
STRATEGY_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "fm_cls_guided": {
        "target_step_size": [0.02, 0.05, 0.1, 0.2],
        "target_num_steps": [1, 3],
        "target_refresh_epochs": [1, 5, 20],
    },
    "fm_cls_rolled": {
        "displacement_penalty": [0.0, 0.03, 0.1, 0.3, 1.0],
        "velocity_penalty": [0.0, 0.1],
    },
}


@dataclasses.dataclass
class TuningRun:
    """One (configuration, seed) result."""

    method: str
    dataset: str
    encoder: str
    seed: int
    overrides: Dict[str, Any]
    initial_val_accuracy: float
    best_val_accuracy: float
    val_delta: float
    baseline_test_accuracy: float
    test_accuracy: float
    test_delta: float
    best_epoch: int
    test_mean_displacement: float

    def label_key(self) -> tuple:
        """Hashable identity of this run's configuration, for grouping."""
        return tuple(sorted(self.overrides.items()))


@dataclasses.dataclass
class TuningSummary:
    """One configuration, averaged over the seeds it was run on.

    `mean_val_delta` is the ranking key. Everything else is for the report.
    """

    method: str
    dataset: str
    encoder: str
    overrides: Dict[str, Any]
    num_seeds: int
    mean_val_delta: float
    std_val_delta: Optional[float]
    mean_test_delta: float
    std_test_delta: Optional[float]
    mean_test_accuracy: float
    mean_baseline_test_accuracy: float
    best_epochs: List[int]
    mean_displacement: float

    @property
    def label(self) -> str:
        """The configuration as a compact `key=value` string, for tables."""
        return ", ".join(f"{key}={value}" for key, value in sorted(self.overrides.items()))


def build_grid(method: str) -> List[Dict[str, Any]]:
    """Expand a strategy's grid into a list of override dicts.

    Args:
        method: a Stage 3 method name.

    Returns:
        One dict per configuration, in a stable order.

    Raises:
        ValueError: if `method` has no grid defined.
    """
    if method not in STRATEGY_GRIDS:
        raise ValueError(f"method must be one of {STAGE3_METHODS}, got {method!r}")

    grid = STRATEGY_GRIDS[method]
    keys = list(grid)
    return [
        dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))
    ]


def _sample_std(values: List[float]) -> Optional[float]:
    """Sample standard deviation, or None for a single value.

    Matches the convention in `src.evaluation.aggregation`.
    """
    return statistics.stdev(values) if len(values) > 1 else None


def evaluate_config(
    method: str,
    data: Stage3Data,
    classifier: nn.Module,
    baseline_test_accuracy: float,
    hyperparams: Stage3Hyperparams,
    dataset: str,
    encoder: str,
    seed: int,
    device: torch.device,
) -> TuningRun:
    """Train one configuration on one seed and score it.

    Takes already-loaded data and an already-loaded classifier rather than
    paths, so a search over dozens of configurations reads the feature cache
    once per seed instead of once per configuration - and so this function
    can be tested without a project directory.

    Args:
        method: which strategy to train.
        data: prepared splits for this seed.
        classifier: that seed's frozen Stage 1 probe, on `device`.
        baseline_test_accuracy: the Stage 1 test accuracy this run's delta is
            measured against.
        hyperparams: the configuration to evaluate.
        dataset, encoder, seed: recorded on the result.
        device: device to train on.

    Returns:
        A `TuningRun`.
    """
    train_result = train_stage3_method(
        method, data, classifier, hyperparams, seed, device
    )
    _, test_accuracy, test_displacement = evaluate_stage3_checkpoint(
        train_result.best_state_dict,
        hyperparams.hidden_dims,
        classifier,
        data.test_features,
        data.test_labels,
        hyperparams.num_euler_steps,
        device,
    )

    return TuningRun(
        method=method,
        dataset=dataset,
        encoder=encoder,
        seed=seed,
        overrides={},  # filled in by the caller, which knows the grid point
        initial_val_accuracy=train_result.initial_val_accuracy,
        best_val_accuracy=train_result.best_val_accuracy,
        val_delta=train_result.best_val_accuracy - train_result.initial_val_accuracy,
        baseline_test_accuracy=baseline_test_accuracy,
        test_accuracy=test_accuracy,
        test_delta=test_accuracy - baseline_test_accuracy,
        best_epoch=train_result.best_epoch,
        test_mean_displacement=test_displacement,
    )


def summarize_runs(runs: Sequence[TuningRun]) -> TuningSummary:
    """Average one configuration's runs across seeds.

    Raises:
        ValueError: if `runs` is empty, or mixes configurations.
    """
    if not runs:
        raise ValueError("runs is empty; nothing to summarize")
    if len({(run.method, run.dataset, run.encoder, run.label_key()) for run in runs}) != 1:
        raise ValueError("all runs in a summary must share one configuration")

    first = runs[0]
    val_deltas = [run.val_delta for run in runs]
    test_deltas = [run.test_delta for run in runs]

    return TuningSummary(
        method=first.method,
        dataset=first.dataset,
        encoder=first.encoder,
        overrides=dict(first.overrides),
        num_seeds=len(runs),
        mean_val_delta=statistics.mean(val_deltas),
        std_val_delta=_sample_std(val_deltas),
        mean_test_delta=statistics.mean(test_deltas),
        std_test_delta=_sample_std(test_deltas),
        mean_test_accuracy=statistics.mean(run.test_accuracy for run in runs),
        mean_baseline_test_accuracy=statistics.mean(
            run.baseline_test_accuracy for run in runs
        ),
        best_epochs=[run.best_epoch for run in runs],
        mean_displacement=statistics.mean(run.test_mean_displacement for run in runs),
    )


def search_setting(
    method: str,
    dataset: str,
    encoder: str,
    k_shot,
    seeds: Sequence[int],
    cache_dir: Union[str, Path],
    output_dir: Union[str, Path],
    device: torch.device,
    base_hyperparams: Optional[Stage3Hyperparams] = None,
    grid: Optional[List[Dict[str, Any]]] = None,
    progress: Optional[Callable[[TuningRun], None]] = None,
) -> List[TuningSummary]:
    """Run the full grid for one (method, dataset, encoder), over all seeds.

    Features and the frozen classifier are loaded once per seed and reused
    across every configuration.

    Args:
        method: the strategy to search.
        dataset, encoder, k_shot: the setting.
        seeds: seeds to average over. Each pairs with its own Stage 1 run.
        cache_dir, output_dir: project directories.
        device: device to train on.
        base_hyperparams: settings held fixed across the grid. Defaults to
            `Stage3Hyperparams()`.
        grid: override dicts to try. Defaults to `build_grid(method)`.
        progress: optional callback invoked with each completed `TuningRun`.

    Returns:
        One `TuningSummary` per configuration, sorted best-first by mean
        validation delta.
    """
    base_hyperparams = base_hyperparams or Stage3Hyperparams()
    grid = grid if grid is not None else build_grid(method)

    # Load each seed's data and classifier once, not once per configuration.
    per_seed = []
    for seed in seeds:
        config = ExperimentConfig(
            dataset=dataset, encoder=encoder, method=method, k_shot=k_shot, seed=seed
        )
        data = prepare_stage3_features(config, cache_dir)
        frozen = load_stage3_classifier(config, data, output_dir, device)
        per_seed.append((seed, data, frozen))

    summaries = []
    for overrides in grid:
        hyperparams = dataclasses.replace(base_hyperparams, **overrides)
        runs = []
        for seed, data, frozen in per_seed:
            run = evaluate_config(
                method, data, frozen.model, frozen.stored_test_accuracy,
                hyperparams, dataset, encoder, seed, device,
            )
            run.overrides = dict(overrides)
            runs.append(run)
            if progress is not None:
                progress(run)
        summaries.append(summarize_runs(runs))

    return sorted(summaries, key=lambda summary: summary.mean_val_delta, reverse=True)


def select_best(summaries: Sequence[TuningSummary]) -> TuningSummary:
    """The configuration with the highest mean validation delta.

    Ties are broken by the smaller mean displacement, preferring the flow
    that achieves the same validation result by changing the representation
    less - which is the spirit of part_3.pdf's regularization suggestion and
    of initializing near identity in the first place.

    Raises:
        ValueError: if `summaries` is empty.
    """
    if not summaries:
        raise ValueError("summaries is empty; nothing to select from")
    return min(
        summaries, key=lambda summary: (-summary.mean_val_delta, summary.mean_displacement)
    )


def format_tuning_table(summaries: Sequence[TuningSummary], top_n: Optional[int] = None) -> str:
    """Render a search as a Markdown table, best-first.

    The validation column is the one selection used; the test column is
    reported alongside so a reader can see how well the choice transferred.

    The selected epochs and mean displacement each configuration produced
    are not shown - they are kept in the saved results, where `select_best`
    still uses displacement to break ties, and in reports/stage3_tuning.json.

    Args:
        summaries: results for one (method, dataset, encoder), already sorted.
        top_n: show only this many rows, or all of them if None.
    """
    if not summaries:
        return "_No tuning results._"

    rows = list(summaries)[:top_n] if top_n else list(summaries)

    def cell(mean: float, std: Optional[float]) -> str:
        percent = f"{mean * 100:+.2f}%"
        return percent if std is None else f"{percent} +/- {std * 100:.2f}"

    lines = [
        "| Configuration | Val delta (selection) | Test delta | Test accuracy |",
        "|---|---|---|---|",
    ]
    for summary in rows:
        lines.append(
            f"| {summary.label} | {cell(summary.mean_val_delta, summary.std_val_delta)} "
            f"| {cell(summary.mean_test_delta, summary.std_test_delta)} "
            f"| {summary.mean_test_accuracy * 100:.2f}% |"
        )
    return "\n".join(lines)


def save_tuning_results(
    summaries: Sequence[TuningSummary], path: Union[str, Path]
) -> None:
    """Write a search to JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([dataclasses.asdict(summary) for summary in summaries], f, indent=2)


def load_tuning_results(path: Union[str, Path]) -> List[TuningSummary]:
    """Read back a search written by `save_tuning_results`."""
    with open(path) as f:
        return [TuningSummary(**entry) for entry in json.load(f)]
