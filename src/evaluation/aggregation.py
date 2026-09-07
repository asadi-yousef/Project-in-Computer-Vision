"""Aggregate individual experiment result.json files into summary statistics.

Both stages require reporting mean and standard deviation across the repeated
runs for each setting. This module scans an outputs/ directory for
result.json files (written by the linear-probe, prototype, and flow-matching
experiment runners) and computes those summary statistics.

Convention used throughout: sample standard deviation (ddof=1), since each
run is treated as a sample from the population of possible seeds. This is
undefined for a single run (e.g. the full-data prototype and flow-matching
settings, which have exactly one run by design) and reported as None in that
case.

A setting is identified by (dataset, encoder, method, k_shot, num_euler_steps).
The Euler-step count is part of the key because Stage 2 evaluates each FM
method at T in {4, 12}, and those are distinct results that must not be
averaged together. It is None for the Stage 1 methods, which have no T, so
Stage 1 records aggregate exactly as they did before.

`aggregate_results` covers test accuracy and the paired delta, which is all
Stages 1 and 2 report. Stage 3 runs additionally record how far the flow
moved the features, which epoch was selected, and the validation improvement
over the untrained identity; `summarize_stage3_runs` averages those, since
they are Stage 3-specific and would be None for every other record.
"""

import json
import statistics
from pathlib import Path
from typing import List, Optional, Union

# Display order for the accuracy table and plot legends: every method sits
# directly after the Stage 1 baseline it is compared against. The Stage 3
# methods follow the linear probe, which is their baseline (part_3.pdf); the
# Stage 2 methods follow the prototype baseline. Alphabetical order would put
# fm_rolled first and split the Stage 3 pair, both of which read backwards.
METHOD_DISPLAY_ORDER = (
    "linear_probe",
    "fm_cls_rolled",
    "fm_cls_guided",
    "prototype",
    "fm_standard",
    "fm_rolled",
)


def load_all_results(output_dir: Union[str, Path]) -> List[dict]:
    """Load every result.json under `output_dir` into a flat list of records.

    Each record combines the run's config (dataset, encoder, method, k_shot,
    seed) with its result. Flow-matching results additionally carry
    num_euler_steps, baseline_test_accuracy and delta_accuracy; those keys
    are filled in as None for Stage 1 records so every record has the same
    shape.
    """
    records = []
    for result_path in sorted(Path(output_dir).rglob("result.json")):
        with open(result_path) as f:
            data = json.load(f)
        record = {
            "dataset": data["config"]["dataset"],
            "encoder": data["config"]["encoder"],
            "method": data["config"]["method"],
            "k_shot": data["config"]["k_shot"],
            "seed": data["config"]["seed"],
            **data["result"],
        }
        for optional_key in ("num_euler_steps", "baseline_test_accuracy", "delta_accuracy"):
            record.setdefault(optional_key, None)
        records.append(record)
    return records


# The three conditions part_3.pdf's main comparison puts side by side:
# "Stage 1 linear probe; end-to-end rolled-out classification training;
# classifier-guided FM training."
STAGE3_COMPARISON_METHODS = ("linear_probe", "fm_cls_rolled", "fm_cls_guided")


def method_label(method: str, num_euler_steps: Optional[int]) -> str:
    """Human-readable name for a method, including its T when it has one.

    Used for table rows and plot legends, e.g. "fm_rolled (T=12)" versus a
    plain "prototype".
    """
    if num_euler_steps is None:
        return method
    return f"{method} (T={num_euler_steps})"


def _sample_std(values: List[float]) -> Optional[float]:
    """Sample standard deviation (ddof=1); None when fewer than 2 values."""
    if len(values) < 2:
        return None
    return statistics.stdev(values)


def _mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    """Mean of the values, or None if any is missing (Stage 1 has no delta)."""
    if not values or any(value is None for value in values):
        return None
    return statistics.mean(values)


def _k_shot_sort_key(k_shot):
    # Orders 5, 10, "full" as 5 < 10 < full, without comparing int to str directly.
    return (1, 0) if k_shot == "full" else (0, k_shot)


def _method_sort_key(method: str):
    # Known methods in their display order; anything unexpected sorts last,
    # alphabetically, rather than raising.
    if method in METHOD_DISPLAY_ORDER:
        return (0, METHOD_DISPLAY_ORDER.index(method), "")
    return (1, 0, method)


def _euler_sort_key(num_euler_steps: Optional[int]) -> int:
    # None (the Stage 1 methods) sorts before any real step count.
    return -1 if num_euler_steps is None else num_euler_steps


def aggregate_results(records: List[dict]) -> List[dict]:
    """Group records by setting and summarize test accuracy across seeds.

    Args:
        records: flat records from `load_all_results` (or constructed
            directly in tests). Records without a "num_euler_steps" key are
            treated as having none, so Stage 1 records need no changes.

    Returns:
        A list of summary dicts, one per (dataset, encoder, method, k_shot,
        num_euler_steps) group, sorted for stable table/plot ordering. Each
        dict has: num_runs, mean_test_accuracy, std_test_accuracy (None if
        num_runs < 2), seed_accuracies (seed -> test_accuracy), and for
        flow-matching settings also mean_baseline_accuracy,
        mean_delta_accuracy and std_delta_accuracy (all None otherwise).

        The delta statistics are computed from each run's *own* paired
        baseline, not from a difference of means: at K=5 and K=10 every seed
        samples a different subset and therefore has a different baseline,
        so pairing within a seed is the meaningful comparison.
    """
    groups: dict = {}
    for record in records:
        key = (
            record["dataset"],
            record["encoder"],
            record["method"],
            record["k_shot"],
            record.get("num_euler_steps"),
        )
        groups.setdefault(key, []).append(record)

    summaries = []
    for (dataset, encoder, method, k_shot, num_euler_steps), group_records in groups.items():
        accuracies = [r["test_accuracy"] for r in group_records]
        deltas = [r.get("delta_accuracy") for r in group_records]
        baselines = [r.get("baseline_test_accuracy") for r in group_records]
        seed_accuracies = dict(sorted((r["seed"], r["test_accuracy"]) for r in group_records))
        has_deltas = all(delta is not None for delta in deltas)
        summaries.append(
            {
                "dataset": dataset,
                "encoder": encoder,
                "method": method,
                "k_shot": k_shot,
                "num_euler_steps": num_euler_steps,
                "num_runs": len(accuracies),
                "mean_test_accuracy": statistics.mean(accuracies),
                "std_test_accuracy": _sample_std(accuracies),
                "mean_baseline_accuracy": _mean_or_none(baselines),
                "mean_delta_accuracy": _mean_or_none(deltas),
                "std_delta_accuracy": _sample_std(deltas) if has_deltas else None,
                "seed_accuracies": seed_accuracies,
            }
        )

    summaries.sort(
        key=lambda s: (
            s["dataset"],
            s["encoder"],
            _method_sort_key(s["method"]),
            _k_shot_sort_key(s["k_shot"]),
            _euler_sort_key(s["num_euler_steps"]),
        )
    )
    return summaries


def summarize_stage3_runs(records: List[dict]) -> List[dict]:
    """Average the Stage 3-only fields the general aggregator does not touch.

    These support the observations rather than the required accuracy table:
    how far the flow actually moved the features, which epoch validation
    selected, and how much validation improved over the untrained identity.
    The displacement in particular is the diagnostic for a flow that improves
    its objective by inflating feature magnitude instead of by improving the
    representation.

    Args:
        records: flat records from `load_all_results`. Records without
            Stage 3's fields are ignored, so this can be handed the whole
            outputs directory.

    Returns:
        One dict per (dataset, encoder, method, k_shot) group, sorted like
        the accuracy table. Each has num_runs, mean_val_delta,
        std_val_delta, mean_displacement, mean_initial_val_accuracy,
        mean_best_val_accuracy and best_epochs (per seed, seed-ordered).
    """
    groups = {}
    for record in records:
        if record.get("test_mean_displacement") is None:
            continue
        key = (record["dataset"], record["encoder"], record["method"], record["k_shot"])
        groups.setdefault(key, []).append(record)

    summaries = []
    for (dataset, encoder, method, k_shot), group in groups.items():
        group = sorted(group, key=lambda record: record["seed"])
        val_deltas = [record["val_delta_accuracy"] for record in group]
        summaries.append(
            {
                "dataset": dataset,
                "encoder": encoder,
                "method": method,
                "k_shot": k_shot,
                "num_runs": len(group),
                "mean_val_delta": statistics.mean(val_deltas),
                "std_val_delta": _sample_std(val_deltas),
                "mean_initial_val_accuracy": statistics.mean(
                    record["initial_val_accuracy"] for record in group
                ),
                "mean_best_val_accuracy": statistics.mean(
                    record["best_val_accuracy"] for record in group
                ),
                "mean_displacement": statistics.mean(
                    record["test_mean_displacement"] for record in group
                ),
                "best_epochs": [record["best_epoch"] for record in group],
            }
        )

    return sorted(
        summaries,
        key=lambda summary: (
            summary["dataset"],
            summary["encoder"],
            _method_sort_key(summary["method"]),
            _k_shot_sort_key(summary["k_shot"]),
        ),
    )

