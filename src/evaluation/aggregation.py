"""Aggregate individual experiment result.json files into summary statistics.

stage_1.pdf requires reporting mean and standard deviation across the
repeated runs for each (dataset, encoder, method, k_shot) setting. This
module scans an outputs/ directory for result.json files (written by the
linear-probe and prototype experiment runners) and computes those summary
statistics.

Convention used throughout: sample standard deviation (ddof=1), since each
run is treated as a sample from the population of possible seeds. This is
undefined for a single run (e.g. the full-data prototype setting, which has
exactly one run by design) and reported as None in that case.
"""

import json
import statistics
from pathlib import Path
from typing import List, Optional, Union


def load_all_results(output_dir: Union[str, Path]) -> List[dict]:
    """Load every result.json under `output_dir` into a flat list of records.

    Each record combines the run's config (dataset, encoder, method, k_shot,
    seed) with its result (test_accuracy, and for linear probe also
    best_val_accuracy/best_epoch).
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
        records.append(record)
    return records


def _sample_std(values: List[float]) -> Optional[float]:
    """Sample standard deviation (ddof=1); None when fewer than 2 values."""
    if len(values) < 2:
        return None
    return statistics.stdev(values)


def _k_shot_sort_key(k_shot):
    # Orders 5, 10, "full" as 5 < 10 < full, without comparing int to str directly.
    return (1, 0) if k_shot == "full" else (0, k_shot)


def aggregate_results(records: List[dict]) -> List[dict]:
    """Group records by (dataset, encoder, method, k_shot) and summarize
    test accuracy as mean and sample standard deviation across seeds.

    Returns:
        A list of summary dicts, one per (dataset, encoder, method, k_shot)
        group, sorted for stable table/plot ordering. Each dict has:
        num_runs, mean_test_accuracy, std_test_accuracy (None if
        num_runs < 2), and seed_accuracies (seed -> test_accuracy).
    """
    groups: dict = {}
    for record in records:
        key = (record["dataset"], record["encoder"], record["method"], record["k_shot"])
        groups.setdefault(key, []).append(record)

    summaries = []
    for (dataset, encoder, method, k_shot), group_records in groups.items():
        accuracies = [r["test_accuracy"] for r in group_records]
        seed_accuracies = dict(sorted((r["seed"], r["test_accuracy"]) for r in group_records))
        summaries.append(
            {
                "dataset": dataset,
                "encoder": encoder,
                "method": method,
                "k_shot": k_shot,
                "num_runs": len(accuracies),
                "mean_test_accuracy": statistics.mean(accuracies),
                "std_test_accuracy": _sample_std(accuracies),
                "seed_accuracies": seed_accuracies,
            }
        )

    summaries.sort(
        key=lambda s: (s["dataset"], s["encoder"], s["method"], _k_shot_sort_key(s["k_shot"]))
    )
    return summaries
