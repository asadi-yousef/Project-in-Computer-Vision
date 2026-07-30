"""Aggregate all experiment results under outputs/ into summary statistics
(mean and sample standard deviation of test accuracy per dataset/encoder/
method/k_shot setting), saved as CSV and JSON.

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --output-dir outputs --save-dir reports
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.aggregation import aggregate_results, load_all_results

_CSV_FIELDNAMES = [
    "dataset", "encoder", "method", "k_shot", "num_runs",
    "mean_test_accuracy", "std_test_accuracy", "seed_accuracies",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--save-dir", default="reports")
    args = parser.parse_args()

    records = load_all_results(args.output_dir)
    print(f"Loaded {len(records)} individual run results from {args.output_dir}")

    summaries = aggregate_results(records)
    print(f"Aggregated into {len(summaries)} (dataset, encoder, method, k_shot) settings")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    with open(save_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        for summary in summaries:
            row = dict(summary)
            row["seed_accuracies"] = json.dumps(row["seed_accuracies"])
            writer.writerow(row)

    print(f"Saved {save_dir / 'summary.json'} and {save_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
