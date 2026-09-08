"""Does part_3.pdf's target-recompute step earn its keep?

Step 6 of the classifier-guided recipe says "Recompute the targets as the FM
changes during training". The hyperparameter search established that
recomputing *less often* works better, but not whether recomputing at all is
necessary: its grid stopped at every 20 epochs.

This sweeps the refresh interval alone, holding each dataset's selected step
size and target-step count fixed, and extends past the searched range to
`target_refresh_epochs = max_epochs`, which means the targets are built once
at the first epoch and never recomputed - step 6 switched off entirely.

This is an **ablation, not a selection**. Its result is reported and does not
change `STAGE3_SELECTED_HYPERPARAMS`: the configuration in use was chosen by
the documented search, and re-selecting on validation again would add another
round of selection bias to numbers that are already slightly optimistic for
that reason. Test accuracy is printed for the report and plays no part in
anything.

Usage:
    python scripts/ablate_stage3_refresh.py
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.flow_matching.stage3_tuning import search_setting
from src.utils.config import (
    STAGE3_K_SHOT,
    STAGE3_SEEDS,
    STAGE3_SELECTED_HYPERPARAMS,
    STAGE3_SETTINGS,
    Stage3Hyperparams,
)
from src.utils.device import get_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METHOD = "fm_cls_guided"

# The searched values, plus two beyond them. The last equals the epoch budget,
# so the targets are never refreshed after the first build.
REFRESH_INTERVALS = [1, 5, 20, 50, 200]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(STAGE3_SEEDS))
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / "cache"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument(
        "--results-path",
        default=str(PROJECT_ROOT / "reports" / "stage3_refresh_ablation.json"),
    )
    args = parser.parse_args()

    device = get_device()
    base = Stage3Hyperparams()
    print(f"device: {device}   seeds: {args.seeds}   K={STAGE3_K_SHOT}")
    print(f"{len(REFRESH_INTERVALS) * len(STAGE3_SETTINGS) * len(args.seeds)} runs\n")

    everything = []
    for dataset, encoder in sorted(STAGE3_SETTINGS.items()):
        # Hold this dataset's selected step size and target-step count fixed;
        # vary only the refresh interval.
        selected = dict(STAGE3_SELECTED_HYPERPARAMS[(METHOD, dataset)])
        fixed = {k: v for k, v in selected.items() if k != "target_refresh_epochs"}
        grid = [
            {**fixed, "target_refresh_epochs": interval}
            for interval in REFRESH_INTERVALS
        ]

        print(f"== {dataset}/{encoder} ==")
        print(f"   holding {fixed}, selected refresh = "
              f"{selected['target_refresh_epochs']}")

        summaries = search_setting(
            METHOD, dataset, encoder, STAGE3_K_SHOT, args.seeds,
            args.cache_dir, args.output_dir, device,
            base_hyperparams=base, grid=grid,
        )
        by_interval = {s.overrides["target_refresh_epochs"]: s for s in summaries}

        header = (
            f"   {'refresh':>8s} {'val delta':>18s} {'test delta':>18s} "
            f"{'displacement':>13s} {'best epochs':>16s}"
        )
        print(header)
        print("   " + "-" * (len(header) - 3))
        for interval in REFRESH_INTERVALS:
            summary = by_interval[interval]
            marker = ""
            if interval == selected["target_refresh_epochs"]:
                marker = "  <- selected"
            elif interval >= base.max_epochs:
                marker = "  <- never refreshed"
            print(
                f"   {interval:>8d} "
                f"{summary.mean_val_delta * 100:>+10.2f} +/- {summary.std_val_delta * 100:4.2f} "
                f"{summary.mean_test_delta * 100:>+10.2f} +/- {summary.std_test_delta * 100:4.2f} "
                f"{summary.mean_displacement:>13.2f} {str(summary.best_epochs):>16s}{marker}"
            )
        print()
        everything.extend(dataclasses.asdict(s) for s in summaries)

    path = Path(args.results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(everything, f, indent=2)
    print(f"Wrote {len(everything)} configurations to {path}")


if __name__ == "__main__":
    main()
