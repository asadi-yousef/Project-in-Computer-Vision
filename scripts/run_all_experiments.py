"""End-to-end orchestration: extract any missing cached features, then run
every linear-probe, prototype, and flow-matching experiment required by the
Stage 1 and Stage 2 protocols (both datasets x their encoders x k-shot
settings x seeds, and for flow matching x method x Euler-step count),
skipping anything already completed.

This trains real models - expect it to take a while (DINOv2 full-data
linear-probe runs are the slowest part on CPU, and rolled-out FM at T=12
backpropagates through 12 chained network evaluations per step). Safe to
interrupt and re-run: completed runs are skipped unless --force-rerun is
passed.

Usage:
    python scripts/run_all_experiments.py
    python scripts/run_all_experiments.py --force-rerun
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.full_sweep import run_full_sweep
from src.utils.device import get_device


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--force-rerun", action="store_true",
        help="Re-run classifier and flow-matching experiments even if a result already "
             "exists. Already-cached features are always reused regardless of this flag.",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    run_full_sweep(
        args.data_dir, args.cache_dir, args.output_dir, device, force_rerun=args.force_rerun
    )

    print("\nRun `python scripts/generate_report.py` to build the results report.")


if __name__ == "__main__":
    main()
