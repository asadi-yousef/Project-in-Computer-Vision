"""Run the Stage 3 experiments: FM before a frozen linear classifier.

Trains both of part_3.pdf's strategies on both datasets over all three seeds,
using the hyperparameters selected by scripts/tune_stage3.py, and writes each
run's artifacts under outputs/<method>/... alongside the Stage 1 and Stage 2
runs.

Requires the Stage 1 linear-probe runs to be complete: Stage 3 reuses their
trained classifiers rather than retraining equivalents, and verifies each one
reproduces its published accuracy before training on top of it.

Safe to re-run: completed runs are skipped unless --force is given.

Usage:
    python scripts/run_stage3_experiments.py
    python scripts/run_stage3_experiments.py --methods fm_cls_guided --force
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stage3_sweep import run_stage3_sweep
from src.utils.config import (
    STAGE3_EXTENSION_METHODS,
    STAGE3_K_SHOT,
    STAGE3_METHODS,
    STAGE3_SEEDS,
    STAGE3_SETTINGS,
)
from src.utils.device import get_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=list(STAGE3_METHODS))
    parser.add_argument(
        "--extension", action="store_true",
        help="Also run part_3.pdf's optional extension: joint fine-tuning of the "
             "flow and classifier, plus the classifier-only control.",
    )
    parser.add_argument("--datasets", nargs="+", default=list(STAGE3_SETTINGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(STAGE3_SEEDS))
    parser.add_argument("--k-shot", type=int, default=STAGE3_K_SHOT)
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / "cache"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument(
        "--force", action="store_true", help="Re-run experiments that are already complete."
    )
    args = parser.parse_args()

    methods = list(args.methods)
    if args.extension:
        methods += [m for m in STAGE3_EXTENSION_METHODS if m not in methods]

    device = get_device()
    print(f"device: {device}   K={args.k_shot}   seeds: {args.seeds}")
    print("== Stage 3 experiments ==")

    results = run_stage3_sweep(
        args.cache_dir, args.output_dir, device,
        force_rerun=args.force, methods=methods, datasets=args.datasets,
        seeds=args.seeds, k_shot=args.k_shot,
    )

    print(f"\nStage 3 complete: {len(results)} runs executed.")


if __name__ == "__main__":
    main()
