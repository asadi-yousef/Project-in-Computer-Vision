"""Evaluate the image-derived-prototype classifier for one (dataset, encoder,
k_shot) setting.

Usage:
    python scripts/run_prototype.py --dataset dtd --encoder resnet18 --k-shot 10 --seed 0
    python scripts/run_prototype.py --dataset dtd --encoder resnet18 --k-shot full

Per stage_1.pdf: k_shot in {5, 10} uses the 3 subset seeds {0,1,2}; the full
setting requires exactly one run (no training occurs, so there is nothing
for a seed to control).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifiers.prototype_runner import run_prototype_experiment
from src.utils.config import ExperimentConfig


def parse_k_shot(value: str):
    return "full" if value == "full" else int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["dtd", "flowers102"])
    parser.add_argument("--encoder", required=True, choices=["resnet18", "dinov2_vits14"])
    parser.add_argument("--k-shot", required=True, type=parse_k_shot, help="5, 10, or 'full'")
    parser.add_argument("--seed", type=int, default=None, help="Required for k-shot 5/10; ignored for 'full'")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--output-dir", default="outputs/prototype")
    args = parser.parse_args()

    if args.k_shot != "full" and args.seed is None:
        raise ValueError("--seed is required when --k-shot is 5 or 10")

    config = ExperimentConfig(
        dataset=args.dataset,
        encoder=args.encoder,
        method="prototype",
        k_shot=args.k_shot,
        seed=args.seed if args.seed is not None else 0,
    )

    print(f"Config: dataset={config.dataset} encoder={config.encoder} k_shot={config.k_shot}")

    result = run_prototype_experiment(config, args.cache_dir, args.output_dir)

    print(f"Test accuracy: {result['test_accuracy']:.4f}")
    print(f"Saved run outputs to {result['run_dir']}")


if __name__ == "__main__":
    main()
