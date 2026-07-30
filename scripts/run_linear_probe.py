"""Train and evaluate one linear-probe run: one (dataset, encoder, k_shot, seed)
combination.

Usage:
    python scripts/run_linear_probe.py --dataset dtd --encoder resnet18 --k-shot 10 --seed 0
    python scripts/run_linear_probe.py --dataset dtd --encoder resnet18 --k-shot full --seed 0

Per stage_1.pdf, each training-set size is run 3 times: for k_shot in {5, 10}
using seeds {0, 1, 2} (which also select the balanced subset, per this
project's seed convention - see src/utils/config.py); for k_shot="full" using
3 classifier-initialization seeds {0, 1, 2}.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifiers.linear_probe_runner import run_linear_probe_experiment
from src.utils.config import ExperimentConfig, LinearProbeHyperparams
from src.utils.device import get_device


def parse_k_shot(value: str):
    return "full" if value == "full" else int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["dtd", "flowers102"])
    parser.add_argument("--encoder", required=True, choices=["resnet18", "dinov2_vits14"])
    parser.add_argument("--k-shot", required=True, type=parse_k_shot, help="5, 10, or 'full'")
    parser.add_argument("--seed", required=True, type=int, help="0, 1, or 2")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--output-dir", default="outputs/linear_probe")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=200)
    args = parser.parse_args()

    config = ExperimentConfig(
        dataset=args.dataset,
        encoder=args.encoder,
        method="linear_probe",
        k_shot=args.k_shot,
        seed=args.seed,
        linear_probe=LinearProbeHyperparams(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
        ),
    )

    device = get_device()
    print(f"Using device: {device}")
    print(f"Config: dataset={config.dataset} encoder={config.encoder} "
          f"k_shot={config.k_shot} seed={config.seed}")

    result = run_linear_probe_experiment(config, args.cache_dir, args.output_dir, device)

    print(f"Best epoch: {result['best_epoch']}, best val accuracy: {result['best_val_accuracy']:.4f}")
    print(f"Test accuracy (best-val checkpoint): {result['test_accuracy']:.4f}")
    print(f"Saved run outputs to {result['run_dir']}")


if __name__ == "__main__":
    main()
