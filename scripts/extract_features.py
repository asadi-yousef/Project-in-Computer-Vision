"""Extract and cache frozen-encoder features for one dataset's official splits.

Usage:
    python scripts/extract_features.py --dataset dtd --encoder resnet18
    python scripts/extract_features.py --dataset dtd --encoder dinov2_vits14
    python scripts/extract_features.py --dataset flowers102 --encoder resnet18
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.pipeline import ENCODER_BUILDERS, extract_and_cache_all_splits
from src.utils.device import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["dtd", "flowers102"])
    parser.add_argument("--encoder", required=True, choices=list(ENCODER_BUILDERS))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    extract_and_cache_all_splits(
        args.dataset, args.encoder, args.data_dir, args.cache_dir, device, args.batch_size
    )


if __name__ == "__main__":
    main()
