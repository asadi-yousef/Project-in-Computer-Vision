"""Load a dataset's official splits and verify they satisfy stage_1.pdf's
protocol: correct split sizes, no overlap between splits, all classes present.

Usage:
    python scripts/verify_dataset_splits.py --dataset dtd --data-dir data
    python scripts/verify_dataset_splits.py --dataset flowers102 --data-dir data --download
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.datasets import get_class_names, load_dataset_splits, VALID_SPLITS
from src.data.split_integrity import (
    assert_no_split_overlap,
    get_image_paths,
    verify_official_split_sizes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["dtd", "flowers102"])
    parser.add_argument("--data-dir", default="data", help="Directory holding raw dataset files.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the dataset into --data-dir if not already present.",
    )
    args = parser.parse_args()

    print(f"Loading official splits for {args.dataset!r} from {args.data_dir!r} "
          f"(download={args.download}) ...")
    splits = load_dataset_splits(args.dataset, args.data_dir, download=args.download)

    for split_name in VALID_SPLITS:
        print(f"  {split_name}: {len(splits[split_name])} images")

    class_names = get_class_names(splits["train"])
    print(f"  classes: {len(class_names)}")

    print("Checking official split sizes match the documented protocol ...")
    verify_official_split_sizes(args.dataset, splits)
    print("  OK")

    print("Checking for image leakage between splits ...")
    split_to_paths = {name: get_image_paths(dataset) for name, dataset in splits.items()}
    assert_no_split_overlap(split_to_paths)
    print("  OK, no overlap found")

    print(f"All checks passed for {args.dataset!r}.")


if __name__ == "__main__":
    main()
