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

from src.data.datasets import VALID_SPLITS, get_class_names, load_dataset_splits
from src.encoders.dinov2 import DINOv2Encoder
from src.encoders.resnet18 import ResNet18Encoder
from src.features.cache import FeatureCacheMetadata, cache_file_path, save_feature_cache
from src.features.extraction import extract_features
from src.utils.device import get_device

_ENCODER_BUILDERS = {
    "resnet18": ResNet18Encoder,
    "dinov2_vits14": DINOv2Encoder,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["dtd", "flowers102"])
    parser.add_argument("--encoder", required=True, choices=list(_ENCODER_BUILDERS))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.encoder == "dinov2_vits14" and args.dataset != "dtd":
        raise ValueError("dinov2_vits14 is only used on 'dtd' in this project")

    device = get_device()
    print(f"Using device: {device}")

    print(f"Building encoder {args.encoder!r} ...")
    encoder = _ENCODER_BUILDERS[args.encoder]()

    splits = load_dataset_splits(
        args.dataset, args.data_dir, download=False, transform=encoder.preprocess
    )
    num_classes = len(get_class_names(splits["train"]))

    for split_name in VALID_SPLITS:
        print(f"Extracting {args.dataset}/{args.encoder}/{split_name} "
              f"({len(splits[split_name])} images) ...")
        features, labels = extract_features(
            encoder, splits[split_name], device=device, batch_size=args.batch_size
        )

        metadata = FeatureCacheMetadata(
            dataset=args.dataset,
            encoder=args.encoder,
            split=split_name,
            num_samples=features.shape[0],
            feature_dim=features.shape[1],
            num_classes=num_classes,
        )
        path = cache_file_path(args.cache_dir, args.dataset, args.encoder, split_name)
        save_feature_cache(path, features, labels, metadata)
        print(f"  saved {features.shape[0]} x {features.shape[1]} features to {path}")


if __name__ == "__main__":
    main()
