"""Pure, dataset-agnostic checks that guard the "no leakage, no merging of
official splits" requirement from stage_1.pdf.

Kept separate from `datasets.py` so these checks can be unit-tested with
small synthetic path lists, without downloading any real dataset.
"""

from typing import Dict, List, Sized

# Official split sizes for our two selected datasets/protocols, used as a
# sanity check that the downloaded data and chosen partition/split settings
# match the numbers documented in stage_1.pdf and the dataset's own docs.
#   DTD partition 1: 47 classes x 40 images x {train,val,test} = 1880 each.
#   Flowers-102: fixed official split regardless of any "partition" concept.
EXPECTED_SPLIT_SIZES = {
    "dtd": {"train": 1880, "val": 1880, "test": 1880},
    "flowers102": {"train": 1020, "val": 1020, "test": 6149},
}


def get_image_paths(dataset) -> List[str]:
    """Extract the per-sample file paths of a loaded DTD/Flowers102 dataset.

    This relies on the `_image_files` attribute that both torchvision
    dataset classes happen to expose internally; there is no public API for
    this, since split-overlap checking is not something torchvision itself
    needs to support.
    """
    if not hasattr(dataset, "_image_files"):
        raise AttributeError(
            f"{type(dataset).__name__} has no `_image_files` attribute; "
            "this helper only supports the DTD and Flowers102 torchvision datasets."
        )
    return [str(path) for path in dataset._image_files]


def assert_no_split_overlap(split_to_paths: Dict[str, List[str]]) -> None:
    """Raise if any file path appears in more than one split.

    Args:
        split_to_paths: mapping from split name ("train"/"val"/"test") to
            the list of image file paths in that split.

    Raises:
        ValueError: if one or more paths are shared between splits, naming
            the offending paths and which splits they leaked between.
    """
    first_seen_in: Dict[str, str] = {}
    overlaps = []
    for split_name, paths in split_to_paths.items():
        for path in paths:
            if path in first_seen_in:
                overlaps.append((path, first_seen_in[path], split_name))
            else:
                first_seen_in[path] = split_name

    if overlaps:
        example_details = "; ".join(
            f"{path!r} appears in both {first!r} and {second!r}"
            for path, first, second in overlaps[:5]
        )
        raise ValueError(
            f"Found {len(overlaps)} image(s) shared across official splits "
            f"(e.g. {example_details})"
        )


def verify_official_split_sizes(dataset_name: str, splits: Dict[str, Sized]) -> None:
    """Raise if any split's size does not match the documented official size.

    Args:
        dataset_name: "dtd" or "flowers102".
        splits: mapping from split name to a loaded dataset (or anything
            supporting `len()`), one entry per "train"/"val"/"test".

    Raises:
        ValueError: if dataset_name is unknown, a split is missing, or any
            split's length does not match `EXPECTED_SPLIT_SIZES`.
    """
    if dataset_name not in EXPECTED_SPLIT_SIZES:
        raise ValueError(
            f"dataset_name must be one of {tuple(EXPECTED_SPLIT_SIZES)}, got {dataset_name!r}"
        )

    expected_sizes = EXPECTED_SPLIT_SIZES[dataset_name]
    for split_name, expected_size in expected_sizes.items():
        if split_name not in splits:
            raise ValueError(f"Missing split {split_name!r} for dataset {dataset_name!r}")
        actual_size = len(splits[split_name])
        if actual_size != expected_size:
            raise ValueError(
                f"{dataset_name} {split_name!r} split has {actual_size} images, "
                f"expected {expected_size}. This usually means the wrong DTD "
                "partition was used, or the dataset download is incomplete/corrupted."
            )
