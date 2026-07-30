"""Loading DTD and Oxford Flowers-102 through their official splits.

Both datasets are provided by torchvision with a `split` argument backed by
each dataset's own official train/val/test index files - no manual split
logic is written here, since re-deriving splits ourselves would risk
diverging from the official protocol the spec requires.
"""

from pathlib import Path
from typing import Callable, Optional, Union

from torch.utils.data import Dataset
from torchvision.datasets import DTD, Flowers102

VALID_SPLITS = ("train", "val", "test")

# stage_1.pdf requires DTD official partition 1 specifically.
DTD_PARTITION = 1


def load_dtd_split(
    root: Union[str, Path],
    split: str,
    download: bool = False,
    transform: Optional[Callable] = None,
) -> DTD:
    """Load one official split of DTD, partition 1.

    Args:
        root: directory holding (or to hold, if download=True) the raw DTD files.
        split: one of "train", "val", "test".
        download: if True, download the dataset when not already present.
        transform: optional image transform (e.g. an encoder's preprocessing).

    Returns:
        A torchvision DTD dataset restricted to the requested split.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
    return DTD(
        root=str(root),
        split=split,
        partition=DTD_PARTITION,
        download=download,
        transform=transform,
    )


def load_flowers102_split(
    root: Union[str, Path],
    split: str,
    download: bool = False,
    transform: Optional[Callable] = None,
) -> Flowers102:
    """Load one official split of Oxford Flowers-102.

    Args:
        root: directory holding (or to hold, if download=True) the raw Flowers-102 files.
        split: one of "train", "val", "test".
        download: if True, download the dataset when not already present.
        transform: optional image transform (e.g. an encoder's preprocessing).

    Returns:
        A torchvision Flowers102 dataset restricted to the requested split.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
    return Flowers102(root=str(root), split=split, download=download, transform=transform)


_LOADERS = {
    "dtd": load_dtd_split,
    "flowers102": load_flowers102_split,
}


def load_dataset_splits(
    dataset_name: str,
    root: Union[str, Path],
    download: bool = False,
    transform: Optional[Callable] = None,
) -> dict:
    """Load all three official splits for a project dataset by name.

    Args:
        dataset_name: "dtd" or "flowers102".
        root: directory holding (or to hold) the raw dataset files.
        download: if True, download the dataset when not already present.
        transform: optional image transform, applied identically to all splits.

    Returns:
        A dict {"train": Dataset, "val": Dataset, "test": Dataset}.
    """
    if dataset_name not in _LOADERS:
        raise ValueError(f"dataset_name must be one of {tuple(_LOADERS)}, got {dataset_name!r}")
    loader = _LOADERS[dataset_name]
    return {split: loader(root, split, download=download, transform=transform) for split in VALID_SPLITS}


def get_class_names(dataset: Dataset) -> list:
    """Return the human-readable class names for a loaded DTD/Flowers102 dataset,
    in label-index order (index i is the name for label i)."""
    if not hasattr(dataset, "classes"):
        raise AttributeError(
            f"{type(dataset).__name__} has no `.classes` attribute; "
            "this helper only supports the DTD and Flowers102 torchvision datasets."
        )
    return list(dataset.classes)
