import pytest

from src.data.datasets import (
    get_class_names,
    load_dataset_splits,
    load_dtd_split,
    load_flowers102_split,
)


class _FakeDatasetWithClasses:
    def __init__(self, classes):
        self.classes = classes


class _FakeDatasetWithoutClasses:
    pass


def test_get_class_names_returns_the_classes_list():
    dataset = _FakeDatasetWithClasses(["a", "b", "c"])
    assert get_class_names(dataset) == ["a", "b", "c"]


def test_get_class_names_raises_for_unsupported_dataset():
    with pytest.raises(AttributeError, match="classes"):
        get_class_names(_FakeDatasetWithoutClasses())


def test_load_dtd_split_rejects_invalid_split_name(tmp_path):
    with pytest.raises(ValueError, match="split"):
        load_dtd_split(tmp_path, split="bogus")


def test_load_flowers102_split_rejects_invalid_split_name(tmp_path):
    with pytest.raises(ValueError, match="split"):
        load_flowers102_split(tmp_path, split="bogus")


def test_load_dataset_splits_rejects_unknown_dataset_name(tmp_path):
    with pytest.raises(ValueError, match="dataset_name"):
        load_dataset_splits("mnist", tmp_path)
