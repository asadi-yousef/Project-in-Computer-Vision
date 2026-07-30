import json

import torch

from src.utils.config import ExperimentConfig
from src.utils.run_metadata import build_run_metadata, save_run_metadata


def _example_config() -> ExperimentConfig:
    return ExperimentConfig(
        dataset="dtd", encoder="resnet18", method="linear_probe", k_shot=10, seed=0
    )


def test_build_run_metadata_contains_expected_keys():
    metadata = build_run_metadata(_example_config(), torch.device("cpu"))

    for key in (
        "timestamp_utc",
        "git_commit",
        "python_version",
        "torch_version",
        "platform",
        "device",
        "config",
    ):
        assert key in metadata

    assert metadata["device"] == "cpu"
    assert metadata["config"]["dataset"] == "dtd"
    assert metadata["config"]["encoder"] == "resnet18"


def test_save_run_metadata_writes_valid_json(tmp_path):
    metadata = build_run_metadata(_example_config(), torch.device("cpu"))
    path = tmp_path / "run_metadata.json"

    save_run_metadata(metadata, path)

    with open(path) as f:
        loaded = json.load(f)
    assert loaded["config"]["method"] == "linear_probe"
