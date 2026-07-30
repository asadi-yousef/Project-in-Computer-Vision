import pytest

from src.utils.config import ExperimentConfig, load_config, save_config


def test_round_trips_through_yaml(tmp_path):
    config = ExperimentConfig(
        dataset="dtd", encoder="resnet18", method="linear_probe", k_shot=10, seed=0
    )
    path = tmp_path / "config.yaml"

    save_config(config, path)
    loaded = load_config(path)

    assert loaded.dataset == config.dataset
    assert loaded.encoder == config.encoder
    assert loaded.method == config.method
    assert loaded.k_shot == config.k_shot
    assert loaded.seed == config.seed
    assert loaded.linear_probe.learning_rate == config.linear_probe.learning_rate
    assert loaded.paths.cache_dir == config.paths.cache_dir


def test_full_k_shot_is_valid():
    config = ExperimentConfig(
        dataset="flowers102", encoder="resnet18", method="prototype", k_shot="full", seed=0
    )
    assert config.k_shot == "full"


def test_invalid_dataset_raises():
    with pytest.raises(ValueError, match="dataset"):
        ExperimentConfig(
            dataset="mnist", encoder="resnet18", method="linear_probe", k_shot=10, seed=0
        )


def test_invalid_k_shot_raises():
    with pytest.raises(ValueError, match="k_shot"):
        ExperimentConfig(
            dataset="dtd", encoder="resnet18", method="linear_probe", k_shot=7, seed=0
        )


def test_dinov2_on_wrong_dataset_raises():
    with pytest.raises(ValueError, match="dinov2_vits14"):
        ExperimentConfig(
            dataset="flowers102",
            encoder="dinov2_vits14",
            method="linear_probe",
            k_shot=10,
            seed=0,
        )


def test_dinov2_on_dtd_is_valid():
    config = ExperimentConfig(
        dataset="dtd", encoder="dinov2_vits14", method="linear_probe", k_shot=10, seed=0
    )
    assert config.encoder == "dinov2_vits14"


def test_negative_seed_raises():
    with pytest.raises(ValueError, match="seed"):
        ExperimentConfig(
            dataset="dtd", encoder="resnet18", method="linear_probe", k_shot=10, seed=-1
        )
