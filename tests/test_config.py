import pytest

from src.utils.config import (
    ExperimentConfig,
    FlowMatchingHyperparams,
    load_config,
    save_config,
)


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


# --- Stage 2: flow-matching methods and hyperparameters ---


@pytest.mark.parametrize("method", ["fm_standard", "fm_rolled"])
def test_flow_matching_methods_are_valid(method):
    config = ExperimentConfig(
        dataset="dtd", encoder="resnet18", method=method, k_shot=5, seed=0
    )
    assert config.method == method


def test_invalid_method_still_raises():
    with pytest.raises(ValueError, match="method"):
        ExperimentConfig(
            dataset="dtd", encoder="resnet18", method="fm_typo", k_shot=5, seed=0
        )


def test_flow_matching_defaults_match_the_spec_suggestion():
    # stage_2.pdf suggests 2 hidden layers of width ~512; the optimizer
    # defaults intentionally mirror the Stage 1 linear probe.
    hyperparams = FlowMatchingHyperparams()
    assert hyperparams.hidden_dims == [512, 512]
    assert hyperparams.num_euler_steps == 4
    assert hyperparams.learning_rate == 1e-3
    assert hyperparams.weight_decay == 1e-4


@pytest.mark.parametrize("num_euler_steps", [4, 12])
def test_valid_euler_step_counts(num_euler_steps):
    hyperparams = FlowMatchingHyperparams(num_euler_steps=num_euler_steps)
    assert hyperparams.num_euler_steps == num_euler_steps


@pytest.mark.parametrize("num_euler_steps", [0, 1, 8, -4])
def test_euler_steps_outside_the_grid_raise(num_euler_steps):
    # T is restricted to the two values stage_2.pdf evaluates, so a typo
    # cannot silently produce a run outside the comparison grid.
    with pytest.raises(ValueError, match="num_euler_steps"):
        FlowMatchingHyperparams(num_euler_steps=num_euler_steps)


def test_empty_hidden_dims_raises():
    with pytest.raises(ValueError, match="hidden_dims"):
        FlowMatchingHyperparams(hidden_dims=[])


def test_non_positive_hidden_dim_raises():
    with pytest.raises(ValueError, match="hidden_dims"):
        FlowMatchingHyperparams(hidden_dims=[512, 0])


def test_flow_matching_config_round_trips_through_yaml(tmp_path):
    config = ExperimentConfig(
        dataset="dtd",
        encoder="dinov2_vits14",
        method="fm_rolled",
        k_shot="full",
        seed=0,
        flow_matching=FlowMatchingHyperparams(hidden_dims=[256, 256], num_euler_steps=12),
    )
    path = tmp_path / "config.yaml"

    save_config(config, path)
    loaded = load_config(path)

    assert loaded.method == "fm_rolled"
    assert loaded.flow_matching.hidden_dims == [256, 256]
    assert loaded.flow_matching.num_euler_steps == 12
    assert loaded.flow_matching.max_epochs == config.flow_matching.max_epochs


def test_stage_1_config_without_flow_matching_block_still_loads(tmp_path):
    # Stage 1's already-saved run configs predate the flow_matching block;
    # they must keep loading rather than failing on a missing key.
    path = tmp_path / "config.yaml"
    path.write_text(
        "dataset: dtd\n"
        "encoder: resnet18\n"
        "method: prototype\n"
        "k_shot: full\n"
        "seed: 0\n"
    )

    loaded = load_config(path)

    assert loaded.method == "prototype"
    assert loaded.flow_matching.hidden_dims == [512, 512]
