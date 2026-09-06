from pathlib import Path

import pytest

from src.utils.config import (
    STAGE3_DISPLACEMENT_PENALTY,
    STAGE3_METHODS,
    STAGE3_NUM_EULER_STEPS,
    ExperimentConfig,
    FlowMatchingHyperparams,
    Stage3Hyperparams,
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


# --- Stage 3: FM before a frozen linear classifier ---


@pytest.mark.parametrize("method", ["fm_cls_rolled", "fm_cls_guided"])
def test_stage3_methods_are_valid(method):
    config = ExperimentConfig(
        dataset="dtd", encoder="dinov2_vits14", method=method, k_shot=10, seed=0
    )
    assert config.method == method


def test_stage3_methods_constant_matches_the_valid_method_list():
    # STAGE3_METHODS exists so callers can ask "is this a Stage 3 run?"
    # without repeating the pair; it must not drift from VALID_METHODS.
    from src.utils.config import VALID_METHODS

    assert set(STAGE3_METHODS) <= set(VALID_METHODS)


def test_stage3_defaults_are_the_decided_values():
    # These are the project's locked Stage 3 choices, not casual defaults:
    # T=4 throughout (part_3.pdf asks for a single T), the validation-selected
    # displacement penalty for Strategy 1, and the four Strategy 2 target
    # knobs part_3.pdf leaves open.
    hyperparams = Stage3Hyperparams()

    assert hyperparams.hidden_dims == [512, 512]
    assert hyperparams.num_euler_steps == STAGE3_NUM_EULER_STEPS == 4

    assert hyperparams.displacement_penalty == STAGE3_DISPLACEMENT_PENALTY == 0.1
    assert hyperparams.velocity_penalty == 0.0

    assert hyperparams.target_step_size == 0.1
    assert hyperparams.target_num_steps == 1
    assert hyperparams.target_refresh_epochs == 1
    assert hyperparams.normalize_target_update is True


def test_stage3_optimizer_defaults_mirror_the_earlier_stages():
    # No spec specifies an optimizer for the velocity network, so Stage 3
    # keeps the settings inherited from stage_1.pdf's suggested configuration
    # via Stage 2, unchanged. Measured across three seeds at the full epoch
    # budget, the learning rate makes no material difference to Stage 3 once
    # the displacement penalty is on, so there is no deviation to report.
    stage3 = Stage3Hyperparams()
    flow_matching = FlowMatchingHyperparams()

    assert stage3.learning_rate == flow_matching.learning_rate == 1e-3
    assert stage3.weight_decay == flow_matching.weight_decay
    assert stage3.batch_size == flow_matching.batch_size
    assert stage3.max_epochs == flow_matching.max_epochs


@pytest.mark.parametrize("num_euler_steps", [0, -1])
def test_stage3_non_positive_euler_steps_raise(num_euler_steps):
    with pytest.raises(ValueError, match="num_euler_steps"):
        Stage3Hyperparams(num_euler_steps=num_euler_steps)


def test_stage3_allows_euler_steps_outside_stage_2s_grid():
    # Unlike FlowMatchingHyperparams, Stage 3 is not restricted to {4, 12}:
    # its T is a single project-wide choice rather than a comparison grid,
    # and the unit tests need small values to stay fast.
    assert Stage3Hyperparams(num_euler_steps=2).num_euler_steps == 2


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"hidden_dims": []}, "hidden_dims"),
        ({"hidden_dims": [512, 0]}, "hidden_dims"),
        ({"displacement_penalty": -0.1}, "displacement_penalty"),
        ({"velocity_penalty": -1.0}, "velocity_penalty"),
        ({"target_step_size": 0.0}, "target_step_size"),
        ({"target_step_size": -0.1}, "target_step_size"),
        ({"target_num_steps": 0}, "target_num_steps"),
        ({"target_refresh_epochs": 0}, "target_refresh_epochs"),
    ],
)
def test_stage3_invalid_hyperparams_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Stage3Hyperparams(**kwargs)


def test_stage3_config_round_trips_through_yaml(tmp_path):
    config = ExperimentConfig(
        dataset="dtd",
        encoder="dinov2_vits14",
        method="fm_cls_guided",
        k_shot=10,
        seed=2,
        stage3=Stage3Hyperparams(
            hidden_dims=[256, 256],
            num_euler_steps=2,
            displacement_penalty=0.5,
            target_step_size=0.05,
            target_num_steps=3,
            target_refresh_epochs=5,
            normalize_target_update=False,
        ),
    )
    path = tmp_path / "config.yaml"

    save_config(config, path)
    loaded = load_config(path)

    assert loaded.method == "fm_cls_guided"
    assert loaded.stage3.hidden_dims == [256, 256]
    assert loaded.stage3.num_euler_steps == 2
    assert loaded.stage3.displacement_penalty == 0.5
    assert loaded.stage3.target_step_size == 0.05
    assert loaded.stage3.target_num_steps == 3
    assert loaded.stage3.target_refresh_epochs == 5
    assert loaded.stage3.normalize_target_update is False


def test_config_without_stage3_block_still_loads(tmp_path):
    # Every Stage 1 and Stage 2 run config already on disk predates the
    # stage3 block; they must keep loading rather than failing on a missing
    # key, since the report pipeline reads them.
    path = tmp_path / "config.yaml"
    path.write_text(
        "dataset: dtd\n"
        "encoder: dinov2_vits14\n"
        "method: fm_standard\n"
        "k_shot: 10\n"
        "seed: 0\n"
        "flow_matching:\n"
        "  hidden_dims: [512, 512]\n"
        "  num_euler_steps: 12\n"
    )

    loaded = load_config(path)

    assert loaded.flow_matching.num_euler_steps == 12
    assert loaded.stage3.num_euler_steps == STAGE3_NUM_EULER_STEPS


def test_every_saved_run_config_still_loads():
    # Integration check against the real Stage 1/2 outputs: adding the
    # stage3 block must not break a single already-completed run.
    outputs = Path(__file__).resolve().parent.parent / "outputs"
    saved = sorted(outputs.rglob("config.yaml")) if outputs.exists() else []
    if not saved:
        pytest.skip("no completed runs in outputs/ to check against")

    for path in saved:
        config = load_config(path)
        assert config.stage3.num_euler_steps == STAGE3_NUM_EULER_STEPS

