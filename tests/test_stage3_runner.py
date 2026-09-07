"""Tests for the Stage 3 experiment runner.

Synthetic caches and a synthetic Stage 1 run in tmp_path for the mechanics;
the real cached features and real Stage 1 checkpoints at the bottom, skipped
when absent.
"""

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.classifiers.linear_probe import LinearProbe
from src.classifiers.linear_probe_runner import linear_probe_run_dir
from src.data.few_shot import sample_balanced_subset_indices
from src.features.cache import cache_file_path
from src.flow_matching.stage3_runner import (
    evaluate_stage3_checkpoint,
    load_stage3_classifier,
    prepare_stage3_features,
    run_stage3_experiment,
    stage3_run_dir,
    train_stage3_method,
)
from src.utils.config import ExperimentConfig, Stage3Hyperparams
from src.utils.seeding import set_seed

CPU = torch.device("cpu")
DATASET = "dtd"
ENCODER = "resnet18"
FEATURE_DIM = 12
NUM_CLASSES = 4
FEATURE_SCALE = 24.0

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_OUTPUTS = PROJECT_ROOT / "outputs"
REAL_CACHE = PROJECT_ROOT / "cache"
STAGE3_SETTINGS = [("dtd", "dinov2_vits14"), ("flowers102", "resnet18")]


def _write_cache(cache_dir, split, num_per_class, seed):
    """Write a synthetic feature cache in the real cache format."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(NUM_CLASSES).repeat_interleave(num_per_class)
    features = torch.randn(
        len(labels), FEATURE_DIM, generator=generator
    ) * FEATURE_SCALE
    # A little class signal, so accuracy is not pinned at chance.
    features += F.one_hot(labels, FEATURE_DIM).float() * FEATURE_SCALE

    path = cache_file_path(cache_dir, DATASET, ENCODER, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features,
            "labels": labels,
            "metadata": {
                "dataset": DATASET,
                "encoder": ENCODER,
                "split": split,
                "num_samples": len(labels),
                "feature_dim": FEATURE_DIM,
                "num_classes": NUM_CLASSES,
            },
        },
        path,
    )
    return features, labels


def _write_stage1_run(output_dir, cache_dir, k_shot, seed):
    """Write a Stage 1 run whose stored accuracy matches its own checkpoint."""
    set_seed(seed)
    probe = LinearProbe(FEATURE_DIM, NUM_CLASSES)

    test_cache = torch.load(cache_file_path(cache_dir, DATASET, ENCODER, "test"), weights_only=False)
    with torch.no_grad():
        logits = probe(test_cache["features"])
        accuracy = (logits.argmax(dim=1) == test_cache["labels"]).float().mean().item()

    run_dir = linear_probe_run_dir(output_dir, DATASET, ENCODER, k_shot, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(probe.state_dict(), run_dir / "checkpoint.pt")
    with open(run_dir / "result.json", "w") as f:
        json.dump({"result": {"test_accuracy": accuracy}}, f)
    return accuracy


@pytest.fixture
def project(tmp_path):
    """A tiny complete project: caches for all splits plus a Stage 1 run."""
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "outputs"
    _write_cache(cache_dir, "train", num_per_class=8, seed=1)
    _write_cache(cache_dir, "val", num_per_class=4, seed=2)
    _write_cache(cache_dir, "test", num_per_class=4, seed=3)
    baseline = _write_stage1_run(output_dir, cache_dir, k_shot=5, seed=0)
    return cache_dir, output_dir, baseline


def _config(method="fm_cls_rolled", seed=0, **stage3):
    settings = dict(hidden_dims=[16, 16], num_euler_steps=2, max_epochs=3, batch_size=8)
    settings.update(stage3)
    return ExperimentConfig(
        dataset=DATASET, encoder=ENCODER, method=method, k_shot=5, seed=seed,
        stage3=Stage3Hyperparams(**settings),
    )


# --- prepare_stage3_features ---


def test_features_are_not_normalized(project):
    # Decision 8, and the property verify_stored_test_accuracy exists to
    # protect: Stage 1's probe was fitted to raw features.
    cache_dir, _, _ = project

    data = prepare_stage3_features(_config(), cache_dir)

    norms = data.train_features.norm(dim=1)
    assert norms.mean() > 1.5  # nowhere near unit norm
    assert not torch.allclose(norms, torch.ones_like(norms))


def test_the_subset_is_the_one_stage_1_sampled(project):
    # part_3.pdf requires the same sampled training subsets, and the whole
    # paired comparison depends on it.
    cache_dir, _, _ = project
    config = _config(seed=2)

    data = prepare_stage3_features(config, cache_dir)

    raw = torch.load(cache_file_path(cache_dir, DATASET, ENCODER, "train"), weights_only=False)
    expected_indices = sample_balanced_subset_indices(raw["labels"].tolist(), 5, 2)
    assert torch.equal(data.train_features, raw["features"][expected_indices])
    assert torch.equal(data.train_labels, raw["labels"][expected_indices])


def test_the_subset_is_balanced_and_the_other_splits_are_complete(project):
    cache_dir, _, _ = project

    data = prepare_stage3_features(_config(), cache_dir)

    assert data.train_features.shape[0] == 5 * NUM_CLASSES
    assert torch.bincount(data.train_labels).tolist() == [5] * NUM_CLASSES
    assert data.val_features.shape[0] == 4 * NUM_CLASSES
    assert data.test_features.shape[0] == 4 * NUM_CLASSES
    assert data.num_classes == NUM_CLASSES
    assert data.feature_dim == FEATURE_DIM


def test_full_k_shot_uses_the_whole_training_split(project):
    cache_dir, _, _ = project
    config = _config()
    config.k_shot = "full"

    data = prepare_stage3_features(config, cache_dir)

    assert data.train_features.shape[0] == 8 * NUM_CLASSES


def test_different_seeds_select_different_subsets(project):
    cache_dir, _, _ = project

    first = prepare_stage3_features(_config(seed=0), cache_dir)
    second = prepare_stage3_features(_config(seed=1), cache_dir)

    assert not torch.equal(first.train_features, second.train_features)


# --- load_stage3_classifier ---


def test_the_classifier_is_verified_against_its_stored_accuracy(project):
    cache_dir, output_dir, baseline = project
    config = _config()
    data = prepare_stage3_features(config, cache_dir)

    frozen = load_stage3_classifier(config, data, output_dir, CPU)

    assert frozen.stored_test_accuracy == pytest.approx(baseline)
    assert all(not p.requires_grad for p in frozen.model.parameters())


def test_a_drifted_stored_accuracy_stops_the_run(project):
    # The guard that makes every delta trustworthy.
    cache_dir, output_dir, _ = project
    config = _config()
    run_dir = linear_probe_run_dir(output_dir, DATASET, ENCODER, 5, 0)
    with open(run_dir / "result.json", "w") as f:
        json.dump({"result": {"test_accuracy": 0.999}}, f)

    data = prepare_stage3_features(config, cache_dir)

    with pytest.raises(ValueError, match="would be invalid"):
        load_stage3_classifier(config, data, output_dir, CPU)


# --- train_stage3_method ---


@pytest.mark.parametrize("method", ["fm_cls_rolled", "fm_cls_guided"])
def test_both_strategies_dispatch_and_train(project, method):
    cache_dir, output_dir, _ = project
    config = _config(method=method)
    data = prepare_stage3_features(config, cache_dir)
    frozen = load_stage3_classifier(config, data, output_dir, CPU)

    result = train_stage3_method(
        method, data, frozen.model, config.stage3, config.seed, CPU
    )

    assert len(result.history) == 3
    assert 1 <= result.best_epoch <= 3


def test_an_unknown_method_raises(project):
    cache_dir, output_dir, _ = project
    config = _config()
    data = prepare_stage3_features(config, cache_dir)
    frozen = load_stage3_classifier(config, data, output_dir, CPU)

    with pytest.raises(ValueError, match="fm_cls_rolled"):
        train_stage3_method(
            "fm_standard", data, frozen.model, config.stage3, 0, CPU
        )


# --- evaluate_stage3_checkpoint ---


def test_an_untrained_checkpoint_scores_exactly_the_baseline(project):
    # The near-identity property, at the runner's level.
    from src.flow_matching.velocity_net import build_near_identity_velocity_network

    cache_dir, output_dir, baseline = project
    config = _config()
    data = prepare_stage3_features(config, cache_dir)
    frozen = load_stage3_classifier(config, data, output_dir, CPU)
    untrained = build_near_identity_velocity_network(FEATURE_DIM, [16, 16])

    _, accuracy, displacement = evaluate_stage3_checkpoint(
        untrained.state_dict(), [16, 16], frozen.model,
        data.test_features, data.test_labels, 2, CPU,
    )

    assert accuracy == pytest.approx(baseline)
    assert displacement == pytest.approx(0.0)


# --- run_stage3_experiment ---


@pytest.mark.parametrize("method", ["fm_cls_rolled", "fm_cls_guided"])
def test_end_to_end_writes_every_artifact(project, method):
    cache_dir, output_dir, baseline = project
    config = _config(method=method)

    result = run_stage3_experiment(config, cache_dir, output_dir, CPU)

    run_dir = Path(result["run_dir"])
    assert run_dir == stage3_run_dir(output_dir, DATASET, ENCODER, method, 5, 2, 0)
    for name in ("config.yaml", "history.json", "result.json", "checkpoint.pt"):
        assert (run_dir / name).exists()

    with open(run_dir / "history.json") as f:
        assert len(json.load(f)) == 3


def test_the_saved_result_is_shaped_like_a_stage_2_result(project):
    # src.evaluation.aggregation reads these keys; Stage 3 must not need a
    # special case there.
    cache_dir, output_dir, baseline = project

    result = run_stage3_experiment(_config(), cache_dir, output_dir, CPU)

    for key in ("test_accuracy", "baseline_test_accuracy", "delta_accuracy", "num_euler_steps"):
        assert key in result
    assert result["baseline_test_accuracy"] == pytest.approx(baseline)
    assert result["delta_accuracy"] == pytest.approx(
        result["test_accuracy"] - baseline
    )
    assert result["num_euler_steps"] == 2


def test_the_saved_config_records_the_stage3_settings(project):
    from src.utils.config import load_config

    cache_dir, output_dir, _ = project
    config = _config(target_step_size=0.25, displacement_penalty=0.5)

    result = run_stage3_experiment(config, cache_dir, output_dir, CPU)

    saved = load_config(Path(result["run_dir"]) / "config.yaml")
    assert saved.method == "fm_cls_rolled"
    assert saved.stage3.target_step_size == 0.25
    assert saved.stage3.displacement_penalty == 0.5
    assert saved.stage3.num_euler_steps == 2


def test_the_checkpoint_reproduces_the_reported_accuracy(project):
    cache_dir, output_dir, _ = project
    config = _config()

    result = run_stage3_experiment(config, cache_dir, output_dir, CPU)

    data = prepare_stage3_features(config, cache_dir)
    frozen = load_stage3_classifier(config, data, output_dir, CPU)
    state_dict = torch.load(Path(result["run_dir"]) / "checkpoint.pt", weights_only=True)
    _, accuracy, _ = evaluate_stage3_checkpoint(
        state_dict, config.stage3.hidden_dims, frozen.model,
        data.test_features, data.test_labels, 2, CPU,
    )

    assert accuracy == pytest.approx(result["test_accuracy"])


def test_a_non_stage3_method_raises(project):
    cache_dir, output_dir, _ = project
    config = ExperimentConfig(
        dataset=DATASET, encoder=ENCODER, method="linear_probe", k_shot=5, seed=0
    )

    with pytest.raises(ValueError, match="fm_cls_rolled"):
        run_stage3_experiment(config, cache_dir, output_dir, CPU)


def test_a_missing_stage1_run_raises(project, tmp_path):
    cache_dir, _, _ = project

    with pytest.raises(FileNotFoundError, match="No Stage 1 checkpoint"):
        run_stage3_experiment(_config(), cache_dir, tmp_path / "empty", CPU)


def test_the_progress_callback_is_forwarded(project):
    cache_dir, output_dir, _ = project
    seen = []

    run_stage3_experiment(_config(), cache_dir, output_dir, CPU, progress=seen.append)

    assert [log.epoch for log in seen] == [1, 2, 3]


# --- Integration: the real project ---


@pytest.mark.parametrize("dataset, encoder", STAGE3_SETTINGS)
def test_real_features_pair_with_the_real_stage_1_baseline(dataset, encoder):
    # Confirms the runner reproduces Stage 1's published number as its own
    # baseline, on the settings Stage 3 actually uses. No training here -
    # that is the sweep's job.
    if not (REAL_CACHE / dataset / encoder / "test.pt").exists():
        pytest.skip(f"no cached features for {dataset}/{encoder}")
    if not linear_probe_run_dir(REAL_OUTPUTS, dataset, encoder, 10, 0).exists():
        pytest.skip(f"Stage 1 run not completed for {dataset}/{encoder}")

    config = ExperimentConfig(
        dataset=dataset, encoder=encoder, method="fm_cls_rolled", k_shot=10, seed=0
    )
    data = prepare_stage3_features(config, REAL_CACHE)
    frozen = load_stage3_classifier(config, data, REAL_OUTPUTS, CPU)

    assert data.train_features.shape[0] == 10 * data.num_classes
    assert frozen.feature_dim == data.feature_dim
    assert frozen.num_classes == data.num_classes
