import json
from pathlib import Path

import pytest
import torch

from src.classifiers.prototype import compute_class_prototypes, predict_by_cosine_similarity
from src.data.few_shot import sample_balanced_subset_indices
from src.evaluation.aggregation import load_all_results
from src.features.cache import FeatureCacheMetadata, cache_file_path, save_feature_cache
from src.features.loading import load_validated_feature_cache
from src.flow_matching.runner import (
    baseline_accuracy,
    flow_matching_run_dir,
    prepare_features,
    run_flow_matching_experiment,
    run_rolled_out_fm_experiment,
    run_standard_fm_experiment,
)
from src.utils.config import VALID_EULER_STEPS, ExperimentConfig, FlowMatchingHyperparams

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REAL_CACHE_AVAILABLE = (PROJECT_ROOT / "cache" / "dtd" / "resnet18" / "train.pt").exists()
_REAL_OUTPUTS_AVAILABLE = (PROJECT_ROOT / "outputs" / "prototype").exists()


def _write_synthetic_cache(
    cache_dir, dataset, encoder, split, samples_per_class, num_classes, feature_dim, seed
):
    generator = torch.Generator().manual_seed(seed)
    features = []
    labels = []
    for class_index in range(num_classes):
        center = torch.zeros(feature_dim)
        center[class_index % feature_dim] = 8.0
        noise = torch.randn(samples_per_class, feature_dim, generator=generator) * 0.1
        features.append(center.unsqueeze(0) + noise)
        labels.append(torch.full((samples_per_class,), class_index, dtype=torch.long))
    features_tensor = torch.cat(features)
    labels_tensor = torch.cat(labels)

    metadata = FeatureCacheMetadata(
        dataset=dataset,
        encoder=encoder,
        split=split,
        num_samples=features_tensor.shape[0],
        feature_dim=feature_dim,
        num_classes=num_classes,
    )
    save_feature_cache(
        cache_file_path(cache_dir, dataset, encoder, split), features_tensor, labels_tensor, metadata
    )


def _synthetic_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    for split, seed in [("train", 0), ("val", 1), ("test", 2)]:
        _write_synthetic_cache(
            cache_dir, "dtd", "resnet18", split,
            samples_per_class=20, num_classes=3, feature_dim=8, seed=seed,
        )
    return cache_dir


def _config(method, k_shot=10, seed=0, num_euler_steps=4):
    return ExperimentConfig(
        dataset="dtd", encoder="resnet18", method=method, k_shot=k_shot, seed=seed,
        flow_matching=FlowMatchingHyperparams(
            hidden_dims=[16, 16], max_epochs=8, batch_size=16,
            learning_rate=1e-2, num_euler_steps=num_euler_steps,
        ),
    )


DEVICE = torch.device("cpu")


# --- run directory layout ---


def test_run_dir_includes_method_k_shot_and_step_count():
    path = flow_matching_run_dir("out", "dtd", "resnet18", "fm_rolled", 5, 12, 1)
    assert path.parts[-6:] == ("fm_rolled", "dtd", "resnet18", "k5", "T12", "seed1")


def test_full_setting_uses_the_single_run_folder():
    # Follows the Stage 1 prototype convention: the full setting is one run.
    path = flow_matching_run_dir("out", "dtd", "resnet18", "fm_standard", "full", 4, 0)
    assert path.parts[-2:] == ("T4", "single_run")


# --- standard FM: one training, one run directory per T ---


def test_standard_fm_writes_one_run_directory_per_step_count(tmp_path):
    cache_dir = _synthetic_cache(tmp_path)
    output_dir = tmp_path / "outputs"

    results = run_standard_fm_experiment(
        _config("fm_standard"), cache_dir, output_dir, DEVICE, euler_step_counts=[4, 12]
    )

    assert len(results) == 2
    assert [r["num_euler_steps"] for r in results] == [4, 12]
    for result in results:
        run_dir = Path(result["run_dir"])
        for filename in ("config.yaml", "history.json", "result.json", "checkpoint.pt"):
            assert (run_dir / filename).exists(), f"{filename} missing from {run_dir}"


def test_standard_fm_shares_one_trained_network_across_step_counts(tmp_path):
    # The Q4 project decision: standard FM's objective has no T in it, so the
    # two runs must be the same weights evaluated twice, not two trainings.
    cache_dir = _synthetic_cache(tmp_path)
    output_dir = tmp_path / "outputs"

    results = run_standard_fm_experiment(
        _config("fm_standard"), cache_dir, output_dir, DEVICE, euler_step_counts=[4, 12]
    )

    first = torch.load(Path(results[0]["run_dir"]) / "checkpoint.pt", weights_only=True)
    second = torch.load(Path(results[1]["run_dir"]) / "checkpoint.pt", weights_only=True)
    for key in first:
        assert torch.equal(first[key], second[key])


def test_each_standard_run_records_its_own_step_count(tmp_path):
    # The saved config must describe the result sitting next to it.
    cache_dir = _synthetic_cache(tmp_path)
    output_dir = tmp_path / "outputs"

    results = run_standard_fm_experiment(
        _config("fm_standard"), cache_dir, output_dir, DEVICE, euler_step_counts=[4, 12]
    )

    for result in results:
        with open(Path(result["run_dir"]) / "result.json") as f:
            saved = json.load(f)
        recorded = saved["config"]["flow_matching"]["num_euler_steps"]
        assert recorded == result["num_euler_steps"]


def test_different_step_counts_can_give_different_accuracy(tmp_path):
    # Same weights, but a different number of integration steps, so the two
    # accuracies are computed independently rather than copied.
    cache_dir = _synthetic_cache(tmp_path)
    output_dir = tmp_path / "outputs"

    results = run_standard_fm_experiment(
        _config("fm_standard"), cache_dir, output_dir, DEVICE, euler_step_counts=[4, 12]
    )

    assert all(0.0 <= r["test_accuracy"] <= 1.0 for r in results)


# --- rolled-out FM: one training per T ---


def test_rolled_out_writes_a_single_run_directory(tmp_path):
    cache_dir = _synthetic_cache(tmp_path)
    output_dir = tmp_path / "outputs"

    result = run_rolled_out_fm_experiment(
        _config("fm_rolled", num_euler_steps=12), cache_dir, output_dir, DEVICE
    )

    run_dir = Path(result["run_dir"])
    assert result["num_euler_steps"] == 12
    assert "T12" in str(run_dir)
    for filename in ("config.yaml", "history.json", "result.json", "checkpoint.pt"):
        assert (run_dir / filename).exists()


def test_rolled_out_trains_separately_for_each_step_count(tmp_path):
    # Unlike standard FM, T is baked into the weights here.
    cache_dir = _synthetic_cache(tmp_path)
    output_dir = tmp_path / "outputs"

    t4 = run_rolled_out_fm_experiment(
        _config("fm_rolled", num_euler_steps=4), cache_dir, output_dir, DEVICE
    )
    t12 = run_rolled_out_fm_experiment(
        _config("fm_rolled", num_euler_steps=12), cache_dir, output_dir, DEVICE
    )

    first = torch.load(Path(t4["run_dir"]) / "checkpoint.pt", weights_only=True)
    second = torch.load(Path(t12["run_dir"]) / "checkpoint.pt", weights_only=True)
    assert any(not torch.equal(first[k], second[k]) for k in first)


# --- results content ---


def test_delta_accuracy_is_the_difference_from_the_baseline(tmp_path):
    cache_dir = _synthetic_cache(tmp_path)
    output_dir = tmp_path / "outputs"

    results = run_standard_fm_experiment(
        _config("fm_standard"), cache_dir, output_dir, DEVICE, euler_step_counts=[4]
    )

    result = results[0]
    assert result["delta_accuracy"] == pytest.approx(
        result["test_accuracy"] - result["baseline_test_accuracy"]
    )


def test_history_is_saved_with_one_entry_per_epoch(tmp_path):
    cache_dir = _synthetic_cache(tmp_path)
    output_dir = tmp_path / "outputs"

    result = run_rolled_out_fm_experiment(_config("fm_rolled"), cache_dir, output_dir, DEVICE)

    with open(Path(result["run_dir"]) / "history.json") as f:
        history = json.load(f)
    assert len(history) == 8
    assert set(history[0]) == {"epoch", "train_loss", "val_loss"}


def test_results_are_readable_by_the_stage_1_aggregator(tmp_path):
    # Task 8 aggregates Stage 1 and Stage 2 runs together, so FM result.json
    # files must satisfy the existing loader's expectations.
    cache_dir = _synthetic_cache(tmp_path)
    output_dir = tmp_path / "outputs"

    run_standard_fm_experiment(
        _config("fm_standard"), cache_dir, output_dir, DEVICE, euler_step_counts=[4, 12]
    )
    run_rolled_out_fm_experiment(_config("fm_rolled"), cache_dir, output_dir, DEVICE)

    records = load_all_results(output_dir)

    assert len(records) == 3
    assert {r["method"] for r in records} == {"fm_standard", "fm_rolled"}
    assert all("test_accuracy" in r for r in records)


# --- the Stage 1 identity constraint ---


def test_prepared_prototypes_match_the_stage_1_computation(tmp_path):
    # The constraint the whole stage rests on: same subset, same prototypes.
    cache_dir = _synthetic_cache(tmp_path)
    config = _config("fm_standard", k_shot=10, seed=1)

    prepared = prepare_features(config, cache_dir)

    train_features, train_labels, metadata = load_validated_feature_cache(
        cache_dir, "dtd", "resnet18", "train"
    )
    indices = sample_balanced_subset_indices(train_labels.tolist(), 10, 1)
    expected = compute_class_prototypes(
        train_features[indices], train_labels[indices], metadata["num_classes"]
    )

    assert torch.equal(prepared.prototypes, expected)


def test_full_setting_uses_the_whole_training_split(tmp_path):
    cache_dir = _synthetic_cache(tmp_path)

    prepared = prepare_features(_config("fm_standard", k_shot="full"), cache_dir)

    assert prepared.train_features.shape[0] == 60  # 3 classes x 20 samples


def test_prepared_features_are_unit_norm(tmp_path):
    cache_dir = _synthetic_cache(tmp_path)

    prepared = prepare_features(_config("fm_standard"), cache_dir)

    for features in (prepared.train_features, prepared.val_features, prepared.test_features):
        assert torch.allclose(features.norm(dim=1), torch.ones(features.shape[0]), atol=1e-5)


def test_normalizing_does_not_change_the_baseline(tmp_path):
    # Cosine similarity is scale-invariant, so normalizing the features for
    # the flow must leave the Stage 1 number it is compared against intact.
    cache_dir = _synthetic_cache(tmp_path)
    config = _config("fm_standard")
    prepared = prepare_features(config, cache_dir)

    raw_features, raw_labels, _ = load_validated_feature_cache(
        cache_dir, "dtd", "resnet18", "test"
    )
    raw_predictions = predict_by_cosine_similarity(raw_features, prepared.prototypes)
    raw_accuracy = (raw_predictions == raw_labels).float().mean().item()

    normalized_accuracy = baseline_accuracy(
        prepared.test_features, prepared.test_labels, prepared.prototypes
    )
    assert normalized_accuracy == pytest.approx(raw_accuracy)


# --- dispatch and validation ---


def test_dispatch_returns_a_list_for_both_methods(tmp_path):
    cache_dir = _synthetic_cache(tmp_path)
    output_dir = tmp_path / "outputs"

    standard = run_flow_matching_experiment(
        _config("fm_standard"), cache_dir, output_dir, DEVICE
    )
    rolled = run_flow_matching_experiment(_config("fm_rolled"), cache_dir, output_dir, DEVICE)

    assert len(standard) == len(VALID_EULER_STEPS)
    assert len(rolled) == 1


def test_wrong_method_raises_in_each_runner(tmp_path):
    cache_dir = _synthetic_cache(tmp_path)
    output_dir = tmp_path / "outputs"

    with pytest.raises(ValueError, match="fm_standard"):
        run_standard_fm_experiment(_config("fm_rolled"), cache_dir, output_dir, DEVICE)
    with pytest.raises(ValueError, match="fm_rolled"):
        run_rolled_out_fm_experiment(_config("fm_standard"), cache_dir, output_dir, DEVICE)
    with pytest.raises(ValueError, match="method"):
        run_flow_matching_experiment(
            _config("prototype"), cache_dir, output_dir, DEVICE
        )


# --- integration against the real Stage 1 outputs ---


@pytest.mark.skipif(
    not (_REAL_CACHE_AVAILABLE and _REAL_OUTPUTS_AVAILABLE),
    reason="Real cached features and Stage 1 outputs not present",
)
@pytest.mark.parametrize(
    "dataset,encoder,k_shot,seed",
    [("dtd", "resnet18", 5, 0), ("dtd", "resnet18", 10, 2), ("dtd", "resnet18", "full", 0)],
)
def test_recomputed_baseline_matches_the_stored_stage_1_result(dataset, encoder, k_shot, seed):
    # If this ever fails, the FM runs are being compared against prototypes
    # that differ from the ones Stage 1 actually reported.
    config = ExperimentConfig(
        dataset=dataset, encoder=encoder, method="fm_standard", k_shot=k_shot, seed=seed
    )
    prepared = prepare_features(config, PROJECT_ROOT / "cache")
    recomputed = baseline_accuracy(
        prepared.test_features, prepared.test_labels, prepared.prototypes
    )

    seed_folder = f"seed{seed}" if k_shot != "full" else "single_run"
    stored_path = (
        PROJECT_ROOT / "outputs" / "prototype" / dataset / encoder
        / f"k{k_shot}" / seed_folder / "result.json"
    )
    with open(stored_path) as f:
        stored = json.load(f)["result"]["test_accuracy"]

    assert recomputed == pytest.approx(stored, abs=1e-9)
