"""Integrity checks guarding stage_1.pdf's experiment rules against
accidental (not just incorrect-implementation) violations.

Earlier tasks' tests mostly check "does this function behave correctly
when used correctly." These tests instead guard against misuse: does eval
mode really stop all drift/randomness in the frozen encoders, does a
classifier module accidentally depend on an encoder, and do the actual
completed experiment outputs satisfy the exact run-count protocol.
"""

from pathlib import Path

import pytest
import torch
from torchvision.models import ResNet18_Weights

from src.encoders.dinov2 import DINOv2Encoder
from src.encoders.resnet18 import ResNet18Encoder
from src.encoders.resnet18 import FEATURE_DIM as RESNET18_FEATURE_DIM
from src.encoders.dinov2 import FEATURE_DIM as DINOV2_FEATURE_DIM

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_REAL_DATA_AVAILABLE = (PROJECT_ROOT / "data" / "dtd").exists()
_REAL_CACHE_AVAILABLE = (PROJECT_ROOT / "cache" / "dtd" / "resnet18").exists()
_REAL_OUTPUTS_AVAILABLE = (PROJECT_ROOT / "outputs" / "linear_probe").exists()


# --- Frozen encoders: eval mode must stop all drift and randomness, not
# just block gradients (requires_grad=False alone doesn't guarantee this:
# BatchNorm running stats update in train mode regardless, and dropout/
# droppath layers are stochastic unless the module is truly in eval mode). ---


def test_resnet18_batchnorm_running_stats_never_drift():
    encoder = ResNet18Encoder()
    bn_layer = encoder.backbone[1]  # first BatchNorm2d in the ResNet-18 stem
    assert isinstance(bn_layer, torch.nn.BatchNorm2d)
    running_mean_before = bn_layer.running_mean.clone()

    dummy_images = torch.rand(4, 3, 224, 224)
    encoder(dummy_images)
    encoder(dummy_images)

    assert torch.equal(bn_layer.running_mean, running_mean_before)


def test_dinov2_forward_is_deterministic_across_repeated_calls():
    encoder = DINOv2Encoder()
    dummy_images = torch.rand(2, 3, 224, 224)

    first_output = encoder(dummy_images)
    second_output = encoder(dummy_images)

    assert torch.equal(first_output, second_output)


def test_resnet18_uses_the_official_checkpoint_preprocessing():
    encoder = ResNet18Encoder()
    expected_transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    assert repr(encoder.preprocess) == repr(expected_transform)


# --- Classifier code must never depend on the encoders: training/
# evaluation only ever runs on cached features, never re-invokes the image
# encoder. A static source check makes this violation loud immediately,
# rather than only showing up as a slowdown someone might not notice. ---

_CLASSIFIER_MODULE_PATHS = [
    "src/classifiers/linear_probe.py",
    "src/classifiers/linear_probe_runner.py",
    "src/classifiers/prototype.py",
    "src/classifiers/prototype_runner.py",
]


def test_classifier_modules_never_import_encoder_modules():
    for relative_path in _CLASSIFIER_MODULE_PATHS:
        source = (PROJECT_ROOT / relative_path).read_text()
        assert "src.encoders" not in source, (
            f"{relative_path} must never import an encoder module: classifier "
            "training/evaluation only ever runs on cached features."
        )


# --- Real-data checks: verify the actual completed experiment outputs and
# feature caches, not just the code in isolation. Skipped gracefully if
# this hasn't been run on a given machine (e.g. fresh clone, CI). ---


@pytest.mark.skipif(not _REAL_DATA_AVAILABLE, reason="DTD dataset not downloaded in ./data")
def test_dataset_label_indices_are_within_class_name_range():
    from src.data.datasets import get_class_names, load_dtd_split

    train_split = load_dtd_split("data", split="train", download=False)
    class_names = get_class_names(train_split)
    assert all(0 <= label < len(class_names) for label in train_split._labels)


@pytest.mark.skipif(not _REAL_CACHE_AVAILABLE, reason="Cached features not found in ./cache")
def test_real_cache_feature_dims_match_declared_encoder_constants():
    from src.features.cache import cache_file_path, load_feature_cache_raw

    expected_dims = {"resnet18": RESNET18_FEATURE_DIM, "dinov2_vits14": DINOV2_FEATURE_DIM}
    checked_any = False
    for dataset, encoder in [("dtd", "resnet18"), ("dtd", "dinov2_vits14"), ("flowers102", "resnet18")]:
        path = cache_file_path("cache", dataset, encoder, "train")
        if not path.exists():
            continue
        raw = load_feature_cache_raw(path)
        assert raw["metadata"]["feature_dim"] == expected_dims[encoder]
        assert raw["features"].shape[1] == expected_dims[encoder]
        checked_any = True
    assert checked_any


@pytest.mark.skipif(not _REAL_CACHE_AVAILABLE, reason="Cached features not found in ./cache")
def test_real_cache_num_classes_matches_known_dataset_class_counts():
    from src.features.cache import cache_file_path, load_feature_cache_raw

    expected_num_classes = {"dtd": 47, "flowers102": 102}
    checked_any = False
    for dataset, encoder in [("dtd", "resnet18"), ("dtd", "dinov2_vits14"), ("flowers102", "resnet18")]:
        path = cache_file_path("cache", dataset, encoder, "train")
        if not path.exists():
            continue
        raw = load_feature_cache_raw(path)
        assert raw["metadata"]["num_classes"] == expected_num_classes[dataset]
        checked_any = True
    assert checked_any


@pytest.mark.skipif(not _REAL_CACHE_AVAILABLE, reason="Cached features not found in ./cache")
def test_full_k_shot_training_data_is_the_entire_official_train_split():
    from src.features.loading import load_validated_feature_cache

    features, _, metadata = load_validated_feature_cache("cache", "dtd", "resnet18", "train")
    assert features.shape[0] == 1880  # DTD partition-1 official train split size
    assert metadata["num_samples"] == 1880


@pytest.mark.skipif(not _REAL_OUTPUTS_AVAILABLE, reason="No real experiment outputs found in ./outputs")
def test_real_linear_probe_runs_satisfy_the_three_seed_protocol():
    from src.evaluation.aggregation import aggregate_results, load_all_results

    records = load_all_results("outputs")
    summaries = aggregate_results(records)
    linear_probe_summaries = [s for s in summaries if s["method"] == "linear_probe"]

    assert linear_probe_summaries, "expected at least one linear_probe result"
    for summary in linear_probe_summaries:
        assert summary["num_runs"] == 3, (
            f"linear_probe {summary['dataset']}/{summary['encoder']}/k={summary['k_shot']} "
            f"has {summary['num_runs']} runs, expected exactly 3 (seeds 0,1,2)"
        )


@pytest.mark.skipif(not _REAL_OUTPUTS_AVAILABLE, reason="No real experiment outputs found in ./outputs")
def test_real_prototype_runs_satisfy_the_run_count_protocol():
    from src.evaluation.aggregation import aggregate_results, load_all_results

    records = load_all_results("outputs")
    summaries = aggregate_results(records)
    prototype_summaries = [s for s in summaries if s["method"] == "prototype"]

    assert prototype_summaries, "expected at least one prototype result"
    for summary in prototype_summaries:
        expected_runs = 1 if summary["k_shot"] == "full" else 3
        assert summary["num_runs"] == expected_runs, (
            f"prototype {summary['dataset']}/{summary['encoder']}/k={summary['k_shot']} "
            f"has {summary['num_runs']} runs, expected {expected_runs}"
        )
