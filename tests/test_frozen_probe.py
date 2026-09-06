"""Tests for Stage 3's frozen-classifier loader.

The synthetic tests build a fake completed Stage 1 run in tmp_path; the
integration tests at the bottom use this project's real Stage 1 outputs and
cached features, and skip when those are absent.
"""

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.classifiers.frozen_probe import (
    frozen_probe_accuracy,
    load_frozen_linear_probe,
    verify_stored_test_accuracy,
)
from src.classifiers.linear_probe import LinearProbe
from src.classifiers.linear_probe_runner import linear_probe_run_dir
from src.flow_matching.inference import euler_transport
from src.flow_matching.velocity_net import build_near_identity_velocity_network
from src.utils.seeding import set_seed

CPU = torch.device("cpu")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_OUTPUTS = PROJECT_ROOT / "outputs"
REAL_CACHE = PROJECT_ROOT / "cache"

# The two settings Stage 3 actually uses (decisions 1 and 2).
STAGE3_SETTINGS = [("dtd", "dinov2_vits14"), ("flowers102", "resnet18")]


def _write_fake_run(
    output_dir, dataset, encoder, k_shot, seed, feature_dim, num_classes, test_accuracy
):
    """Write a checkpoint + result.json shaped exactly like a Stage 1 run."""
    run_dir = linear_probe_run_dir(output_dir, dataset, encoder, k_shot, seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    probe = LinearProbe(feature_dim, num_classes)
    torch.save(probe.state_dict(), run_dir / "checkpoint.pt")

    with open(run_dir / "result.json", "w") as f:
        json.dump({"result": {"test_accuracy": test_accuracy}}, f)

    return run_dir, probe


def test_loads_the_checkpoint_and_its_stored_accuracy(tmp_path):
    run_dir, original = _write_fake_run(
        tmp_path, "dtd", "resnet18", 10, 0, feature_dim=16, num_classes=5,
        test_accuracy=0.4242,
    )

    frozen = load_frozen_linear_probe(tmp_path, "dtd", "resnet18", 10, 0, CPU)

    assert frozen.run_dir == run_dir
    assert frozen.stored_test_accuracy == pytest.approx(0.4242)
    assert torch.equal(frozen.model.linear.weight, original.linear.weight)
    assert torch.equal(frozen.model.linear.bias, original.linear.bias)


def test_dimensions_are_read_from_the_checkpoint(tmp_path):
    # Not passed in: a mismatch with the cached features should surface at
    # the first forward pass rather than being silently accepted.
    _write_fake_run(
        tmp_path, "dtd", "resnet18", 10, 0, feature_dim=384, num_classes=47,
        test_accuracy=0.5,
    )

    frozen = load_frozen_linear_probe(tmp_path, "dtd", "resnet18", 10, 0, CPU)

    assert frozen.feature_dim == 384
    assert frozen.num_classes == 47


def test_parameters_are_frozen_and_the_model_is_in_eval_mode(tmp_path):
    _write_fake_run(tmp_path, "dtd", "resnet18", 10, 0, 16, 5, 0.5)

    frozen = load_frozen_linear_probe(tmp_path, "dtd", "resnet18", 10, 0, CPU)

    assert not frozen.model.training
    assert all(not p.requires_grad for p in frozen.model.parameters())


def test_gradients_reach_the_flow_but_not_the_classifier(tmp_path):
    # The property Strategy 1 depends on. "Frozen" must mean the classifier
    # is not *updated*, while gradients still flow *through* it to the
    # velocity network - so freezing by requires_grad, never by no_grad or
    # detaching, which would sever the rollout's gradient path entirely.
    _write_fake_run(tmp_path, "dtd", "resnet18", 10, 0, 16, 5, 0.5)
    frozen = load_frozen_linear_probe(tmp_path, "dtd", "resnet18", 10, 0, CPU)

    velocity_net = build_near_identity_velocity_network(16, [32, 32])
    features = torch.randn(8, 16) * 24.0
    labels = torch.randint(0, 5, (8,))

    logits = frozen.model(euler_transport(velocity_net, features, num_steps=4))
    nn.functional.cross_entropy(logits, labels).backward()

    assert frozen.model.linear.weight.grad is None
    assert frozen.model.linear.bias.grad is None
    assert velocity_net.net[-1].weight.grad is not None
    assert velocity_net.net[-1].weight.grad.abs().sum() > 0


def test_missing_checkpoint_raises_a_useful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No Stage 1 checkpoint"):
        load_frozen_linear_probe(tmp_path, "dtd", "resnet18", 10, 0, CPU)


def test_missing_result_json_raises(tmp_path):
    run_dir = linear_probe_run_dir(tmp_path, "dtd", "resnet18", 10, 0)
    run_dir.mkdir(parents=True)
    torch.save(LinearProbe(16, 5).state_dict(), run_dir / "checkpoint.pt")

    with pytest.raises(FileNotFoundError, match="No Stage 1 result.json"):
        load_frozen_linear_probe(tmp_path, "dtd", "resnet18", 10, 0, CPU)


def test_a_checkpoint_that_is_not_a_linear_probe_raises(tmp_path):
    run_dir = linear_probe_run_dir(tmp_path, "dtd", "resnet18", 10, 0)
    run_dir.mkdir(parents=True)
    torch.save({"net.0.weight": torch.zeros(4, 4)}, run_dir / "checkpoint.pt")
    with open(run_dir / "result.json", "w") as f:
        json.dump({"result": {"test_accuracy": 0.5}}, f)

    with pytest.raises(KeyError, match="linear.weight"):
        load_frozen_linear_probe(tmp_path, "dtd", "resnet18", 10, 0, CPU)


def test_each_seed_loads_its_own_checkpoint(tmp_path):
    # Stage 3 pairs each run with the Stage 1 run of the same seed, so the
    # loader must not quietly return a different seed's classifier.
    for seed in (0, 1, 2):
        _write_fake_run(tmp_path, "dtd", "resnet18", 10, seed, 16, 5, 0.5 + seed / 100)

    loaded = [
        load_frozen_linear_probe(tmp_path, "dtd", "resnet18", 10, seed, CPU)
        for seed in (0, 1, 2)
    ]

    assert [f.stored_test_accuracy for f in loaded] == pytest.approx([0.50, 0.51, 0.52])
    assert not torch.equal(loaded[0].model.linear.weight, loaded[1].model.linear.weight)


def test_verify_accepts_a_matching_accuracy(tmp_path):
    features = torch.randn(64, 16)
    labels = torch.randint(0, 5, (64,))

    _write_fake_run(tmp_path, "dtd", "resnet18", 10, 0, 16, 5, test_accuracy=0.0)
    frozen = load_frozen_linear_probe(tmp_path, "dtd", "resnet18", 10, 0, CPU)
    # Rewrite result.json with the accuracy this checkpoint actually gets.
    actual = frozen_probe_accuracy(frozen, features, labels, CPU)
    frozen.stored_test_accuracy = actual

    assert verify_stored_test_accuracy(frozen, features, labels, CPU) == pytest.approx(actual)


def test_verify_rejects_a_mismatched_accuracy(tmp_path):
    features = torch.randn(64, 16)
    labels = torch.randint(0, 5, (64,))

    _write_fake_run(tmp_path, "dtd", "resnet18", 10, 0, 16, 5, test_accuracy=0.99)
    frozen = load_frozen_linear_probe(tmp_path, "dtd", "resnet18", 10, 0, CPU)

    with pytest.raises(ValueError, match="would be invalid"):
        verify_stored_test_accuracy(frozen, features, labels, CPU)


# --- Integration: this project's real Stage 1 runs ---


def _real_test_split(dataset, encoder):
    from src.features.loading import load_validated_feature_cache

    if not (REAL_CACHE / dataset / encoder / "test.pt").exists():
        pytest.skip(f"no cached features for {dataset}/{encoder}")
    features, labels, _ = load_validated_feature_cache(REAL_CACHE, dataset, encoder, "test")
    return features, labels


@pytest.mark.parametrize("dataset, encoder", STAGE3_SETTINGS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_real_stage1_checkpoints_reproduce_their_stored_accuracy(dataset, encoder, seed):
    if not linear_probe_run_dir(REAL_OUTPUTS, dataset, encoder, 10, seed).exists():
        pytest.skip(f"Stage 1 run not completed for {dataset}/{encoder} k10 seed{seed}")

    features, labels = _real_test_split(dataset, encoder)
    frozen = load_frozen_linear_probe(REAL_OUTPUTS, dataset, encoder, 10, seed, CPU)

    recomputed = verify_stored_test_accuracy(frozen, features, labels, CPU)

    assert recomputed == pytest.approx(frozen.stored_test_accuracy, abs=1e-6)


@pytest.mark.parametrize("dataset, encoder", STAGE3_SETTINGS)
def test_normalizing_features_would_fail_verification(dataset, encoder):
    # Guards decision 8. Stage 2's pipeline L2-normalizes before the flow;
    # doing that here costs several accuracy points, because the frozen
    # probe was fitted to raw features with norms of roughly 24 to 48. The
    # integrity check must catch that rather than let it pass silently.
    if not linear_probe_run_dir(REAL_OUTPUTS, dataset, encoder, 10, 0).exists():
        pytest.skip(f"Stage 1 run not completed for {dataset}/{encoder}")

    features, labels = _real_test_split(dataset, encoder)
    frozen = load_frozen_linear_probe(REAL_OUTPUTS, dataset, encoder, 10, 0, CPU)

    with pytest.raises(ValueError, match="would be invalid"):
        verify_stored_test_accuracy(
            frozen, torch.nn.functional.normalize(features, dim=1), labels, CPU
        )
