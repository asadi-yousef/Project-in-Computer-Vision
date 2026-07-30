import torch

from src.classifiers.linear_probe import LinearProbe, evaluate_linear_probe, train_linear_probe
from src.utils.config import LinearProbeHyperparams


def _make_synthetic_linearly_separable_data(num_classes=3, samples_per_class=20, feature_dim=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    centers = torch.eye(num_classes, feature_dim) * 10.0
    features = []
    labels = []
    for class_index in range(num_classes):
        noise = torch.randn(samples_per_class, feature_dim, generator=generator) * 0.1
        features.append(centers[class_index].unsqueeze(0) + noise)
        labels.append(torch.full((samples_per_class,), class_index, dtype=torch.long))
    return torch.cat(features), torch.cat(labels)


def test_linear_probe_output_shape():
    model = LinearProbe(feature_dim=8, num_classes=5)
    logits = model(torch.randn(3, 8))
    assert logits.shape == (3, 5)


def test_training_reduces_loss_and_reaches_high_val_accuracy_on_separable_data():
    train_features, train_labels = _make_synthetic_linearly_separable_data(seed=0)
    val_features, val_labels = _make_synthetic_linearly_separable_data(seed=1)
    hyperparams = LinearProbeHyperparams(
        learning_rate=1e-2, weight_decay=0.0, batch_size=16, max_epochs=30
    )

    result = train_linear_probe(
        train_features, train_labels, val_features, val_labels,
        num_classes=3, hyperparams=hyperparams, seed=0, device=torch.device("cpu"),
    )

    assert result.history[0].train_loss > result.history[-1].train_loss
    assert result.best_val_accuracy > 0.9
    assert 1 <= result.best_epoch <= 30
    assert len(result.history) == 30


def test_same_seed_gives_identical_training_history():
    train_features, train_labels = _make_synthetic_linearly_separable_data(seed=0)
    val_features, val_labels = _make_synthetic_linearly_separable_data(seed=1)
    hyperparams = LinearProbeHyperparams(max_epochs=5, batch_size=16)

    result_a = train_linear_probe(
        train_features, train_labels, val_features, val_labels, 3, hyperparams,
        seed=42, device=torch.device("cpu"),
    )
    result_b = train_linear_probe(
        train_features, train_labels, val_features, val_labels, 3, hyperparams,
        seed=42, device=torch.device("cpu"),
    )

    assert [e.train_loss for e in result_a.history] == [e.train_loss for e in result_b.history]
    assert [e.val_accuracy for e in result_a.history] == [e.val_accuracy for e in result_b.history]


def test_different_seeds_generally_diverge():
    train_features, train_labels = _make_synthetic_linearly_separable_data(seed=0)
    val_features, val_labels = _make_synthetic_linearly_separable_data(seed=1)
    hyperparams = LinearProbeHyperparams(max_epochs=3, batch_size=16)

    result_a = train_linear_probe(
        train_features, train_labels, val_features, val_labels, 3, hyperparams,
        seed=0, device=torch.device("cpu"),
    )
    result_b = train_linear_probe(
        train_features, train_labels, val_features, val_labels, 3, hyperparams,
        seed=1, device=torch.device("cpu"),
    )

    assert [e.train_loss for e in result_a.history] != [e.train_loss for e in result_b.history]


def test_evaluate_linear_probe_matches_best_val_accuracy_when_evaluated_on_val_set():
    train_features, train_labels = _make_synthetic_linearly_separable_data(seed=0)
    val_features, val_labels = _make_synthetic_linearly_separable_data(seed=1)
    hyperparams = LinearProbeHyperparams(max_epochs=20, batch_size=16)

    result = train_linear_probe(
        train_features, train_labels, val_features, val_labels, 3, hyperparams,
        seed=0, device=torch.device("cpu"),
    )
    recomputed_accuracy = evaluate_linear_probe(
        result.best_state_dict, feature_dim=4, num_classes=3,
        test_features=val_features, test_labels=val_labels, device=torch.device("cpu"),
    )

    assert abs(recomputed_accuracy - result.best_val_accuracy) < 1e-6


def test_rejects_unsupported_optimizer():
    train_features, train_labels = _make_synthetic_linearly_separable_data(seed=0)
    val_features, val_labels = _make_synthetic_linearly_separable_data(seed=1)
    hyperparams = LinearProbeHyperparams(optimizer="sgd", max_epochs=1)

    import pytest
    with pytest.raises(ValueError, match="adamw"):
        train_linear_probe(
            train_features, train_labels, val_features, val_labels, 3, hyperparams,
            seed=0, device=torch.device("cpu"),
        )
