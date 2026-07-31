"""Recompute test-set predictions for an already-completed run.

Confusion matrices need per-image predictions, but the experiment runners
only ever saved aggregate test_accuracy. Rather than re-training, this
module recomputes predictions post-hoc: a single forward pass through the
saved linear-probe checkpoint, or a deterministic prototype recomputation
from cached features - either way, no gradient descent occurs here.
"""

from pathlib import Path
from typing import Tuple, Union

import torch

from src.classifiers.linear_probe import predict_linear_probe
from src.classifiers.prototype import compute_class_prototypes, predict_by_cosine_similarity
from src.data.few_shot import sample_balanced_subset_indices
from src.features.loading import load_validated_feature_cache


def get_linear_probe_test_predictions(
    cache_dir: Union[str, Path],
    output_dir: Union[str, Path],
    dataset: str,
    encoder: str,
    k_shot,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a completed linear-probe run's checkpoint and recompute its test predictions.

    Returns:
        (true_labels, predicted_labels) for the full test split.

    Raises:
        FileNotFoundError: if the run's checkpoint does not exist (it has
            not been trained yet).
    """
    test_features, test_labels, test_metadata = load_validated_feature_cache(
        cache_dir, dataset, encoder, "test"
    )
    checkpoint_path = (
        Path(output_dir) / "linear_probe" / dataset / encoder / f"k{k_shot}" / f"seed{seed}" / "checkpoint.pt"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}; run scripts/run_linear_probe.py first"
        )
    state_dict = torch.load(checkpoint_path, weights_only=False)

    predicted_labels = predict_linear_probe(
        state_dict,
        feature_dim=test_features.shape[1],
        num_classes=test_metadata["num_classes"],
        features=test_features,
        device=device,
    )
    return test_labels, predicted_labels


def get_prototype_test_predictions(
    cache_dir: Union[str, Path],
    dataset: str,
    encoder: str,
    k_shot,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Recompute a prototype run's test predictions directly from cached
    features: deterministic given (dataset, encoder, k_shot, seed), so no
    checkpoint is needed.

    Returns:
        (true_labels, predicted_labels) for the full test split.
    """
    train_features, train_labels, train_metadata = load_validated_feature_cache(
        cache_dir, dataset, encoder, "train"
    )
    test_features, test_labels, _ = load_validated_feature_cache(cache_dir, dataset, encoder, "test")
    num_classes = train_metadata["num_classes"]

    if k_shot != "full":
        indices = sample_balanced_subset_indices(train_labels.tolist(), k_shot, seed)
        train_features = train_features[indices]
        train_labels = train_labels[indices]

    prototypes = compute_class_prototypes(train_features, train_labels, num_classes)
    predicted_labels = predict_by_cosine_similarity(test_features, prototypes)
    return test_labels, predicted_labels
