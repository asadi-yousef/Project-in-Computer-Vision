"""Image-derived class prototypes (stage_1.pdf, Option A).

No gradient descent occurs here - the exact prescribed sequence is:
  1. L2-normalize every selected training feature.
  2. Average the normalized features within each class.
  3. L2-normalize the resulting class mean -> the prototype.
  4. L2-normalize evaluation features.
  5. Classify by cosine similarity to the nearest prototype.
"""

import torch
import torch.nn.functional as F


def compute_class_prototypes(
    features: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Compute one L2-normalized prototype per class.

    Args:
        features: (N, D) training features, already restricted to the
            selected K-shot subset if applicable.
        labels: (N,) integer class labels in [0, num_classes).
        num_classes: total number of classes in the dataset.

    Returns:
        (num_classes, D) tensor of L2-normalized class prototypes.

    Raises:
        ValueError: if any class in [0, num_classes) has zero features
            (a prototype cannot be computed for an empty class).
    """
    normalized_features = F.normalize(features, dim=1)
    prototypes = torch.zeros(num_classes, features.shape[1], dtype=features.dtype)

    for class_index in range(num_classes):
        class_mask = labels == class_index
        if int(class_mask.sum()) == 0:
            raise ValueError(f"Class {class_index} has no training features to build a prototype from")
        class_mean = normalized_features[class_mask].mean(dim=0)
        prototypes[class_index] = F.normalize(class_mean, dim=0)

    return prototypes


def predict_by_cosine_similarity(
    features: torch.Tensor, prototypes: torch.Tensor
) -> torch.Tensor:
    """Classify features by highest cosine similarity to a class prototype.

    Args:
        features: (N, D) evaluation features (val or test).
        prototypes: (num_classes, D) L2-normalized class prototypes.

    Returns:
        (N,) predicted class indices.
    """
    normalized_features = F.normalize(features, dim=1)
    similarities = normalized_features @ prototypes.T  # (N, num_classes)
    return similarities.argmax(dim=1)
