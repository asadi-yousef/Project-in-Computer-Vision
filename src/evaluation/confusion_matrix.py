"""Row-normalized confusion matrices.

stage_1.pdf convention: rows are true labels, columns are predicted labels.
Each row is normalized to sum to 1 (fraction of that true class's test
samples predicted into each class).
"""

import numpy as np
import torch


def compute_confusion_matrix(
    true_labels: torch.Tensor, predicted_labels: torch.Tensor, num_classes: int
) -> np.ndarray:
    """Raw (unnormalized) confusion matrix: counts[true_label, predicted_label]."""
    counts = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, predicted_label in zip(true_labels.tolist(), predicted_labels.tolist()):
        counts[true_label, predicted_label] += 1
    return counts


def row_normalize(confusion_matrix: np.ndarray) -> np.ndarray:
    """Normalize each row to sum to 1.

    Rows with zero samples (a class absent from the evaluated data) stay
    all-zero rather than becoming NaN.
    """
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion_matrix,
        row_sums,
        out=np.zeros_like(confusion_matrix, dtype=np.float64),
        where=row_sums != 0,
    )
    return normalized
