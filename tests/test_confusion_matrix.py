import numpy as np
import torch

from src.evaluation.confusion_matrix import compute_confusion_matrix, row_normalize


def test_compute_confusion_matrix_counts_correctly():
    true_labels = torch.tensor([0, 0, 1, 1, 1])
    predicted_labels = torch.tensor([0, 1, 1, 1, 0])

    matrix = compute_confusion_matrix(true_labels, predicted_labels, num_classes=2)

    expected = np.array([[1, 1], [1, 2]])
    assert np.array_equal(matrix, expected)


def test_row_normalize_rows_sum_to_one():
    matrix = np.array([[1, 1], [1, 2]], dtype=np.int64)

    normalized = row_normalize(matrix)

    assert np.allclose(normalized.sum(axis=1), [1.0, 1.0])
    assert np.allclose(normalized, [[0.5, 0.5], [1 / 3, 2 / 3]])


def test_row_normalize_handles_empty_class_without_nan():
    matrix = np.array([[0, 0], [1, 1]], dtype=np.int64)

    normalized = row_normalize(matrix)

    assert not np.isnan(normalized).any()
    assert np.array_equal(normalized[0], [0.0, 0.0])
    assert np.allclose(normalized[1], [0.5, 0.5])
