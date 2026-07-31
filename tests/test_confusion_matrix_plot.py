import numpy as np
import pytest

from src.visualization.confusion_matrix_plot import plot_confusion_matrix


def test_plot_confusion_matrix_is_saved_to_disk(tmp_path):
    matrix = np.array([[0.8, 0.2], [0.1, 0.9]])
    save_path = tmp_path / "confusion.png"

    plot_confusion_matrix(matrix, ["cat", "dog"], "Example confusion matrix", save_path)

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_confusion_matrix_handles_many_classes(tmp_path):
    num_classes = 60
    matrix = np.eye(num_classes)
    class_names = [f"class_{i}" for i in range(num_classes)]
    save_path = tmp_path / "confusion_large.png"

    plot_confusion_matrix(matrix, class_names, "Many-class example", save_path)

    assert save_path.exists()


def test_plot_confusion_matrix_raises_for_mismatched_class_names(tmp_path):
    matrix = np.eye(3)
    with pytest.raises(ValueError, match="class_names"):
        plot_confusion_matrix(matrix, ["only", "two"], "Bad input", tmp_path / "unused.png")
