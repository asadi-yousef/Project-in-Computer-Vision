"""Plot a row-normalized confusion matrix (rows = true label, columns =
predicted label). For datasets with many classes, per-class tick labels are
thinned out for readability, per stage_1.pdf's "handle large class counts
in a readable way" requirement.
"""

from pathlib import Path
from typing import List, Union

import matplotlib

matplotlib.use("Agg")  # renders to file without needing a display
import matplotlib.pyplot as plt
import numpy as np

# Above this many classes, labeling every tick becomes unreadable; only
# every Nth class is labeled instead.
_MAX_CLASSES_FOR_FULL_LABELS = 50


def plot_confusion_matrix(
    normalized_matrix: np.ndarray,
    class_names: List[str],
    title: str,
    save_path: Union[str, Path],
) -> None:
    """Plot a row-normalized confusion matrix as a heatmap.

    Args:
        normalized_matrix: (num_classes, num_classes) array, rows summing to 1.
        class_names: class name for each row/column index.
        title: plot title (should identify dataset/encoder/method/setting).
        save_path: where to save the PNG.
    """
    num_classes = normalized_matrix.shape[0]
    if len(class_names) != num_classes:
        raise ValueError(
            f"class_names has {len(class_names)} entries but matrix has {num_classes} classes"
        )

    figure_size = max(6.0, num_classes * 0.18)
    fig, ax = plt.subplots(figsize=(figure_size, figure_size), dpi=150)
    image = ax.imshow(normalized_matrix, vmin=0, vmax=1, cmap="viridis")
    fig.colorbar(image, ax=ax, label="Fraction of true-class test samples", fraction=0.046, pad=0.04)

    if num_classes <= _MAX_CLASSES_FOR_FULL_LABELS:
        tick_positions = list(range(num_classes))
    else:
        tick_step = max(1, num_classes // 20)
        tick_positions = list(range(0, num_classes, tick_step))

    tick_labels = [class_names[i] for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=6)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
