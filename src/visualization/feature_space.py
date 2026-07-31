"""Reproducible feature-space visualizations: select a fixed set of classes
and test samples, jointly project (test-image features + class prototypes)
into 2D via t-SNE, and plot them with consistent per-class colors.

stage_1.pdf requirements this satisfies:
  - ~8-10 classes, kept identical when comparing encoders/methods on the
    same dataset (the selection is saved to disk and reloaded, not
    re-randomized per encoder).
  - the 2D projection is fit jointly over images and prototypes together.
  - images and prototypes are visually distinguished (marker shape/size).
  - the selected class IDs and sample indices are stored, so the plot is
    reproducible.
"""

import dataclasses
import json
import random
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib

matplotlib.use("Agg")  # renders to file without needing a display
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

DEFAULT_NUM_CLASSES = 10
DEFAULT_SAMPLES_PER_CLASS = 15


@dataclasses.dataclass
class FeatureVisualizationSelection:
    """A fixed set of class IDs and test-sample indices for one dataset,
    reused across every encoder/method plotted for that dataset."""

    dataset: str
    class_ids: List[int]
    sample_indices: List[int]  # indices into the dataset's cached test features/labels


def select_classes_and_samples(
    dataset: str,
    test_labels: torch.Tensor,
    num_classes: int = DEFAULT_NUM_CLASSES,
    samples_per_class: int = DEFAULT_SAMPLES_PER_CLASS,
    seed: int = 0,
) -> FeatureVisualizationSelection:
    """Randomly (but reproducibly) select classes and test samples to visualize.

    Args:
        dataset: dataset name, stored for reference only.
        test_labels: (N,) integer labels of the full test split.
        num_classes: how many classes to select (~8-10 per stage_1.pdf).
        samples_per_class: how many test samples per selected class to plot.
        seed: controls both which classes and which samples are chosen.

    Raises:
        ValueError: if num_classes exceeds the classes available, or a
            selected class has fewer than samples_per_class test samples.
    """
    rng = random.Random(seed)
    all_classes = sorted(set(test_labels.tolist()))
    if num_classes > len(all_classes):
        raise ValueError(f"Requested {num_classes} classes but dataset only has {len(all_classes)}")
    selected_class_ids = sorted(rng.sample(all_classes, num_classes))

    indices_by_class = {}
    for index, label in enumerate(test_labels.tolist()):
        indices_by_class.setdefault(label, []).append(index)

    selected_sample_indices = []
    for class_id in selected_class_ids:
        available_indices = indices_by_class[class_id]
        if len(available_indices) < samples_per_class:
            raise ValueError(
                f"Class {class_id} has only {len(available_indices)} test samples, "
                f"fewer than requested samples_per_class={samples_per_class}"
            )
        selected_sample_indices.extend(rng.sample(available_indices, samples_per_class))

    return FeatureVisualizationSelection(
        dataset=dataset,
        class_ids=selected_class_ids,
        sample_indices=sorted(selected_sample_indices),
    )


def save_selection(selection: FeatureVisualizationSelection, path: Union[str, Path]) -> None:
    """Persist a selection to JSON so later runs reuse the exact same classes/samples."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dataclasses.asdict(selection), f, indent=2)


def load_selection(path: Union[str, Path]) -> FeatureVisualizationSelection:
    """Load a previously saved selection."""
    with open(path) as f:
        data = json.load(f)
    return FeatureVisualizationSelection(**data)


def project_features_and_prototypes(
    sample_features: torch.Tensor, prototype_features: torch.Tensor, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Jointly fit a 2D t-SNE projection over samples and prototypes together.

    Fitting jointly (one fit_transform call over the concatenation), rather
    than fitting on samples and separately projecting prototypes, ensures
    both live in the same 2D coordinate system, per stage_1.pdf.

    Both inputs are L2-normalized first. Prototypes are already unit-norm by
    construction, but raw encoder features are not (ResNet-18 features on
    DTD, for example, have norms ranging roughly 11-54) - without
    normalizing the samples too, t-SNE's distance computation is dominated
    by that scale mismatch and every prototype collapses into one corner
    regardless of direction. Normalizing both projects them into the same
    space the prototype classifier itself operates in (cosine similarity).

    Args:
        sample_features: (N, D) test-image features to plot.
        prototype_features: (C, D) class prototype features to plot.
        seed: t-SNE is stochastic; this makes the layout reproducible.

    Returns:
        (sample_2d, prototype_2d): (N, 2) and (C, 2) arrays.
    """
    normalized_samples = torch.nn.functional.normalize(sample_features, dim=1)
    normalized_prototypes = torch.nn.functional.normalize(prototype_features, dim=1)
    combined = torch.cat([normalized_samples, normalized_prototypes], dim=0).numpy()
    num_samples = sample_features.shape[0]

    # Perplexity must be less than the number of points being embedded.
    perplexity = min(30, max(2, combined.shape[0] // 4))
    projected = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca").fit_transform(
        combined
    )

    return projected[:num_samples], projected[num_samples:]


def plot_feature_space(
    sample_2d: np.ndarray,
    sample_class_ids: List[int],
    prototype_2d: np.ndarray,
    prototype_class_ids: List[int],
    class_names: List[str],
    title: str,
    save_path: Union[str, Path],
) -> None:
    """Scatter-plot projected test-image features (circles) and class
    prototypes (larger star markers), colored consistently by class.

    Args:
        sample_2d: (N, 2) projected test-image coordinates.
        sample_class_ids: (N,) true class id for each test image.
        prototype_2d: (C, 2) projected prototype coordinates.
        prototype_class_ids: (C,) class id for each prototype, same order as prototype_2d.
        class_names: full dataset class-name list, indexed by class id.
        title: plot title.
        save_path: where to save the PNG.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=150)

    unique_class_ids = sorted(set(sample_class_ids))
    color_map = {class_id: plt.cm.tab10(i % 10) for i, class_id in enumerate(unique_class_ids)}

    sample_class_ids_array = np.array(sample_class_ids)
    for class_id in unique_class_ids:
        mask = sample_class_ids_array == class_id
        ax.scatter(
            sample_2d[mask, 0],
            sample_2d[mask, 1],
            color=color_map[class_id],
            marker="o",
            s=25,
            alpha=0.7,
            label=class_names[class_id],
        )

    for prototype_point, class_id in zip(prototype_2d, prototype_class_ids):
        ax.scatter(
            prototype_point[0],
            prototype_point[1],
            color=color_map[class_id],
            marker="*",
            s=400,
            edgecolors="black",
            linewidths=1.2,
            zorder=5,
        )

    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.set_title(title)
    ax.legend(
        loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8,
        title="Class (o = image, * = prototype)",
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
