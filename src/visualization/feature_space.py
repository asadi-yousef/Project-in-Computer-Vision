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
from typing import List, Sequence, Tuple, Union

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


def project_feature_groups(
    feature_groups: Sequence[torch.Tensor], seed: int = 0
) -> List[np.ndarray]:
    """Jointly fit one 2D t-SNE over several feature sets and split the result.

    stage_2.pdf requires the projection be computed jointly over the feature
    sets and prototypes being compared, so that the different views
    correspond to the same low-dimensional representation. Fitting each panel
    separately would produce three unrelated coordinate systems, and a sample
    "moving" between panels would mean nothing.

    Every group is L2-normalized first, for the same reason
    `project_features_and_prototypes` does it: prototypes are unit-norm by
    construction while raw encoder features are not, and post-FM features
    have drifted off the sphere entirely, so without normalizing, t-SNE's
    distances would be dominated by magnitude rather than direction - the
    opposite of what the cosine classifier actually uses.

    Args:
        feature_groups: the (N_i, D) tensors to project together. All must
            share the same feature dimension D.
        seed: t-SNE is stochastic; this makes the layout reproducible.

    Returns:
        One (N_i, 2) array per input group, in the same order.

    Raises:
        ValueError: if no groups are given, or their dimensions disagree.
    """
    if not feature_groups:
        raise ValueError("feature_groups is empty; nothing to project")
    dimensions = {group.shape[1] for group in feature_groups}
    if len(dimensions) != 1:
        raise ValueError(f"All feature groups must share one dimension, got {sorted(dimensions)}")

    normalized = [torch.nn.functional.normalize(group, dim=1) for group in feature_groups]
    combined = torch.cat(normalized, dim=0).numpy()

    perplexity = min(30, max(2, combined.shape[0] // 4))
    projected = TSNE(
        n_components=2, random_state=seed, perplexity=perplexity, init="pca"
    ).fit_transform(combined)

    outputs = []
    offset = 0
    for group in feature_groups:
        outputs.append(projected[offset : offset + group.shape[0]])
        offset += group.shape[0]
    return outputs


def plot_feature_space_comparison(
    panels: Sequence[Tuple[str, np.ndarray]],
    prototype_2d: np.ndarray,
    sample_class_ids: List[int],
    prototype_class_ids: List[int],
    class_names: List[str],
    suptitle: str,
    save_path: Union[str, Path],
) -> None:
    """Plot several views of the same test samples side by side.

    All panels share one colour map, one set of axis limits, and the same
    prototype positions, so differences between panels are differences in
    where the samples went - not in how the panels were drawn. The prototypes
    are identical across panels by construction: they are the flow's fixed
    targets and are never transported.

    Args:
        panels: (title, sample_2d) per panel, e.g. original / after standard
            FM / after rolled-out FM. All coordinates must come from one
            joint projection (see `project_feature_groups`).
        prototype_2d: (C, 2) projected prototype coordinates, shared by every panel.
        sample_class_ids: (N,) true class id per test sample, shared by every panel.
        prototype_class_ids: (C,) class id per prototype, matching prototype_2d order.
        class_names: full dataset class-name list, indexed by class id.
        suptitle: figure-level title.
        save_path: where to save the PNG.

    Raises:
        ValueError: if no panels are given.
    """
    if not panels:
        raise ValueError("panels is empty; nothing to plot")

    unique_class_ids = sorted(set(sample_class_ids))
    color_map = {class_id: plt.cm.tab10(i % 10) for i, class_id in enumerate(unique_class_ids)}
    sample_class_ids_array = np.array(sample_class_ids)

    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.2 * len(panels), 5.0), dpi=150, squeeze=False
    )
    axes = axes[0]

    all_x = np.concatenate([sample_2d[:, 0] for _, sample_2d in panels] + [prototype_2d[:, 0]])
    all_y = np.concatenate([sample_2d[:, 1] for _, sample_2d in panels] + [prototype_2d[:, 1]])
    margin_x = 0.05 * (all_x.max() - all_x.min())
    margin_y = 0.05 * (all_y.max() - all_y.min())

    for axis, (title, sample_2d) in zip(axes, panels):
        for class_id in unique_class_ids:
            mask = sample_class_ids_array == class_id
            axis.scatter(
                sample_2d[mask, 0], sample_2d[mask, 1],
                color=color_map[class_id], marker="o", s=22, alpha=0.7,
                label=class_names[class_id],
            )
        for prototype_point, class_id in zip(prototype_2d, prototype_class_ids):
            axis.scatter(
                prototype_point[0], prototype_point[1],
                color=color_map[class_id], marker="*", s=340,
                edgecolors="black", linewidths=1.2, zorder=5,
            )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("t-SNE dimension 1")
        axis.set_xlim(all_x.min() - margin_x, all_x.max() + margin_x)
        axis.set_ylim(all_y.min() - margin_y, all_y.max() + margin_y)

    axes[0].set_ylabel("t-SNE dimension 2")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8,
        title="Class (o = image, * = prototype)",
    )
    fig.suptitle(suptitle, fontsize=11)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
