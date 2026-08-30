"""Visualize the Euler integration path itself: where a test feature starts,
the intermediate states it passes through, where it ends up, and where its
class prototype sits (stage_2.pdf, deliverable 4).

PCA rather than t-SNE here, as the spec recommends. The reason is specific
to trajectories: PCA is a linear projection, so a straight path in feature
space projects to a straight path on the page, and relative distances along
a path stay meaningful. t-SNE is neither linear nor distance-preserving, so
a perfectly smooth flow would project to a kinked one and the geometry would
be an artifact of the embedding. (The feature-space comparison in
feature_space.py uses t-SNE for the opposite reason: there, cluster
structure matters more than the geometry of any individual point's motion.)

Unlike the t-SNE views, trajectories are projected **without** L2-normalizing
first. The Euler path is a path in the raw feature space, and intermediate
states genuinely drift off the unit sphere; normalizing each state would
project that motion away and draw a path the model never took.
"""

import random
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")  # renders to file without needing a display
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

DEFAULT_EXAMPLES_PER_CLASS = 2


def select_trajectory_examples(
    test_labels: torch.Tensor,
    class_ids: Sequence[int],
    examples_per_class: int = DEFAULT_EXAMPLES_PER_CLASS,
    seed: int = 0,
) -> List[int]:
    """Pick a small, reproducible set of test examples to trace.

    stage_2.pdf asks for "a small number of representative test examples" -
    a trajectory plot with hundreds of paths is unreadable. Callers pass the
    class ids already chosen for the feature-space figure, so the two figures
    show the same classes in the same colours.

    Args:
        test_labels: (N,) integer labels of the full test split.
        class_ids: which classes to draw examples from.
        examples_per_class: how many test examples per class to trace.
        seed: makes the choice reproducible.

    Returns:
        A sorted list of indices into the test split.

    Raises:
        ValueError: if a requested class has too few test samples.
    """
    indices_by_class: dict = {}
    for index, label in enumerate(test_labels.tolist()):
        indices_by_class.setdefault(label, []).append(index)

    rng = random.Random(seed)
    selected: List[int] = []
    for class_id in class_ids:
        available = indices_by_class.get(class_id, [])
        if len(available) < examples_per_class:
            raise ValueError(
                f"Class {class_id} has only {len(available)} test samples, "
                f"fewer than the requested examples_per_class={examples_per_class}"
            )
        selected.extend(rng.sample(available, examples_per_class))
    return sorted(selected)


def project_trajectories(
    trajectory_groups: Sequence[torch.Tensor],
    prototypes: torch.Tensor,
    seed: int = 0,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Fit one PCA over every trajectory state and prototype, then split it.

    Fitting jointly is what makes the panels comparable: the standard and
    rolled-out paths are drawn in one coordinate system, so a longer arrow in
    one panel really is a longer step.

    Args:
        trajectory_groups: one (T+1, N, D) tensor per variant being compared.
            Different groups may have different T.
        prototypes: (C, D) class prototypes, the flow's fixed targets.
        seed: PCA's solver is deterministic for full SVD, but the seed is
            passed through for reproducibility if a randomized solver is
            selected for large inputs.

    Returns:
        (projected_groups, prototype_2d, explained_variance_ratio) where each
        projected group is (T+1, N, 2), prototype_2d is (C, 2), and the ratio
        is the fraction of variance each of the 2 components captures.

    Raises:
        ValueError: if no groups are given, or dimensions disagree.
    """
    if not trajectory_groups:
        raise ValueError("trajectory_groups is empty; nothing to project")
    dimensions = {group.shape[-1] for group in trajectory_groups} | {prototypes.shape[-1]}
    if len(dimensions) != 1:
        raise ValueError(
            f"All trajectories and prototypes must share one dimension, got {sorted(dimensions)}"
        )

    flattened = [group.reshape(-1, group.shape[-1]) for group in trajectory_groups]
    combined = torch.cat(flattened + [prototypes], dim=0).numpy()

    pca = PCA(n_components=2, random_state=seed)
    projected = pca.fit_transform(combined)

    outputs = []
    offset = 0
    for group in trajectory_groups:
        count = group.shape[0] * group.shape[1]
        outputs.append(projected[offset : offset + count].reshape(group.shape[0], group.shape[1], 2))
        offset += count

    return outputs, projected[offset:], pca.explained_variance_ratio_


def plot_flow_trajectories(
    panels: Sequence[Tuple[str, np.ndarray]],
    prototype_2d: np.ndarray,
    sample_class_ids: Sequence[int],
    prototype_class_ids: Sequence[int],
    class_names: Sequence[str],
    suptitle: str,
    save_path: Union[str, Path],
    explained_variance_ratio: Sequence[float] = None,
    legend_title: str = None,
    share_limits: bool = True,
    background: Sequence[Tuple[np.ndarray, Sequence[int]]] = None,
) -> None:
    """Draw each traced example's path from its original feature to its endpoint.

    Every path is drawn with its start (circle), each intermediate Euler
    state (small dot), and its endpoint (cross), coloured by true class, with
    the class prototypes as stars. Panels share colours, axis limits, and
    prototype positions, so only the paths differ.

    Args:
        panels: (title, trajectory_2d) per variant, each (T+1, N, 2) from
            `project_trajectories`.
        prototype_2d: (C, 2) projected prototypes shared by every panel, or
            one (C, 2) array per panel. A single array is right when the
            panels come from one joint projection; per-panel arrays are
            required when each panel was projected in its own basis, since
            the prototypes then have different coordinates in each.
        sample_class_ids: (N,) true class id of each traced example.
        prototype_class_ids: (C,) class id per prototype, matching prototype_2d.
        class_names: full dataset class-name list, indexed by class id.
        suptitle: figure-level title.
        save_path: where to save the PNG.
        explained_variance_ratio: optional 2 values, shown on the axis labels
            so the reader knows how much structure the projection retains.
        legend_title: overrides the marker key. The default describes a
            forward flow; a reverse flow starts at the prototype and ends at
            a generated point, so it needs different wording.
        share_limits: draw all panels on one set of axis limits (the default,
            and what makes forward panels directly comparable). Set False
            when one panel's paths are orders of magnitude longer than
            another's - as happens with reverse flow, where a diverging field
            would otherwise squash the other panel to a single dot. When it
            is False the panels are no longer comparable by eye, so the
            magnitude difference must be reported some other way.
        background: optional (points_2d, class_ids) per panel, drawn as faint
            dots behind everything else. Used to place real test samples
            beside the paths, so a reverse flow can be judged against the
            actual class it claims to reconstruct rather than against nothing.
            Must be projected in the same basis as that panel's paths.

    Raises:
        ValueError: if no panels are given, or `background` is given but does
            not have one entry per panel.
    """
    if not panels:
        raise ValueError("panels is empty; nothing to plot")
    if background is not None and len(background) != len(panels):
        raise ValueError(
            f"Got {len(background)} background sets for {len(panels)} panels"
        )

    unique_class_ids = sorted(set(sample_class_ids))
    color_map = {class_id: plt.cm.tab10(i % 10) for i, class_id in enumerate(unique_class_ids)}

    fig, axes = plt.subplots(
        1, len(panels), figsize=(6.0 * len(panels), 5.4), dpi=150, squeeze=False
    )
    axes = axes[0]

    if isinstance(prototype_2d, np.ndarray):
        prototypes_per_panel = [prototype_2d] * len(panels)
    else:
        prototypes_per_panel = list(prototype_2d)
        if len(prototypes_per_panel) != len(panels):
            raise ValueError(
                f"Got {len(prototypes_per_panel)} prototype arrays for {len(panels)} panels"
            )

    backgrounds = list(background) if background is not None else [None] * len(panels)

    all_points = np.concatenate(
        [trajectory.reshape(-1, 2) for _, trajectory in panels]
        + prototypes_per_panel
        + [entry[0] for entry in backgrounds if entry is not None]
    )
    margin = 0.06 * (all_points.max(axis=0) - all_points.min(axis=0))
    lower = all_points.min(axis=0) - margin
    upper = all_points.max(axis=0) + margin

    if explained_variance_ratio is not None:
        x_label = f"PC 1 ({explained_variance_ratio[0] * 100:.1f}% of variance)"
        y_label = f"PC 2 ({explained_variance_ratio[1] * 100:.1f}% of variance)"
    else:
        x_label, y_label = "PC 1", "PC 2"

    labelled_classes = set()
    for axis, (title, trajectory_2d), panel_prototypes, panel_background in zip(
        axes, panels, prototypes_per_panel, backgrounds
    ):
        # Real samples go down first, faint, so the paths read on top of them.
        if panel_background is not None:
            points, point_class_ids = panel_background
            point_class_ids = np.asarray(point_class_ids)
            for class_id in sorted(set(point_class_ids.tolist())):
                mask = point_class_ids == class_id
                axis.scatter(
                    points[mask, 0], points[mask, 1],
                    color=color_map.get(class_id, "tab:gray"),
                    marker="o", s=46, alpha=0.5, linewidths=0, zorder=1,
                )

        for sample_index, class_id in enumerate(sample_class_ids):
            path = trajectory_2d[:, sample_index, :]
            color = color_map[class_id]
            label = None
            if class_id not in labelled_classes:
                label = class_names[class_id]
                labelled_classes.add(class_id)
            axis.plot(path[:, 0], path[:, 1], color=color, alpha=0.55, linewidth=1.2, label=label)
            axis.scatter(path[1:-1, 0], path[1:-1, 1], color=color, s=12, alpha=0.8, zorder=3)
            axis.scatter(
                path[0, 0], path[0, 1], color=color, marker="o", s=55,
                edgecolors="black", linewidths=0.8, zorder=4,
            )
            axis.scatter(
                path[-1, 0], path[-1, 1], color=color, marker="X", s=75,
                edgecolors="black", linewidths=0.8, zorder=4,
            )

        for prototype_point, class_id in zip(panel_prototypes, prototype_class_ids):
            axis.scatter(
                prototype_point[0], prototype_point[1], color=color_map[class_id],
                marker="*", s=340, edgecolors="black", linewidths=1.2, zorder=5,
            )

        axis.set_title(title, fontsize=10)
        axis.set_xlabel(x_label)
        if share_limits:
            axis.set_xlim(lower[0], upper[0])
            axis.set_ylim(lower[1], upper[1])
        axis.grid(alpha=0.25)

    axes[0].set_ylabel(y_label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8,
        title=legend_title
        or "Class\n(o = original, . = Euler step,\nX = transported, * = prototype)",
    )
    fig.suptitle(suptitle, fontsize=11)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Leave headroom so a multi-line suptitle cannot overlap the panel titles.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
