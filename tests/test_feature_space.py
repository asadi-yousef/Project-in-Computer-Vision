from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from src.visualization.feature_space import (
    FeatureVisualizationSelection,
    load_selection,
    plot_feature_space,
    plot_feature_space_comparison,
    project_feature_groups,
    project_features_and_prototypes,
    save_selection,
    select_classes_and_samples,
)


def _synthetic_labels(num_classes=10, samples_per_class=20):
    return torch.tensor([c for c in range(num_classes) for _ in range(samples_per_class)])


def test_select_classes_and_samples_picks_requested_counts():
    labels = _synthetic_labels(num_classes=10, samples_per_class=20)

    selection = select_classes_and_samples("dtd", labels, num_classes=5, samples_per_class=4, seed=0)

    assert len(selection.class_ids) == 5
    assert len(selection.sample_indices) == 20  # 5 classes * 4 samples
    assert len(set(selection.sample_indices)) == 20  # no duplicate indices

    selected_labels = labels[selection.sample_indices].tolist()
    assert set(selected_labels) == set(selection.class_ids)


def test_select_classes_and_samples_is_deterministic_given_same_seed():
    labels = _synthetic_labels()
    first = select_classes_and_samples("dtd", labels, num_classes=5, samples_per_class=4, seed=0)
    second = select_classes_and_samples("dtd", labels, num_classes=5, samples_per_class=4, seed=0)
    assert first == second


def test_select_classes_and_samples_raises_when_too_many_classes_requested():
    labels = _synthetic_labels(num_classes=3, samples_per_class=20)
    with pytest.raises(ValueError, match="only has 3"):
        select_classes_and_samples("dtd", labels, num_classes=5, samples_per_class=4, seed=0)


def test_select_classes_and_samples_raises_when_class_too_small():
    labels = _synthetic_labels(num_classes=5, samples_per_class=2)
    with pytest.raises(ValueError, match="fewer than requested"):
        select_classes_and_samples("dtd", labels, num_classes=5, samples_per_class=4, seed=0)


def test_save_and_load_selection_round_trips(tmp_path):
    selection = FeatureVisualizationSelection(dataset="dtd", class_ids=[1, 3, 5], sample_indices=[10, 11, 12])
    path = tmp_path / "selection.json"

    save_selection(selection, path)
    loaded = load_selection(path)

    assert loaded == selection


def test_project_features_and_prototypes_returns_correct_shapes():
    torch.manual_seed(0)
    sample_features = torch.randn(30, 8)
    prototype_features = torch.randn(3, 8)

    sample_2d, prototype_2d = project_features_and_prototypes(sample_features, prototype_features, seed=0)

    assert sample_2d.shape == (30, 2)
    assert prototype_2d.shape == (3, 2)


def test_project_features_and_prototypes_normalizes_before_projecting():
    # Two orthogonal prototypes (unit norm by construction); "image"
    # features pointing in the same directions but with a huge norm -
    # reproducing the real scale mismatch (raw ResNet-18 features can have
    # norms ~50x a prototype's). Without normalizing samples first, t-SNE's
    # distance computation is dominated by norm rather than direction, and
    # both prototypes would collapse together regardless of class.
    torch.manual_seed(0)
    dim = 8
    prototype_features = torch.zeros(2, dim)
    prototype_features[0, 0] = 1.0
    prototype_features[1, 1] = 1.0

    noise = torch.randn(8, dim) * 0.01
    class_0_samples = prototype_features[0].unsqueeze(0).repeat(4, 1) * 50.0 + noise[:4]
    class_1_samples = prototype_features[1].unsqueeze(0).repeat(4, 1) * 50.0 + noise[4:]
    sample_features = torch.cat([class_0_samples, class_1_samples], dim=0)

    sample_2d, prototype_2d = project_features_and_prototypes(
        sample_features, prototype_features, seed=0
    )

    class_0_center = sample_2d[:4].mean(axis=0)
    class_1_center = sample_2d[4:].mean(axis=0)

    dist_0_to_proto0 = ((class_0_center - prototype_2d[0]) ** 2).sum()
    dist_0_to_proto1 = ((class_0_center - prototype_2d[1]) ** 2).sum()
    dist_1_to_proto0 = ((class_1_center - prototype_2d[0]) ** 2).sum()
    dist_1_to_proto1 = ((class_1_center - prototype_2d[1]) ** 2).sum()

    assert dist_0_to_proto0 < dist_0_to_proto1
    assert dist_1_to_proto1 < dist_1_to_proto0


def test_plot_feature_space_is_saved_to_disk(tmp_path):
    sample_2d = torch.randn(12, 2).numpy()
    sample_class_ids = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    prototype_2d = torch.randn(3, 2).numpy()
    prototype_class_ids = [0, 1, 2]
    class_names = ["a", "b", "c"]

    save_path = tmp_path / "feature_space.png"
    plot_feature_space(
        sample_2d, sample_class_ids, prototype_2d, prototype_class_ids, class_names,
        "test plot", save_path,
    )

    assert save_path.exists()
    assert save_path.stat().st_size > 0


# --- Stage 2: joint projection over several feature sets ---


def test_joint_projection_returns_one_array_per_group():
    groups = [torch.randn(12, 6), torch.randn(12, 6), torch.randn(5, 6)]

    projected = project_feature_groups(groups, seed=0)

    assert [array.shape for array in projected] == [(12, 2), (12, 2), (5, 2)]


def test_joint_projection_fits_once_over_the_concatenation():
    # The property stage_2.pdf requires: one projection covering every group,
    # so the panels share a coordinate system. Asserted on the call itself
    # rather than on output coordinates, because t-SNE is not a function -
    # duplicate points repel each other rather than coinciding, so equal
    # inputs are not expected to produce equal outputs.
    groups = [torch.randn(12, 6), torch.randn(9, 6), torch.randn(4, 6)]
    captured = {}

    class FakeTSNE:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def fit_transform(self, array):
            captured["calls"] = captured.get("calls", 0) + 1
            captured["shape"] = array.shape
            return np.zeros((array.shape[0], 2))

    with mock.patch("src.visualization.feature_space.TSNE", FakeTSNE):
        project_feature_groups(groups, seed=0)

    assert captured["calls"] == 1
    assert captured["shape"] == (25, 6)  # 12 + 9 + 4 rows, projected together


def test_joint_projection_is_reproducible_given_a_seed():
    groups = [torch.randn(15, 6), torch.randn(15, 6)]

    first = project_feature_groups(groups, seed=3)
    second = project_feature_groups(groups, seed=3)

    assert all(np.allclose(a, b) for a, b in zip(first, second))


def test_joint_projection_normalizes_before_fitting():
    # Post-FM features drift off the unit sphere and raw encoder features
    # never were on it, so without normalizing, t-SNE distances would be
    # dominated by magnitude instead of direction.
    groups = [torch.randn(10, 6) * 37.0, torch.randn(4, 6) * 0.01]
    captured = {}

    class FakeTSNE:
        def __init__(self, **kwargs):
            pass

        def fit_transform(self, array):
            captured["array"] = array
            return np.zeros((array.shape[0], 2))

    with mock.patch("src.visualization.feature_space.TSNE", FakeTSNE):
        project_feature_groups(groups, seed=0)

    norms = np.linalg.norm(captured["array"], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_joint_projection_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        project_feature_groups([], seed=0)


def test_joint_projection_rejects_mismatched_dimensions():
    with pytest.raises(ValueError, match="dimension"):
        project_feature_groups([torch.randn(10, 6), torch.randn(10, 8)], seed=0)


# --- Stage 2: the three-panel comparison figure ---


def _comparison_inputs(num_classes=3, per_class=5):
    sample_class_ids = [c for c in range(num_classes) for _ in range(per_class)]
    num_samples = len(sample_class_ids)
    rng = np.random.default_rng(0)
    panels = [
        ("original", rng.normal(size=(num_samples, 2))),
        ("after standard FM", rng.normal(size=(num_samples, 2))),
        ("after rolled-out FM", rng.normal(size=(num_samples, 2))),
    ]
    prototype_2d = rng.normal(size=(num_classes, 2))
    class_names = [f"class{c}" for c in range(num_classes)]
    return panels, prototype_2d, sample_class_ids, list(range(num_classes)), class_names


def test_comparison_figure_is_saved(tmp_path):
    panels, prototype_2d, sample_ids, prototype_ids, names = _comparison_inputs()
    save_path = tmp_path / "compare.png"

    plot_feature_space_comparison(
        panels, prototype_2d, sample_ids, prototype_ids, names, "title", save_path
    )

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_comparison_figure_draws_one_panel_per_view(tmp_path):
    panels, prototype_2d, sample_ids, prototype_ids, names = _comparison_inputs()
    figures = []
    original_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        result = original_subplots(*args, **kwargs)
        figures.append(result)
        return result

    with mock.patch.object(plt, "subplots", side_effect=capture_subplots):
        plot_feature_space_comparison(
            panels, prototype_2d, sample_ids, prototype_ids, names, "title",
            tmp_path / "compare.png",
        )

    figure, axes = figures[0]
    assert axes.shape == (1, 3)
    assert [axis.get_title() for axis in axes[0]] == [
        "original", "after standard FM", "after rolled-out FM",
    ]


def test_all_panels_share_axis_limits(tmp_path):
    # The projection is joint, so the panels must be drawn on one scale;
    # otherwise a sample appearing to move would be a drawing artifact.
    panels, prototype_2d, sample_ids, prototype_ids, names = _comparison_inputs()
    figures = []
    original_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        result = original_subplots(*args, **kwargs)
        figures.append(result)
        return result

    with mock.patch.object(plt, "subplots", side_effect=capture_subplots):
        plot_feature_space_comparison(
            panels, prototype_2d, sample_ids, prototype_ids, names, "title",
            tmp_path / "compare.png",
        )

    axes = figures[0][1][0]
    assert len({axis.get_xlim() for axis in axes}) == 1
    assert len({axis.get_ylim() for axis in axes}) == 1


def test_every_panel_shows_the_same_prototypes(tmp_path):
    # Prototypes are the flow's fixed targets and are never transported.
    panels, prototype_2d, sample_ids, prototype_ids, names = _comparison_inputs()
    figures = []
    original_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        result = original_subplots(*args, **kwargs)
        figures.append(result)
        return result

    with mock.patch.object(plt, "subplots", side_effect=capture_subplots):
        plot_feature_space_comparison(
            panels, prototype_2d, sample_ids, prototype_ids, names, "title",
            tmp_path / "compare.png",
        )

    axes = figures[0][1][0]
    # 3 class collections + 3 prototype collections per panel.
    prototype_offsets = [
        np.concatenate([c.get_offsets() for c in axis.collections[3:]]) for axis in axes
    ]
    assert all(np.allclose(offsets, prototype_offsets[0]) for offsets in prototype_offsets)


def test_comparison_figure_rejects_empty_panels(tmp_path):
    _, prototype_2d, sample_ids, prototype_ids, names = _comparison_inputs()
    with pytest.raises(ValueError, match="empty"):
        plot_feature_space_comparison(
            [], prototype_2d, sample_ids, prototype_ids, names, "t", tmp_path / "c.png"
        )
