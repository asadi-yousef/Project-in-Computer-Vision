from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from src.visualization.flow_trajectories import (
    plot_flow_trajectories,
    project_trajectories,
    select_trajectory_examples,
)


def _test_labels(num_classes=5, samples_per_class=12):
    return torch.tensor([c for c in range(num_classes) for _ in range(samples_per_class)])


# --- selecting which examples to trace ---


def test_selection_returns_the_requested_count_per_class():
    labels = _test_labels()

    selected = select_trajectory_examples(labels, class_ids=[0, 2, 4], examples_per_class=2)

    assert len(selected) == 6
    assert sorted(labels[selected].tolist()) == [0, 0, 2, 2, 4, 4]


def test_selection_is_sorted_and_reproducible():
    labels = _test_labels()

    first = select_trajectory_examples(labels, [0, 1], examples_per_class=3, seed=7)
    second = select_trajectory_examples(labels, [0, 1], examples_per_class=3, seed=7)

    assert first == second
    assert first == sorted(first)


def test_different_seeds_select_different_examples():
    labels = _test_labels()

    first = select_trajectory_examples(labels, [0, 1], examples_per_class=3, seed=0)
    second = select_trajectory_examples(labels, [0, 1], examples_per_class=3, seed=1)

    assert first != second


def test_selection_raises_when_a_class_has_too_few_samples():
    labels = _test_labels(num_classes=2, samples_per_class=2)

    with pytest.raises(ValueError, match="fewer than"):
        select_trajectory_examples(labels, [0], examples_per_class=5)


# --- the joint PCA projection ---


def test_projection_preserves_trajectory_shape():
    standard = torch.randn(5, 8, 6)  # (T+1, N, D) with T=4
    rolled = torch.randn(13, 8, 6)  # a different T
    prototypes = torch.randn(3, 6)

    (standard_2d, rolled_2d), prototype_2d, ratio = project_trajectories(
        [standard, rolled], prototypes
    )

    assert standard_2d.shape == (5, 8, 2)
    assert rolled_2d.shape == (13, 8, 2)
    assert prototype_2d.shape == (3, 2)
    assert len(ratio) == 2


def test_projection_fits_once_over_everything():
    # One coordinate system across both panels and the prototypes, so a
    # longer step in one panel really is a longer step.
    standard = torch.randn(5, 4, 6)
    rolled = torch.randn(5, 4, 6)
    prototypes = torch.randn(3, 6)
    captured = {}

    class FakePCA:
        def __init__(self, **kwargs):
            self.explained_variance_ratio_ = np.array([0.5, 0.3])

        def fit_transform(self, array):
            captured["calls"] = captured.get("calls", 0) + 1
            captured["shape"] = array.shape
            return np.zeros((array.shape[0], 2))

    with mock.patch("src.visualization.flow_trajectories.PCA", FakePCA):
        project_trajectories([standard, rolled], prototypes)

    assert captured["calls"] == 1
    assert captured["shape"] == (5 * 4 + 5 * 4 + 3, 6)


def test_projection_is_linear_so_straight_paths_stay_straight():
    # The whole reason the spec recommends PCA here. A path moving in equal
    # steps along a line must project to equally spaced collinear points.
    start = torch.zeros(6)
    direction = torch.randn(6)
    trajectory = torch.stack([start + (k / 4.0) * direction for k in range(5)]).unsqueeze(1)
    prototypes = (start + direction).unsqueeze(0)

    (projected,), _, _ = project_trajectories([trajectory], prototypes)

    path = projected[:, 0, :]
    steps = np.diff(path, axis=0)
    assert np.allclose(steps, steps[0], atol=1e-4)


def test_projection_does_not_normalize_the_states():
    # Trajectories are drawn in raw feature space; normalizing each state
    # would project away the drift off the unit sphere that actually happens.
    trajectory = (torch.randn(3, 4, 6) * 25.0).contiguous()
    prototypes = torch.nn.functional.normalize(torch.randn(2, 6), dim=1)
    captured = {}

    class FakePCA:
        def __init__(self, **kwargs):
            self.explained_variance_ratio_ = np.array([0.5, 0.3])

        def fit_transform(self, array):
            captured["array"] = array
            return np.zeros((array.shape[0], 2))

    with mock.patch("src.visualization.flow_trajectories.PCA", FakePCA):
        project_trajectories([trajectory], prototypes)

    norms = np.linalg.norm(captured["array"][: 3 * 4], axis=1)
    assert not np.allclose(norms, 1.0, atol=1e-3)


def test_projection_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        project_trajectories([], torch.randn(3, 6))


def test_projection_rejects_mismatched_dimensions():
    with pytest.raises(ValueError, match="dimension"):
        project_trajectories([torch.randn(5, 4, 6)], torch.randn(3, 8))


# --- the figure ---


def _panel_inputs(num_classes=3, examples_per_class=2, num_steps=4):
    sample_class_ids = [c for c in range(num_classes) for _ in range(examples_per_class)]
    num_samples = len(sample_class_ids)
    rng = np.random.default_rng(0)
    panels = [
        ("standard FM", rng.normal(size=(num_steps + 1, num_samples, 2))),
        ("rolled-out FM", rng.normal(size=(num_steps + 1, num_samples, 2))),
    ]
    prototype_2d = rng.normal(size=(num_classes, 2))
    class_names = [f"class{c}" for c in range(num_classes)]
    return panels, prototype_2d, sample_class_ids, list(range(num_classes)), class_names


def test_figure_is_saved(tmp_path):
    panels, prototype_2d, ids, prototype_ids, names = _panel_inputs()
    save_path = tmp_path / "traj.png"

    plot_flow_trajectories(panels, prototype_2d, ids, prototype_ids, names, "title", save_path)

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def _capture_axes(monkeypatch):
    figures = []
    original_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        result = original_subplots(*args, **kwargs)
        figures.append(result)
        return result

    monkeypatch.setattr(plt, "subplots", capture_subplots)
    return figures


def test_one_panel_per_variant_with_shared_limits(tmp_path, monkeypatch):
    figures = _capture_axes(monkeypatch)
    panels, prototype_2d, ids, prototype_ids, names = _panel_inputs()

    plot_flow_trajectories(
        panels, prototype_2d, ids, prototype_ids, names, "title", tmp_path / "t.png"
    )

    axes = figures[0][1][0]
    assert [axis.get_title() for axis in axes] == ["standard FM", "rolled-out FM"]
    assert len({axis.get_xlim() for axis in axes}) == 1
    assert len({axis.get_ylim() for axis in axes}) == 1


def test_one_path_is_drawn_per_traced_example(tmp_path, monkeypatch):
    figures = _capture_axes(monkeypatch)
    panels, prototype_2d, ids, prototype_ids, names = _panel_inputs(
        num_classes=3, examples_per_class=2
    )

    plot_flow_trajectories(
        panels, prototype_2d, ids, prototype_ids, names, "title", tmp_path / "t.png"
    )

    for axis in figures[0][1][0]:
        assert len(axis.get_lines()) == 6  # 3 classes x 2 examples


def test_each_class_is_labelled_exactly_once(tmp_path, monkeypatch):
    # Two examples share a class; labelling both would duplicate the legend.
    figures = _capture_axes(monkeypatch)
    panels, prototype_2d, ids, prototype_ids, names = _panel_inputs()

    plot_flow_trajectories(
        panels, prototype_2d, ids, prototype_ids, names, "title", tmp_path / "t.png"
    )

    axis = figures[0][1][0][0]
    labels = [line.get_label() for line in axis.get_lines() if not line.get_label().startswith("_")]
    assert sorted(labels) == ["class0", "class1", "class2"]


def test_explained_variance_appears_on_the_axis_labels(tmp_path, monkeypatch):
    figures = _capture_axes(monkeypatch)
    panels, prototype_2d, ids, prototype_ids, names = _panel_inputs()

    plot_flow_trajectories(
        panels, prototype_2d, ids, prototype_ids, names, "title", tmp_path / "t.png",
        explained_variance_ratio=[0.412, 0.187],
    )

    axes = figures[0][1][0]
    assert "41.2%" in axes[0].get_xlabel()
    assert "18.7%" in axes[0].get_ylabel()


def test_figure_rejects_empty_panels(tmp_path):
    _, prototype_2d, ids, prototype_ids, names = _panel_inputs()
    with pytest.raises(ValueError, match="empty"):
        plot_flow_trajectories(
            [], prototype_2d, ids, prototype_ids, names, "t", tmp_path / "t.png"
        )


def test_panels_can_opt_out_of_shared_limits(tmp_path, monkeypatch):
    # Reverse flow can diverge in one panel by orders of magnitude, which
    # would squash the other panel to a single dot under shared limits.
    figures = _capture_axes(monkeypatch)
    panels, prototype_2d, ids, prototype_ids, names = _panel_inputs()
    panels = [panels[0], (panels[1][0], panels[1][1] * 5000.0)]

    plot_flow_trajectories(
        panels, prototype_2d, ids, prototype_ids, names, "title", tmp_path / "t.png",
        share_limits=False,
    )

    axes = figures[0][1][0]
    assert len({axis.get_xlim() for axis in axes}) == 2


def test_shared_limits_remain_the_default(tmp_path, monkeypatch):
    figures = _capture_axes(monkeypatch)
    panels, prototype_2d, ids, prototype_ids, names = _panel_inputs()
    panels = [panels[0], (panels[1][0], panels[1][1] * 5000.0)]

    plot_flow_trajectories(
        panels, prototype_2d, ids, prototype_ids, names, "title", tmp_path / "t.png"
    )

    axes = figures[0][1][0]
    assert len({axis.get_xlim() for axis in axes}) == 1


def test_custom_legend_title_is_used(tmp_path, monkeypatch):
    captured = {}
    original_legend = plt.Figure.legend

    def capture_legend(self, *args, **kwargs):
        captured["title"] = kwargs.get("title")
        return original_legend(self, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "legend", capture_legend)
    panels, prototype_2d, ids, prototype_ids, names = _panel_inputs()

    plot_flow_trajectories(
        panels, prototype_2d, ids, prototype_ids, names, "title", tmp_path / "t.png",
        legend_title="reverse key",
    )

    assert captured["title"] == "reverse key"


def test_a_single_prototype_array_is_shared_by_every_panel(tmp_path, monkeypatch):
    figures = _capture_axes(monkeypatch)
    panels, prototype_2d, ids, prototype_ids, names = _panel_inputs()

    plot_flow_trajectories(
        panels, prototype_2d, ids, prototype_ids, names, "title", tmp_path / "t.png"
    )

    axes = figures[0][1][0]
    drawn = [
        np.concatenate([c.get_offsets() for c in axis.collections[-len(prototype_ids):]])
        for axis in axes
    ]
    assert np.allclose(drawn[0], drawn[1])


def test_per_panel_prototypes_are_drawn_in_their_own_panel(tmp_path, monkeypatch):
    # Each panel projected in its own basis gives the prototypes different
    # coordinates per panel; reusing one panel's would misplace them.
    figures = _capture_axes(monkeypatch)
    panels, prototype_2d, ids, prototype_ids, names = _panel_inputs()
    per_panel = [prototype_2d, prototype_2d + 100.0]

    plot_flow_trajectories(
        panels, per_panel, ids, prototype_ids, names, "title", tmp_path / "t.png",
        share_limits=False,
    )

    axes = figures[0][1][0]
    drawn = [
        np.concatenate([c.get_offsets() for c in axis.collections[-len(prototype_ids):]])
        for axis in axes
    ]
    assert not np.allclose(drawn[0], drawn[1])
    assert np.allclose(drawn[1] - drawn[0], 100.0)


def test_wrong_number_of_prototype_arrays_raises(tmp_path):
    panels, prototype_2d, ids, prototype_ids, names = _panel_inputs()

    with pytest.raises(ValueError, match="prototype arrays"):
        plot_flow_trajectories(
            panels, [prototype_2d], ids, prototype_ids, names, "t", tmp_path / "t.png"
        )


def test_background_samples_are_drawn_behind_the_paths(tmp_path, monkeypatch):
    # stage_2.pdf's optional item asks to compare samples and prototypes, so
    # real test features have to appear alongside the reverse-flow paths.
    figures = _capture_axes(monkeypatch)
    panels, prototype_2d, ids, prototype_ids, names = _panel_inputs()
    rng = np.random.default_rng(1)
    background = [
        (rng.normal(size=(30, 2)), [c for c in range(3) for _ in range(10)])
        for _ in panels
    ]

    plot_flow_trajectories(
        panels, prototype_2d, ids, prototype_ids, names, "title", tmp_path / "t.png",
        background=background,
    )

    for axis in figures[0][1][0]:
        # One faint collection per background class, drawn beneath everything.
        background_layers = [c for c in axis.collections if c.get_zorder() == 1]
        assert len(background_layers) == 3
        # Translucent and behind the paths, but still large enough to read.
        assert all(c.get_alpha() < 1.0 for c in background_layers)
        assert all(c.get_sizes()[0] >= 30 for c in background_layers)


def test_background_points_are_included_in_the_axis_limits(tmp_path, monkeypatch):
    figures = _capture_axes(monkeypatch)
    panels, prototype_2d, ids, prototype_ids, names = _panel_inputs()
    far_away = np.array([[500.0, 500.0]])
    background = [(far_away, [0]) for _ in panels]

    plot_flow_trajectories(
        panels, prototype_2d, ids, prototype_ids, names, "title", tmp_path / "t.png",
        background=background,
    )

    assert figures[0][1][0][0].get_xlim()[1] > 400


def test_background_is_optional(tmp_path, monkeypatch):
    figures = _capture_axes(monkeypatch)
    panels, prototype_2d, ids, prototype_ids, names = _panel_inputs()

    plot_flow_trajectories(
        panels, prototype_2d, ids, prototype_ids, names, "title", tmp_path / "t.png"
    )

    for axis in figures[0][1][0]:
        assert not [c for c in axis.collections if c.get_zorder() == 1]


def test_wrong_number_of_background_sets_raises(tmp_path):
    panels, prototype_2d, ids, prototype_ids, names = _panel_inputs()
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError, match="background sets"):
        plot_flow_trajectories(
            panels, prototype_2d, ids, prototype_ids, names, "t", tmp_path / "t.png",
            background=[(rng.normal(size=(5, 2)), [0] * 5)],
        )
