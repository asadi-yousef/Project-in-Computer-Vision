import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn.functional as F

from src.classifiers.prototype import predict_by_cosine_similarity
from src.flow_matching.per_step import compute_per_step_metrics
from src.visualization.per_step_curves import plot_per_step_curves


def _problem(num_classes=3, per_class=6, feature_dim=5, seed=0):
    generator = torch.Generator().manual_seed(seed)
    prototypes = F.normalize(torch.randn(num_classes, feature_dim, generator=generator), dim=1)
    labels = torch.arange(num_classes).repeat_interleave(per_class)
    features = F.normalize(
        prototypes[labels] + 0.35 * torch.randn(len(labels), feature_dim, generator=generator),
        dim=1,
    )
    return features, labels, prototypes


def _straight_line_trajectory(features, labels, prototypes, num_steps):
    """The ideal flow: interpolate linearly onto each sample's own prototype."""
    targets = prototypes[labels]
    return torch.stack(
        [
            (1.0 - k / num_steps) * features + (k / num_steps) * targets
            for k in range(num_steps + 1)
        ]
    )


# --- the metrics themselves ---


def test_one_entry_per_state():
    features, labels, prototypes = _problem()
    trajectory = _straight_line_trajectory(features, labels, prototypes, 4)

    metrics = compute_per_step_metrics(trajectory, labels, prototypes)

    assert metrics.steps == [0, 1, 2, 3, 4]
    assert len(metrics.accuracies) == 5
    assert len(metrics.mean_own_similarity) == 5
    assert len(metrics.mean_margin) == 5


def test_times_are_normalized_so_different_step_counts_align():
    # Comparing by step index would place step 4 of a 4-step flow (the end)
    # alongside step 4 of a 12-step flow (a third of the way).
    features, labels, prototypes = _problem()

    short = compute_per_step_metrics(
        _straight_line_trajectory(features, labels, prototypes, 4), labels, prototypes
    )
    long = compute_per_step_metrics(
        _straight_line_trajectory(features, labels, prototypes, 12), labels, prototypes
    )

    assert short.times[0] == 0.0 and short.times[-1] == 1.0
    assert long.times[0] == 0.0 and long.times[-1] == 1.0
    assert short.times[2] == pytest.approx(0.5)
    assert long.times[6] == pytest.approx(0.5)


def test_step_zero_reproduces_the_untransported_baseline():
    # z_hat_0 is the original feature, so step 0 must equal exactly what the
    # Stage 1 prototype classifier reports. This makes the curve
    # self-validating against the headline table.
    features, labels, prototypes = _problem()
    trajectory = _straight_line_trajectory(features, labels, prototypes, 12)

    metrics = compute_per_step_metrics(trajectory, labels, prototypes)

    baseline = (predict_by_cosine_similarity(features, prototypes) == labels).float().mean()
    assert metrics.accuracies[0] == pytest.approx(baseline.item())


def test_the_ideal_flow_reaches_perfect_accuracy():
    # Interpolating every sample onto its own prototype must end at 100%.
    features, labels, prototypes = _problem()
    trajectory = _straight_line_trajectory(features, labels, prototypes, 4)

    metrics = compute_per_step_metrics(trajectory, labels, prototypes)

    assert metrics.accuracies[-1] == pytest.approx(1.0)
    assert metrics.mean_own_similarity[-1] == pytest.approx(1.0, abs=1e-5)


def test_own_similarity_increases_along_the_ideal_flow():
    features, labels, prototypes = _problem()
    trajectory = _straight_line_trajectory(features, labels, prototypes, 12)

    metrics = compute_per_step_metrics(trajectory, labels, prototypes)

    similarities = metrics.mean_own_similarity
    assert all(b >= a - 1e-6 for a, b in zip(similarities, similarities[1:]))


def test_a_stationary_flow_leaves_every_metric_unchanged():
    features, labels, prototypes = _problem()
    trajectory = features.unsqueeze(0).repeat(5, 1, 1)

    metrics = compute_per_step_metrics(trajectory, labels, prototypes)

    assert len(set(metrics.accuracies)) == 1
    assert metrics.mean_margin[0] == pytest.approx(metrics.mean_margin[-1])


def test_rising_own_similarity_can_coincide_with_a_shrinking_margin():
    # The distinction the margin panel exists to expose, built explicitly.
    # Two orthogonal prototypes; samples start almost orthogonal to both,
    # leaning very slightly toward their own. Flowing to the midpoint of the
    # two prototypes raises similarity to *both* - so own-prototype
    # similarity climbs sharply while the margin that decides the prediction
    # collapses to zero.
    prototypes = torch.eye(2, 4)
    labels = torch.tensor([0, 1])
    features = F.normalize(
        torch.tensor([[0.05, 0.0, 1.0, 0.0], [0.0, 0.05, 1.0, 0.0]]), dim=1
    )
    midpoint = F.normalize(prototypes.mean(dim=0, keepdim=True), dim=1).expand_as(features)
    trajectory = torch.stack(
        [(1.0 - k / 4) * features + (k / 4) * midpoint for k in range(5)]
    )

    metrics = compute_per_step_metrics(trajectory, labels, prototypes)

    assert metrics.mean_own_similarity[-1] > metrics.mean_own_similarity[0]
    assert metrics.mean_margin[-1] < metrics.mean_margin[0]
    assert metrics.mean_margin[-1] == pytest.approx(0.0, abs=1e-6)


def test_margin_is_positive_exactly_when_the_prediction_is_correct():
    features, labels, prototypes = _problem()
    trajectory = features.unsqueeze(0)

    metrics = compute_per_step_metrics(trajectory, labels, prototypes)

    similarities = F.normalize(features, dim=1) @ prototypes.T
    correct = (similarities.argmax(dim=1) == labels).float().mean().item()
    assert metrics.accuracies[0] == pytest.approx(correct)


def test_single_step_trajectory_does_not_divide_by_zero():
    features, labels, prototypes = _problem()
    trajectory = features.unsqueeze(0)

    metrics = compute_per_step_metrics(trajectory, labels, prototypes)

    assert metrics.times == [0.0]


# --- input validation ---


def test_non_3d_trajectory_raises():
    features, labels, prototypes = _problem()
    with pytest.raises(ValueError, match="T\\+1, N, D"):
        compute_per_step_metrics(features, labels, prototypes)


def test_sample_count_mismatch_raises():
    features, labels, prototypes = _problem()
    trajectory = _straight_line_trajectory(features, labels, prototypes, 4)
    with pytest.raises(ValueError, match="samples"):
        compute_per_step_metrics(trajectory, labels[:-1], prototypes)


def test_feature_dimension_mismatch_raises():
    features, labels, prototypes = _problem()
    trajectory = _straight_line_trajectory(features, labels, prototypes, 4)
    with pytest.raises(ValueError, match="dimension"):
        compute_per_step_metrics(trajectory, labels, prototypes[:, :-1])


# --- the figure ---


def _series():
    features, labels, prototypes = _problem()
    return [
        (
            "fm_standard", num_euler_steps,
            compute_per_step_metrics(
                _straight_line_trajectory(features, labels, prototypes, num_euler_steps),
                labels, prototypes,
            ),
        )
        for num_euler_steps in (4, 12)
    ]


def test_figure_is_saved(tmp_path):
    save_path = tmp_path / "per_step.png"

    plot_per_step_curves(_series(), "dtd", "resnet18", 10, save_path, baseline_accuracy=0.5)

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_figure_has_three_panels(tmp_path, monkeypatch):
    figures = []
    original_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        result = original_subplots(*args, **kwargs)
        figures.append(result)
        return result

    monkeypatch.setattr(plt, "subplots", capture_subplots)
    plot_per_step_curves(_series(), "dtd", "resnet18", 10, tmp_path / "p.png")

    _, axes = figures[0]
    assert len(axes) == 3
    assert "accuracy" in axes[0].get_title()
    assert "contraction" in axes[1].get_title()
    assert "separation" in axes[2].get_title()


def test_baseline_reference_is_drawn_only_when_given(tmp_path, monkeypatch):
    figures = []
    original_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        result = original_subplots(*args, **kwargs)
        figures.append(result)
        return result

    monkeypatch.setattr(plt, "subplots", capture_subplots)

    plot_per_step_curves(_series(), "dtd", "resnet18", 10, tmp_path / "a.png")
    without = [line.get_label() for line in figures[0][1][0].get_lines()]

    plot_per_step_curves(
        _series(), "dtd", "resnet18", 10, tmp_path / "b.png", baseline_accuracy=0.5
    )
    with_baseline = [line.get_label() for line in figures[1][1][0].get_lines()]

    assert "prototype baseline" not in without
    assert "prototype baseline" in with_baseline


def test_figure_rejects_empty_series(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        plot_per_step_curves([], "dtd", "resnet18", 10, tmp_path / "p.png")
