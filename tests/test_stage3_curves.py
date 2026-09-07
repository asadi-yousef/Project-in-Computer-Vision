"""Tests for the Stage 3 training-curve figure.

Matplotlib figures are checked structurally - panels, references, labels -
rather than by comparing pixels, which is how the other plotting tests in
this project work.
"""

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.visualization.stage3_curves import (
    TRAIN_OBJECTIVE_LABELS,
    Stage3CurveRun,
    plot_stage3_curves,
)
from src.visualization.style import METHOD_COLORS, method_color


def _history(num_epochs=10, train_loss=1.0, accuracy=0.5):
    return [
        {
            "epoch": epoch,
            "train_loss": train_loss / epoch,
            "val_loss": 1.5 - 0.01 * epoch,
            "train_accuracy": min(1.0, accuracy + 0.02 * epoch),
            "val_accuracy": accuracy + 0.005 * epoch,
            "mean_displacement": 0.1 * epoch,
        }
        for epoch in range(1, num_epochs + 1)
    ]


def _runs():
    return [
        Stage3CurveRun("fm_cls_rolled", _history(), initial_val_accuracy=0.50, best_epoch=7),
        Stage3CurveRun(
            "fm_cls_guided", _history(train_loss=200.0), initial_val_accuracy=0.50,
            best_epoch=3,
        ),
    ]


def test_the_figure_is_written(tmp_path):
    path = tmp_path / "figures" / "stage3_curves.png"

    plot_stage3_curves(_runs(), "dtd", "dinov2_vits14", 10, 0, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_one_column_per_method_and_two_rows(tmp_path):
    captured = {}
    original = plt.subplots

    def spy(*args, **kwargs):
        fig, axes = original(*args, **kwargs)
        captured["shape"] = axes.shape
        return fig, axes

    plt.subplots = spy
    try:
        plot_stage3_curves(_runs(), "dtd", "dinov2_vits14", 10, 0, tmp_path / "f.png")
    finally:
        plt.subplots = original

    assert captured["shape"] == (2, 2)  # (loss, accuracy) x (rolled, guided)


def test_the_accuracy_panel_marks_the_baseline_and_the_selected_epoch(tmp_path):
    # The point of the figure: the horizontal line is the Stage 1 probe, so
    # the gap to the validation curve is what Stage 3 contributed.
    figures_before = set(plt.get_fignums())
    saved = {}

    original_savefig = plt.Figure.savefig

    def spy(self, *args, **kwargs):
        saved["figure"] = self
        # Capture before the caller closes it.
        saved["axes"] = list(self.axes)
        saved["lines"] = [list(axis.lines) for axis in self.axes]
        return original_savefig(self, *args, **kwargs)

    plt.Figure.savefig = spy
    try:
        plot_stage3_curves(
            [Stage3CurveRun("fm_cls_rolled", _history(), 0.42, best_epoch=6)],
            "dtd", "dinov2_vits14", 10, 0, tmp_path / "f.png",
        )
    finally:
        plt.Figure.savefig = original_savefig
        for number in set(plt.get_fignums()) - figures_before:
            plt.close(number)

    # Accuracy panel is the second primary axis; the twin adds a third.
    accuracy_lines = saved["lines"][1]
    horizontals = [
        line for line in accuracy_lines
        if len(set(line.get_ydata())) == 1 and abs(line.get_ydata()[0] - 42.0) < 1e-6
    ]
    verticals = [
        line for line in accuracy_lines
        if len(set(line.get_xdata())) == 1 and abs(line.get_xdata()[0] - 6) < 1e-6
    ]

    assert horizontals, "no baseline reference line at the initial validation accuracy"
    assert verticals, "no marker at the selected epoch"


def test_the_loss_panel_uses_a_twin_axis(tmp_path):
    # Strategy 2's velocity loss and the pipeline's cross-entropy are in
    # different units; sharing one axis would invite reading them together.
    figures_before = set(plt.get_fignums())
    saved = {}
    original_savefig = plt.Figure.savefig

    def spy(self, *args, **kwargs):
        saved["num_axes"] = len(self.axes)
        return original_savefig(self, *args, **kwargs)

    plt.Figure.savefig = spy
    try:
        plot_stage3_curves(
            [Stage3CurveRun("fm_cls_guided", _history(), 0.5, 3)],
            "dtd", "dinov2_vits14", 10, 0, tmp_path / "f.png",
        )
    finally:
        plt.Figure.savefig = original_savefig
        for number in set(plt.get_fignums()) - figures_before:
            plt.close(number)

    # 2 panels + 1 twin on the loss panel.
    assert saved["num_axes"] == 3


def test_each_strategy_declares_what_its_training_loss_measures():
    # The two objectives are not comparable, so the axis must say which is
    # which rather than leaving a reader to remember.
    assert set(TRAIN_OBJECTIVE_LABELS) == {"fm_cls_rolled", "fm_cls_guided"}
    assert TRAIN_OBJECTIVE_LABELS["fm_cls_rolled"] != TRAIN_OBJECTIVE_LABELS["fm_cls_guided"]


def test_the_stage_3_methods_have_their_own_colours():
    # Distinct from each other and from the Stage 2 pair, so a reader never
    # confuses fm_rolled with fm_cls_rolled across figures.
    assert method_color("fm_cls_rolled") != method_color("fm_cls_guided")
    assert method_color("fm_cls_rolled") != method_color("fm_rolled")
    assert method_color("fm_cls_guided") != method_color("fm_standard")
    assert len(set(METHOD_COLORS.values())) == len(METHOD_COLORS)


def test_an_empty_run_list_raises():
    with pytest.raises(ValueError, match="nothing to plot"):
        plot_stage3_curves([], "dtd", "dinov2_vits14", 10, 0, "unused.png")


def test_an_empty_history_raises(tmp_path):
    with pytest.raises(ValueError, match="fm_cls_rolled"):
        plot_stage3_curves(
            [Stage3CurveRun("fm_cls_rolled", [], 0.5, 1)],
            "dtd", "dinov2_vits14", 10, 0, tmp_path / "f.png",
        )


def test_a_single_method_still_plots(tmp_path):
    # A partially-completed sweep should still produce a figure.
    path = tmp_path / "one.png"

    plot_stage3_curves(
        [Stage3CurveRun("fm_cls_guided", _history(), 0.5, 4)],
        "flowers102", "resnet18", 10, 1, path,
    )

    assert path.exists()
