import json
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from src.visualization.loss_curves import (
    load_history,
    plot_flow_matching_loss_curves,
    plot_loss_curve,
)


def _synthetic_history():
    return [
        {"epoch": 1, "train_loss": 1.5, "val_loss": 1.6, "train_accuracy": 0.2, "val_accuracy": 0.15, "learning_rate": 1e-3},
        {"epoch": 2, "train_loss": 1.2, "val_loss": 1.3, "train_accuracy": 0.4, "val_accuracy": 0.35, "learning_rate": 1e-3},
        {"epoch": 3, "train_loss": 0.9, "val_loss": 1.1, "train_accuracy": 0.6, "val_accuracy": 0.5, "learning_rate": 1e-3},
    ]


def test_load_history_reads_json_list(tmp_path):
    history = _synthetic_history()
    path = tmp_path / "history.json"
    with open(path, "w") as f:
        json.dump(history, f)

    loaded = load_history(path)

    assert loaded == history


def test_plot_loss_curve_is_saved_to_disk(tmp_path):
    save_path = tmp_path / "loss.png"
    plot_loss_curve(_synthetic_history(), "dtd", "resnet18", save_path)

    assert save_path.exists()
    assert save_path.stat().st_size > 0


# --- Stage 2: flow-matching training curves ---


def _fm_history(start, end, epochs=20, val_offset=0.3):
    step = (start - end) / max(epochs - 1, 1)
    return [
        {"epoch": i + 1, "train_loss": start - i * step, "val_loss": start - i * step + val_offset}
        for i in range(epochs)
    ]


def _fm_history_without_validation(epochs=5):
    return [
        {"epoch": i + 1, "train_loss": 1.0 - 0.1 * i, "val_loss": float("nan")}
        for i in range(epochs)
    ]


def test_flow_matching_curves_are_saved_to_disk(tmp_path):
    save_path = tmp_path / "fm_loss.png"

    plot_flow_matching_loss_curves(
        _fm_history(0.6, 0.13),
        {4: _fm_history(0.6, 0.014), 12: _fm_history(0.6, 0.016)},
        "dtd", "resnet18", 10, 0, save_path,
    )

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_the_two_panels_have_independent_y_axes(tmp_path):
    # The two objectives measure different quantities; a shared axis would
    # invite reading rolled-out's smaller values as better training.
    captured = {}
    original_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        captured["sharey"] = kwargs.get("sharey", False)
        return original_subplots(*args, **kwargs)

    with mock.patch.object(plt, "subplots", side_effect=capture_subplots):
        plot_flow_matching_loss_curves(
            _fm_history(0.6, 0.13),
            {4: _fm_history(0.6, 0.014)},
            "dtd", "resnet18", 10, 0, tmp_path / "fm.png",
        )

    assert captured["sharey"] is False


def test_standard_panel_holds_one_training_and_rolled_panel_one_per_step_count(tmp_path):
    figures = []
    original_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        result = original_subplots(*args, **kwargs)
        figures.append(result)
        return result

    with mock.patch.object(plt, "subplots", side_effect=capture_subplots):
        plot_flow_matching_loss_curves(
            _fm_history(0.6, 0.13),
            {4: _fm_history(0.6, 0.014), 12: _fm_history(0.6, 0.016)},
            "dtd", "resnet18", 10, 0, tmp_path / "fm.png",
        )

    _, (standard_axis, rolled_axis) = figures[0]
    standard_labels = [line.get_label() for line in standard_axis.get_lines()]
    rolled_labels = [line.get_label() for line in rolled_axis.get_lines()]

    assert standard_labels == ["train", "validation"]
    assert rolled_labels == [
        "train (T=4)", "validation (T=4)", "train (T=12)", "validation (T=12)",
    ]


def test_step_counts_are_colour_coded_in_the_rolled_panel(tmp_path):
    figures = []
    original_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        result = original_subplots(*args, **kwargs)
        figures.append(result)
        return result

    with mock.patch.object(plt, "subplots", side_effect=capture_subplots):
        plot_flow_matching_loss_curves(
            _fm_history(0.6, 0.13),
            {4: _fm_history(0.6, 0.014), 12: _fm_history(0.6, 0.016)},
            "dtd", "resnet18", 10, 0, tmp_path / "fm.png",
        )

    _, (_, rolled_axis) = figures[0]
    colors = {line.get_label(): line.get_color() for line in rolled_axis.get_lines()}

    assert colors["train (T=4)"] == colors["validation (T=4)"]
    assert colors["train (T=4)"] != colors["train (T=12)"]


def test_train_and_validation_are_distinguished_by_line_style(tmp_path):
    figures = []
    original_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        result = original_subplots(*args, **kwargs)
        figures.append(result)
        return result

    with mock.patch.object(plt, "subplots", side_effect=capture_subplots):
        plot_flow_matching_loss_curves(
            _fm_history(0.6, 0.13), {4: _fm_history(0.6, 0.014)},
            "dtd", "resnet18", 10, 0, tmp_path / "fm.png",
        )

    _, (standard_axis, _) = figures[0]
    styles = {line.get_label(): line.get_linestyle() for line in standard_axis.get_lines()}

    assert styles["train"] != styles["validation"]


def test_all_nan_validation_series_is_omitted(tmp_path):
    # A run trained without a validation split logs NaN; plotting it would
    # produce an empty line and a misleading legend entry.
    figures = []
    original_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        result = original_subplots(*args, **kwargs)
        figures.append(result)
        return result

    with mock.patch.object(plt, "subplots", side_effect=capture_subplots):
        plot_flow_matching_loss_curves(
            _fm_history_without_validation(),
            {4: _fm_history_without_validation()},
            "dtd", "resnet18", 10, 0, tmp_path / "fm.png",
        )

    _, (standard_axis, rolled_axis) = figures[0]
    assert [line.get_label() for line in standard_axis.get_lines()] == ["train"]
    assert [line.get_label() for line in rolled_axis.get_lines()] == ["train (T=4)"]


def test_empty_standard_history_raises(tmp_path):
    with pytest.raises(ValueError, match="standard_history"):
        plot_flow_matching_loss_curves(
            [], {4: _fm_history(0.6, 0.014)}, "dtd", "resnet18", 10, 0, tmp_path / "fm.png"
        )


def test_empty_rolled_histories_raises(tmp_path):
    with pytest.raises(ValueError, match="rolled_histories"):
        plot_flow_matching_loss_curves(
            _fm_history(0.6, 0.13), {}, "dtd", "resnet18", 10, 0, tmp_path / "fm.png"
        )
