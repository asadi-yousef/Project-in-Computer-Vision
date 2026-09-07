import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes

from src.visualization.accuracy_vs_shot import plot_accuracy_vs_shot, plot_delta_vs_shot


def _synthetic_summaries():
    return [
        {"dataset": "dtd", "encoder": "resnet18", "method": "linear_probe", "k_shot": 5,
         "num_runs": 3, "mean_test_accuracy": 0.3, "std_test_accuracy": 0.05, "seed_accuracies": {}},
        {"dataset": "dtd", "encoder": "resnet18", "method": "linear_probe", "k_shot": 10,
         "num_runs": 3, "mean_test_accuracy": 0.5, "std_test_accuracy": 0.04, "seed_accuracies": {}},
        {"dataset": "dtd", "encoder": "resnet18", "method": "linear_probe", "k_shot": "full",
         "num_runs": 3, "mean_test_accuracy": 0.7, "std_test_accuracy": 0.02, "seed_accuracies": {}},
        {"dataset": "dtd", "encoder": "resnet18", "method": "prototype", "k_shot": 5,
         "num_runs": 3, "mean_test_accuracy": 0.25, "std_test_accuracy": 0.03, "seed_accuracies": {}},
        {"dataset": "dtd", "encoder": "resnet18", "method": "prototype", "k_shot": "full",
         "num_runs": 1, "mean_test_accuracy": 0.4, "std_test_accuracy": None, "seed_accuracies": {}},
    ]


def test_plot_is_saved_to_disk(tmp_path):
    save_path = tmp_path / "plot.png"
    plot_accuracy_vs_shot(_synthetic_summaries(), "dtd", "resnet18", save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_raises_for_missing_dataset_encoder_combination():
    with pytest.raises(ValueError, match="flowers102"):
        plot_accuracy_vs_shot(_synthetic_summaries(), "flowers102", "resnet18", "unused.png")


# --- Stage 2: one series per (method, T) ---


def _fm_summaries():
    summaries = [
        {"dataset": "dtd", "encoder": "resnet18", "method": "prototype", "k_shot": k,
         "num_euler_steps": None, "num_runs": 3, "mean_test_accuracy": acc,
         "std_test_accuracy": 0.01, "mean_delta_accuracy": None,
         "std_delta_accuracy": None, "seed_accuracies": {}}
        for k, acc in [(5, 0.47), (10, 0.52), ("full", 0.59)]
    ]
    for method, base in [("fm_standard", 0.46), ("fm_rolled", 0.41)]:
        for num_euler_steps in (4, 12):
            for k, offset in [(5, 0.0), (10, 0.03), ("full", 0.10)]:
                summaries.append({
                    "dataset": "dtd", "encoder": "resnet18", "method": method, "k_shot": k,
                    "num_euler_steps": num_euler_steps, "num_runs": 3,
                    "mean_test_accuracy": base + offset, "std_test_accuracy": 0.012,
                    "mean_delta_accuracy": -0.02 + offset, "std_delta_accuracy": 0.005,
                    "seed_accuracies": {},
                })
    return summaries


def _capture_series(monkeypatch):
    """Record each plotted series' label, line style and colour.

    errorbar() puts the label on an ErrorbarContainer rather than on a
    Line2D, so both the containers and any plain lines (e.g. the delta
    plot's zero reference) have to be inspected.
    """
    captured = {}
    original_legend = Axes.legend

    def capture_legend(self, *args, **kwargs):
        series = {}
        for container in self.containers:
            data_line = container.lines[0]
            series[container.get_label()] = (data_line.get_linestyle(), data_line.get_color())
        for line in self.get_lines():
            label = line.get_label()
            if not label.startswith("_") and label not in series:
                series[label] = (line.get_linestyle(), line.get_color())
        captured["series"] = series
        captured["labels"] = list(series)
        return original_legend(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "legend", capture_legend)
    return captured


def test_series_are_labelled_by_method_and_step_count(tmp_path, monkeypatch):
    captured = _capture_series(monkeypatch)

    plot_accuracy_vs_shot(_fm_summaries(), "dtd", "resnet18", tmp_path / "p.png")

    assert set(captured["labels"]) == {
        "prototype",
        "fm_standard (T=4)", "fm_standard (T=12)",
        "fm_rolled (T=4)", "fm_rolled (T=12)",
    }


def test_legend_orders_baseline_before_flow_matching(tmp_path, monkeypatch):
    captured = _capture_series(monkeypatch)

    plot_accuracy_vs_shot(_fm_summaries(), "dtd", "resnet18", tmp_path / "p.png")

    assert captured["labels"][0] == "prototype"
    assert captured["labels"][1:3] == ["fm_standard (T=4)", "fm_standard (T=12)"]


def test_methods_filter_restricts_the_series(tmp_path, monkeypatch):
    captured = _capture_series(monkeypatch)

    plot_accuracy_vs_shot(
        _fm_summaries(), "dtd", "resnet18", tmp_path / "p.png",
        methods=["prototype", "fm_standard"],
    )

    assert all("rolled" not in label for label in captured["labels"])
    assert len(captured["labels"]) == 3


def test_methods_filter_that_matches_nothing_raises(tmp_path):
    with pytest.raises(ValueError, match="methods"):
        plot_accuracy_vs_shot(
            _fm_summaries(), "dtd", "resnet18", tmp_path / "p.png", methods=["nonexistent"]
        )


def test_colour_encodes_method_and_line_style_encodes_step_count(tmp_path, monkeypatch):
    captured = _capture_series(monkeypatch)

    plot_accuracy_vs_shot(_fm_summaries(), "dtd", "resnet18", tmp_path / "p.png")

    series = captured["series"]
    # Same method, different T -> same colour, different line style.
    assert series["fm_standard (T=4)"][1] == series["fm_standard (T=12)"][1]
    assert series["fm_standard (T=4)"][0] != series["fm_standard (T=12)"][0]
    # Different method -> different colour.
    assert series["fm_standard (T=4)"][1] != series["fm_rolled (T=4)"][1]


def test_custom_title_is_applied(tmp_path, monkeypatch):
    captured = {}
    original_set_title = Axes.set_title

    def capture_title(self, label, *args, **kwargs):
        captured["title"] = label
        return original_set_title(self, label, *args, **kwargs)

    monkeypatch.setattr(Axes, "set_title", capture_title)
    plot_accuracy_vs_shot(
        _fm_summaries(), "dtd", "resnet18", tmp_path / "p.png", title="custom title"
    )

    assert captured["title"] == "custom title"


def test_single_run_settings_plot_without_error_bars(tmp_path):
    # std is None for the K=full settings; this must render, not crash.
    summaries = [
        {"dataset": "dtd", "encoder": "resnet18", "method": "fm_standard", "k_shot": "full",
         "num_euler_steps": 4, "num_runs": 1, "mean_test_accuracy": 0.5952,
         "std_test_accuracy": None, "mean_delta_accuracy": 0.0074,
         "std_delta_accuracy": None, "seed_accuracies": {}},
    ]

    plot_accuracy_vs_shot(summaries, "dtd", "resnet18", tmp_path / "p.png")

    assert (tmp_path / "p.png").exists()


# --- delta plot ---


def test_delta_plot_is_saved(tmp_path):
    plot_delta_vs_shot(_fm_summaries(), "dtd", "resnet18", tmp_path / "d.png")
    assert (tmp_path / "d.png").exists()


def test_delta_plot_shows_only_flow_matching_series_plus_a_zero_reference(tmp_path, monkeypatch):
    captured = _capture_series(monkeypatch)

    plot_delta_vs_shot(_fm_summaries(), "dtd", "resnet18", tmp_path / "d.png")

    labels = captured["labels"]
    assert "prototype baseline" in labels  # the zero reference line
    assert "prototype" not in labels  # not plotted as a series
    assert len([label for label in labels if label.startswith("fm_")]) == 4


def test_delta_plot_raises_without_flow_matching_results(tmp_path):
    with pytest.raises(ValueError, match="flow-matching"):
        plot_delta_vs_shot(_synthetic_summaries(), "dtd", "resnet18", tmp_path / "d.png")


# --- the delta plot's baseline must match the methods on it ---


def _mixed_stage_summaries():
    """Stage 2 and Stage 3 deltas for one setting.

    They are measured against different baselines - the prototype classifier
    and the linear probe respectively - so they must never share an axis
    whose zero line represents only one of them.
    """
    def summary(method, delta, num_euler_steps=4):
        return {
            "dataset": "dtd", "encoder": "dinov2_vits14", "method": method,
            "k_shot": 10, "num_euler_steps": num_euler_steps, "num_runs": 3,
            "mean_test_accuracy": 0.7, "std_test_accuracy": 0.01,
            "mean_delta_accuracy": delta, "std_delta_accuracy": 0.002,
            "seed_accuracies": {},
        }

    return [
        summary("fm_standard", -0.03),
        summary("fm_rolled", -0.07),
        summary("fm_cls_rolled", 0.002),
        summary("fm_cls_guided", 0.010),
    ]


def _plotted_series_labels(monkeypatch, **kwargs):
    captured = []
    original = Axes.errorbar

    def spy(self, *args, **inner):
        captured.append(inner.get("label"))
        return original(self, *args, **inner)

    monkeypatch.setattr(Axes, "errorbar", spy)
    plot_delta_vs_shot(_mixed_stage_summaries(), "dtd", "dinov2_vits14", **kwargs)
    plt.close("all")
    return captured


def test_the_delta_plot_can_be_restricted_to_one_stages_methods(monkeypatch, tmp_path):
    # Regression test. Without the filter, adding Stage 3 runs to outputs/
    # silently put linear-probe-relative series onto the Stage 2 figure,
    # whose zero line is the prototype baseline.
    labels = _plotted_series_labels(
        monkeypatch,
        save_path=tmp_path / "stage2.png",
        methods=["fm_standard", "fm_rolled"],
    )

    assert labels
    assert all("fm_cls" not in (label or "") for label in labels)


def test_the_delta_plot_shows_every_method_when_unrestricted(monkeypatch, tmp_path):
    labels = _plotted_series_labels(monkeypatch, save_path=tmp_path / "all.png")

    assert any("fm_cls" in (label or "") for label in labels)


def test_the_baseline_line_can_be_relabelled(monkeypatch, tmp_path):
    captured = {}
    original = Axes.axhline

    def spy(self, *args, **kwargs):
        captured["label"] = kwargs.get("label")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "axhline", spy)
    plot_delta_vs_shot(
        _mixed_stage_summaries(), "dtd", "dinov2_vits14",
        save_path=tmp_path / "f.png", methods=["fm_cls_guided"],
        baseline_label="linear-probe baseline",
    )
    plt.close("all")

    assert captured["label"] == "linear-probe baseline"


def test_restricting_to_a_method_with_no_results_raises(tmp_path):
    with pytest.raises(ValueError, match="No flow-matching results"):
        plot_delta_vs_shot(
            _mixed_stage_summaries(), "dtd", "dinov2_vits14",
            save_path=tmp_path / "f.png", methods=["prototype"],
        )

