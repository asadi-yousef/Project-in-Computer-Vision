"""Tests for Stage 3's validation-driven hyperparameter search.

The selection logic is tested on hand-built summaries, `evaluate_config` on
synthetic tensors, and `search_setting` end to end against the real project
(skipped when the cache or Stage 1 outputs are absent).
"""

import dataclasses
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.classifiers.linear_probe import LinearProbe
from src.classifiers.linear_probe_runner import linear_probe_run_dir
from src.flow_matching.stage3_runner import Stage3Data
from src.flow_matching.stage3_tuning import (
    STRATEGY_GRIDS,
    TuningRun,
    TuningSummary,
    build_grid,
    evaluate_config,
    format_tuning_table,
    load_tuning_results,
    save_tuning_results,
    search_setting,
    select_best,
    summarize_runs,
)
from src.utils.config import Stage3Hyperparams
from src.utils.seeding import set_seed

CPU = torch.device("cpu")
FEATURE_DIM = 12
NUM_CLASSES = 4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_OUTPUTS = PROJECT_ROOT / "outputs"
REAL_CACHE = PROJECT_ROOT / "cache"


# --- build_grid ---


def test_grid_sizes_match_the_declared_values():
    assert len(build_grid("fm_cls_guided")) == 4 * 2 * 3
    assert len(build_grid("fm_cls_rolled")) == 5 * 2


def test_every_grid_point_is_a_valid_hyperparameter_override():
    # A typo in a grid key would otherwise surface only after hours of
    # training, as a dataclasses.replace TypeError.
    for method, grid in STRATEGY_GRIDS.items():
        for overrides in build_grid(method):
            dataclasses.replace(Stage3Hyperparams(), **overrides)


def test_grid_points_are_unique_and_stable():
    grid = build_grid("fm_cls_guided")
    keys = [tuple(sorted(point.items())) for point in grid]

    assert len(set(keys)) == len(grid)
    assert grid == build_grid("fm_cls_guided")


def test_an_unknown_method_has_no_grid():
    with pytest.raises(ValueError, match="fm_cls_rolled"):
        build_grid("fm_standard")


def test_the_searched_knobs_are_the_ones_the_spec_names():
    # part_3.pdf names the step size, the number of target-improvement steps
    # and the recompute frequency for Strategy 2, and the two penalties for
    # Strategy 1. Architecture, T and the learning rate are deliberately out.
    assert set(STRATEGY_GRIDS["fm_cls_guided"]) == {
        "target_step_size", "target_num_steps", "target_refresh_epochs",
    }
    assert set(STRATEGY_GRIDS["fm_cls_rolled"]) == {
        "displacement_penalty", "velocity_penalty",
    }
    for grid in STRATEGY_GRIDS.values():
        assert "hidden_dims" not in grid
        assert "num_euler_steps" not in grid
        assert "learning_rate" not in grid


# --- summarize_runs / select_best ---


def _run(overrides, seed, val_delta, test_delta, displacement=1.0, best_epoch=10):
    return TuningRun(
        method="fm_cls_guided", dataset="dtd", encoder="dinov2_vits14", seed=seed,
        overrides=dict(overrides),
        initial_val_accuracy=0.68, best_val_accuracy=0.68 + val_delta,
        val_delta=val_delta,
        baseline_test_accuracy=0.686, test_accuracy=0.686 + test_delta,
        test_delta=test_delta, best_epoch=best_epoch,
        test_mean_displacement=displacement,
    )


def _summary(val_delta, test_delta=0.0, displacement=1.0, **overrides):
    return summarize_runs(
        [_run(overrides, seed, val_delta, test_delta, displacement) for seed in (0, 1, 2)]
    )


def test_summarize_averages_across_seeds():
    runs = [
        _run({"target_step_size": 0.1}, 0, 0.01, 0.005),
        _run({"target_step_size": 0.1}, 1, 0.02, 0.015),
        _run({"target_step_size": 0.1}, 2, 0.03, 0.010),
    ]

    summary = summarize_runs(runs)

    assert summary.num_seeds == 3
    assert summary.mean_val_delta == pytest.approx(0.02)
    assert summary.mean_test_delta == pytest.approx(0.01)
    assert summary.std_val_delta == pytest.approx(0.01)
    assert summary.best_epochs == [10, 10, 10]
    assert summary.overrides == {"target_step_size": 0.1}


def test_a_single_seed_has_no_standard_deviation():
    summary = summarize_runs([_run({"target_step_size": 0.1}, 0, 0.01, 0.005)])

    assert summary.num_seeds == 1
    assert summary.std_val_delta is None
    assert summary.std_test_delta is None


def test_summarizing_nothing_raises():
    with pytest.raises(ValueError, match="nothing to summarize"):
        summarize_runs([])


def test_summarizing_mixed_configurations_raises():
    runs = [
        _run({"target_step_size": 0.1}, 0, 0.01, 0.0),
        _run({"target_step_size": 0.2}, 1, 0.01, 0.0),
    ]

    with pytest.raises(ValueError, match="share one configuration"):
        summarize_runs(runs)


def test_selection_maximizes_the_validation_delta():
    summaries = [
        _summary(val_delta=0.005, target_step_size=0.02),
        _summary(val_delta=0.020, target_step_size=0.05),
        _summary(val_delta=0.010, target_step_size=0.10),
    ]

    assert select_best(summaries).overrides == {"target_step_size": 0.05}


def test_selection_ignores_test_accuracy():
    # The rule that keeps the search honest: the configuration with the best
    # test delta must not win unless validation also prefers it.
    summaries = [
        _summary(val_delta=0.020, test_delta=0.001, target_step_size=0.05),
        _summary(val_delta=0.005, test_delta=0.900, target_step_size=0.20),
    ]

    assert select_best(summaries).overrides == {"target_step_size": 0.05}


def test_ties_prefer_the_smaller_displacement():
    summaries = [
        _summary(val_delta=0.02, displacement=50.0, target_step_size=0.2),
        _summary(val_delta=0.02, displacement=2.0, target_step_size=0.05),
    ]

    assert select_best(summaries).overrides == {"target_step_size": 0.05}


def test_selecting_from_nothing_raises():
    with pytest.raises(ValueError, match="nothing to select"):
        select_best([])


# --- reporting and persistence ---


def test_the_table_lists_configurations_in_the_given_order():
    summaries = [
        _summary(val_delta=0.02, target_step_size=0.05),
        _summary(val_delta=0.01, target_step_size=0.10),
    ]

    table = format_tuning_table(summaries)
    lines = table.splitlines()

    assert "Val delta (selection)" in lines[0]
    assert "target_step_size=0.05" in lines[2]
    assert "target_step_size=0.1" in lines[3]


def test_the_table_can_be_truncated():
    summaries = [_summary(val_delta=0.01 * i, target_step_size=i) for i in range(5)]

    table = format_tuning_table(summaries, top_n=2)

    assert len(table.splitlines()) == 2 + 2  # header, separator, two rows


def test_an_empty_table_says_so():
    assert "No tuning results" in format_tuning_table([])


def test_results_round_trip_through_json(tmp_path):
    summaries = [
        _summary(val_delta=0.02, test_delta=0.01, target_step_size=0.05),
        _summary(val_delta=0.01, test_delta=0.00, target_step_size=0.10),
    ]
    path = tmp_path / "stage3_tuning.json"

    save_tuning_results(summaries, path)
    loaded = load_tuning_results(path)

    assert [s.overrides for s in loaded] == [s.overrides for s in summaries]
    assert loaded[0].mean_val_delta == pytest.approx(summaries[0].mean_val_delta)
    assert isinstance(loaded[0], TuningSummary)


# --- evaluate_config ---


def _synthetic_data(num_per_class=6, seed=0):
    generator = torch.Generator().manual_seed(seed)

    def split(count):
        labels = torch.arange(NUM_CLASSES).repeat_interleave(count)
        features = torch.randn(len(labels), FEATURE_DIM, generator=generator) * 24.0
        features += F.one_hot(labels, FEATURE_DIM).float() * 24.0
        return features, labels

    train_features, train_labels = split(num_per_class)
    val_features, val_labels = split(4)
    test_features, test_labels = split(4)
    return Stage3Data(
        train_features, train_labels, val_features, val_labels,
        test_features, test_labels, NUM_CLASSES,
    )


def test_evaluate_config_reports_deltas_against_the_right_references():
    set_seed(0)
    classifier = LinearProbe(FEATURE_DIM, NUM_CLASSES)
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    data = _synthetic_data()

    with torch.no_grad():
        baseline = (
            classifier(data.test_features).argmax(dim=1) == data.test_labels
        ).float().mean().item()

    run = evaluate_config(
        "fm_cls_rolled", data, classifier, baseline,
        Stage3Hyperparams(hidden_dims=[16, 16], num_euler_steps=2, max_epochs=2),
        "dtd", "resnet18", seed=0, device=CPU,
    )

    # The validation delta is measured against this run's own identity start,
    # and the test delta against Stage 1's published number.
    assert run.val_delta == pytest.approx(
        run.best_val_accuracy - run.initial_val_accuracy
    )
    assert run.test_delta == pytest.approx(run.test_accuracy - baseline)
    assert run.baseline_test_accuracy == pytest.approx(baseline)
    assert 1 <= run.best_epoch <= 2


# --- Integration against the real project ---


def test_search_setting_runs_a_small_grid_on_real_data():
    dataset, encoder = "dtd", "dinov2_vits14"
    if not (REAL_CACHE / dataset / encoder / "test.pt").exists():
        pytest.skip(f"no cached features for {dataset}/{encoder}")
    if not linear_probe_run_dir(REAL_OUTPUTS, dataset, encoder, 10, 0).exists():
        pytest.skip("Stage 1 run not completed")

    grid = [{"target_step_size": 0.05}, {"target_step_size": 0.2}]
    seen = []

    summaries = search_setting(
        "fm_cls_guided", dataset, encoder, 10, [0], REAL_CACHE, REAL_OUTPUTS, CPU,
        base_hyperparams=Stage3Hyperparams(
            hidden_dims=[32, 32], num_euler_steps=2, max_epochs=2
        ),
        grid=grid,
        progress=seen.append,
    )

    assert len(summaries) == 2
    assert len(seen) == 2
    assert {s.overrides["target_step_size"] for s in summaries} == {0.05, 0.2}
    # Sorted best-first by the selection statistic.
    assert summaries[0].mean_val_delta >= summaries[1].mean_val_delta
    assert select_best(summaries) is summaries[0] or summaries[0].mean_val_delta == summaries[1].mean_val_delta


def test_the_search_table_omits_epochs_and_displacement():
    summaries = [_summary(val_delta=0.02, displacement=12.58, target_step_size=0.05)]

    lines = format_tuning_table(summaries).splitlines()

    assert lines[0] == (
        "| Configuration | Val delta (selection) | Test delta | Test accuracy |"
    )
    assert "12.58" not in lines[2]
    assert "[10, 10, 10]" not in lines[2]


def test_displacement_still_breaks_ties_after_leaving_the_table():
    # Taking the column out of the rendered table must not take the number
    # out of the data selection depends on.
    summaries = [
        _summary(val_delta=0.02, displacement=50.0, target_step_size=0.2),
        _summary(val_delta=0.02, displacement=2.0, target_step_size=0.05),
    ]

    assert select_best(summaries).overrides == {"target_step_size": 0.05}

