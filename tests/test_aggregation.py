import json
import statistics
from pathlib import Path

import pytest

from src.evaluation.aggregation import aggregate_results, load_all_results, method_label


def _write_result(output_dir, dataset, encoder, method, k_shot, seed, test_accuracy):
    run_dir = Path(output_dir) / method / dataset / encoder / f"k{k_shot}" / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "config": {
            "dataset": dataset, "encoder": encoder, "method": method,
            "k_shot": k_shot, "seed": seed,
        },
        "result": {"test_accuracy": test_accuracy},
    }
    with open(run_dir / "result.json", "w") as f:
        json.dump(data, f)


def test_load_all_results_finds_every_result_file(tmp_path):
    _write_result(tmp_path, "dtd", "resnet18", "linear_probe", 10, 0, 0.5)
    _write_result(tmp_path, "dtd", "resnet18", "linear_probe", 10, 1, 0.6)

    records = load_all_results(tmp_path)

    assert len(records) == 2
    assert {r["seed"] for r in records} == {0, 1}


def test_aggregate_computes_mean_and_sample_std():
    records = [
        {"dataset": "dtd", "encoder": "resnet18", "method": "linear_probe", "k_shot": 10, "seed": 0, "test_accuracy": 0.5},
        {"dataset": "dtd", "encoder": "resnet18", "method": "linear_probe", "k_shot": 10, "seed": 1, "test_accuracy": 0.7},
        {"dataset": "dtd", "encoder": "resnet18", "method": "linear_probe", "k_shot": 10, "seed": 2, "test_accuracy": 0.6},
    ]

    summaries = aggregate_results(records)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["num_runs"] == 3
    assert abs(summary["mean_test_accuracy"] - 0.6) < 1e-9
    assert abs(summary["std_test_accuracy"] - 0.1) < 1e-9  # stdev([0.5,0.6,0.7]) == 0.1 exactly
    assert summary["seed_accuracies"] == {0: 0.5, 1: 0.7, 2: 0.6}


def test_aggregate_returns_none_std_for_single_run():
    records = [
        {"dataset": "dtd", "encoder": "resnet18", "method": "prototype", "k_shot": "full", "seed": 0, "test_accuracy": 0.8},
    ]

    summaries = aggregate_results(records)

    assert summaries[0]["num_runs"] == 1
    assert summaries[0]["std_test_accuracy"] is None


def test_aggregate_separates_different_settings():
    records = [
        {"dataset": "dtd", "encoder": "resnet18", "method": "linear_probe", "k_shot": 5, "seed": 0, "test_accuracy": 0.3},
        {"dataset": "dtd", "encoder": "resnet18", "method": "linear_probe", "k_shot": 10, "seed": 0, "test_accuracy": 0.5},
        {"dataset": "flowers102", "encoder": "resnet18", "method": "linear_probe", "k_shot": 5, "seed": 0, "test_accuracy": 0.4},
    ]

    summaries = aggregate_results(records)

    assert len(summaries) == 3


def test_aggregate_orders_k_shot_as_5_then_10_then_full():
    records = [
        {"dataset": "dtd", "encoder": "resnet18", "method": "linear_probe", "k_shot": "full", "seed": 0, "test_accuracy": 0.9},
        {"dataset": "dtd", "encoder": "resnet18", "method": "linear_probe", "k_shot": 5, "seed": 0, "test_accuracy": 0.3},
        {"dataset": "dtd", "encoder": "resnet18", "method": "linear_probe", "k_shot": 10, "seed": 0, "test_accuracy": 0.5},
    ]

    summaries = aggregate_results(records)

    assert [s["k_shot"] for s in summaries] == [5, 10, "full"]


# --- Stage 2: Euler-step count as part of the setting key ---


def _fm_record(method, k_shot, seed, num_euler_steps, test_accuracy, baseline):
    return {
        "dataset": "dtd", "encoder": "resnet18", "method": method,
        "k_shot": k_shot, "seed": seed, "test_accuracy": test_accuracy,
        "num_euler_steps": num_euler_steps,
        "baseline_test_accuracy": baseline,
        "delta_accuracy": test_accuracy - baseline,
    }


def test_step_counts_are_separate_settings():
    # Without T in the key these six runs would collapse into one group and
    # the T=4 and T=12 results would be averaged together.
    records = [
        _fm_record("fm_standard", 5, seed, num_euler_steps, 0.4 + 0.01 * seed, 0.45)
        for num_euler_steps in (4, 12)
        for seed in (0, 1, 2)
    ]

    summaries = aggregate_results(records)

    assert len(summaries) == 2
    assert {s["num_euler_steps"] for s in summaries} == {4, 12}
    assert all(s["num_runs"] == 3 for s in summaries)


def test_stage_1_records_get_a_none_step_count():
    records = [
        {"dataset": "dtd", "encoder": "resnet18", "method": "prototype",
         "k_shot": "full", "seed": 0, "test_accuracy": 0.8},
    ]

    summaries = aggregate_results(records)

    assert summaries[0]["num_euler_steps"] is None
    assert summaries[0]["mean_delta_accuracy"] is None
    assert summaries[0]["std_delta_accuracy"] is None


def test_delta_is_averaged_from_paired_per_seed_baselines():
    # Each seed samples a different subset, so it has its own baseline. The
    # reported delta must average those paired differences rather than
    # subtracting one mean from another.
    records = [
        _fm_record("fm_standard", 5, 0, 4, 0.50, 0.40),  # +0.10
        _fm_record("fm_standard", 5, 1, 4, 0.60, 0.58),  # +0.02
        _fm_record("fm_standard", 5, 2, 4, 0.55, 0.52),  # +0.03
    ]

    summary = aggregate_results(records)[0]

    assert summary["mean_delta_accuracy"] == pytest.approx(0.05)
    assert summary["mean_baseline_accuracy"] == pytest.approx(0.50)
    assert summary["std_delta_accuracy"] == pytest.approx(
        statistics.stdev([0.10, 0.02, 0.03])
    )


def test_single_run_delta_has_no_std():
    records = [_fm_record("fm_rolled", "full", 0, 12, 0.73, 0.72)]

    summary = aggregate_results(records)[0]

    assert summary["mean_delta_accuracy"] == pytest.approx(0.01)
    assert summary["std_delta_accuracy"] is None


def test_methods_are_ordered_baselines_first_then_flow_matching():
    # Alphabetical order would put fm_rolled ahead of the prototype baseline
    # it is compared against.
    records = [
        {"dataset": "dtd", "encoder": "resnet18", "method": m, "k_shot": 5,
         "seed": 0, "test_accuracy": 0.5}
        for m in ("fm_rolled", "prototype", "fm_standard", "linear_probe")
    ]

    summaries = aggregate_results(records)

    assert [s["method"] for s in summaries] == [
        "linear_probe", "prototype", "fm_standard", "fm_rolled",
    ]


def test_step_counts_sort_ascending_within_a_method():
    records = [
        _fm_record("fm_standard", 5, 0, 12, 0.5, 0.45),
        _fm_record("fm_standard", 5, 0, 4, 0.5, 0.45),
    ]

    summaries = aggregate_results(records)

    assert [s["num_euler_steps"] for s in summaries] == [4, 12]


def test_method_label_includes_step_count_only_when_present():
    assert method_label("prototype", None) == "prototype"
    assert method_label("fm_rolled", 12) == "fm_rolled (T=12)"


def test_load_all_results_fills_missing_flow_matching_keys(tmp_path):
    _write_result(tmp_path, "dtd", "resnet18", "prototype", 10, 0, 0.5)

    records = load_all_results(tmp_path)

    assert records[0]["num_euler_steps"] is None
    assert records[0]["delta_accuracy"] is None


def test_load_all_results_keeps_flow_matching_fields(tmp_path):
    run_dir = tmp_path / "fm_rolled" / "dtd" / "resnet18" / "k5" / "T12" / "seed0"
    run_dir.mkdir(parents=True)
    with open(run_dir / "result.json", "w") as f:
        json.dump(
            {
                "config": {"dataset": "dtd", "encoder": "resnet18", "method": "fm_rolled",
                           "k_shot": 5, "seed": 0},
                "result": {"test_accuracy": 0.4, "baseline_test_accuracy": 0.47,
                           "delta_accuracy": -0.07, "num_euler_steps": 12},
            },
            f,
        )

    record = load_all_results(tmp_path)[0]

    assert record["num_euler_steps"] == 12
    assert record["delta_accuracy"] == pytest.approx(-0.07)
