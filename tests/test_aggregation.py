import json
from pathlib import Path

from src.evaluation.aggregation import aggregate_results, load_all_results


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
