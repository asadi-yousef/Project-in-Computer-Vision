from src.evaluation.tables import format_accuracy_table


def test_table_includes_mean_and_std_when_available():
    summaries = [
        {
            "dataset": "dtd", "encoder": "resnet18", "method": "linear_probe", "k_shot": 10,
            "num_runs": 3, "mean_test_accuracy": 0.6, "std_test_accuracy": 0.1,
            "seed_accuracies": {0: 0.5, 1: 0.6, 2: 0.7},
        }
    ]
    table = format_accuracy_table(summaries)

    assert "dtd" in table
    assert "resnet18" in table
    assert "linear_probe" in table
    assert "60.00%" in table
    assert "+/- 10.00%" in table


def test_table_omits_std_for_single_run_settings():
    summaries = [
        {
            "dataset": "dtd", "encoder": "resnet18", "method": "prototype", "k_shot": "full",
            "num_runs": 1, "mean_test_accuracy": 0.8, "std_test_accuracy": None,
            "seed_accuracies": {0: 0.8},
        }
    ]
    table = format_accuracy_table(summaries)

    assert "80.00%" in table
    assert "+/-" not in table


def test_table_has_a_row_per_summary():
    summaries = [
        {"dataset": "dtd", "encoder": "resnet18", "method": "linear_probe", "k_shot": 5,
         "num_runs": 3, "mean_test_accuracy": 0.3, "std_test_accuracy": 0.05, "seed_accuracies": {}},
        {"dataset": "dtd", "encoder": "resnet18", "method": "linear_probe", "k_shot": 10,
         "num_runs": 3, "mean_test_accuracy": 0.5, "std_test_accuracy": 0.05, "seed_accuracies": {}},
    ]
    table = format_accuracy_table(summaries)
    data_rows = [line for line in table.splitlines() if line.startswith("| dtd")]
    assert len(data_rows) == 2
