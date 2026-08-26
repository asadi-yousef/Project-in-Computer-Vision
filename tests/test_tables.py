from src.evaluation.tables import (
    format_accuracy_table,
    format_flow_matching_comparison_table,
)


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


# --- Stage 2: T and delta columns ---


def _fm_summary(method, num_euler_steps, k_shot, accuracy, std, delta, delta_std, runs=3):
    return {
        "dataset": "dtd", "encoder": "resnet18", "method": method, "k_shot": k_shot,
        "num_euler_steps": num_euler_steps, "num_runs": runs,
        "mean_test_accuracy": accuracy, "std_test_accuracy": std,
        "mean_baseline_accuracy": accuracy - delta,
        "mean_delta_accuracy": delta, "std_delta_accuracy": delta_std,
        "seed_accuracies": {},
    }


def test_accuracy_table_shows_step_count_and_signed_delta():
    table = format_accuracy_table(
        [_fm_summary("fm_standard", 12, 10, 0.7717, 0.0021, 0.0195, 0.0030)]
    )

    assert "| 12 |" in table
    assert "77.17%" in table
    assert "+1.95%" in table


def test_accuracy_table_shows_negative_deltas_with_a_sign():
    table = format_accuracy_table(
        [_fm_summary("fm_rolled", 4, 10, 0.4062, 0.0100, -0.1135, 0.0090)]
    )

    assert "-11.35%" in table


def test_accuracy_table_dashes_the_stage_1_columns():
    # Stage 1 rows have no T and no baseline to compare against.
    summaries = [
        {"dataset": "dtd", "encoder": "resnet18", "method": "prototype", "k_shot": "full",
         "num_runs": 1, "mean_test_accuracy": 0.5878, "std_test_accuracy": None,
         "num_euler_steps": None, "mean_delta_accuracy": None, "std_delta_accuracy": None,
         "seed_accuracies": {}},
    ]

    row = [line for line in format_accuracy_table(summaries).splitlines()
           if line.startswith("| dtd")][0]

    assert "| - |" in row
    assert row.rstrip().endswith("| - |")


def test_accuracy_table_omits_delta_std_for_single_run_settings():
    table = format_accuracy_table(
        [_fm_summary("fm_standard", 4, "full", 0.5952, None, 0.0074, None, runs=1)]
    )

    assert "+0.74%" in table
    assert "+/-" not in table


def test_comparison_table_has_one_row_per_k_and_five_condition_columns():
    summaries = (
        [{"dataset": "dtd", "encoder": "resnet18", "method": "prototype", "k_shot": k,
          "num_euler_steps": None, "num_runs": 3, "mean_test_accuracy": 0.5,
          "std_test_accuracy": 0.01, "mean_delta_accuracy": None,
          "std_delta_accuracy": None, "seed_accuracies": {}} for k in (5, 10, "full")]
        + [_fm_summary(m, t, k, 0.52, 0.01, 0.02, 0.005)
           for m in ("fm_standard", "fm_rolled") for t in (4, 12) for k in (5, 10, "full")]
    )

    table = format_flow_matching_comparison_table(summaries, "dtd", "resnet18")

    lines = table.splitlines()
    assert "prototype" in lines[0]
    assert "fm_standard (T=4)" in lines[0]
    assert "fm_rolled (T=12)" in lines[0]
    data_rows = [line for line in lines if line.startswith("| 5 |") or line.startswith("| 10 |")
                 or line.startswith("| full |")]
    assert len(data_rows) == 3
    assert data_rows[0].count("|") == 7  # 5 condition columns + K column


def test_comparison_table_marks_missing_settings_as_not_available():
    # A partially-completed sweep must still render.
    summaries = [_fm_summary("fm_standard", 4, 5, 0.52, 0.01, 0.02, 0.005)]

    table = format_flow_matching_comparison_table(summaries, "dtd", "resnet18")

    assert "n/a" in table


def test_comparison_table_only_includes_the_requested_pair():
    summaries = [
        _fm_summary("fm_standard", 4, 5, 0.52, 0.01, 0.02, 0.005),
        {**_fm_summary("fm_standard", 4, 5, 0.99, 0.01, 0.02, 0.005), "dataset": "flowers102"},
    ]

    table = format_flow_matching_comparison_table(summaries, "dtd", "resnet18")

    assert "52.00%" in table
    assert "99.00%" not in table
