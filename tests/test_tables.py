from src.evaluation.tables import (
    format_accuracy_table,
    format_flow_matching_comparison_table,
    format_stage3_comparison_table,
    format_stage3_diagnostics_table,
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


# --- Stage 3 tables ---


SETTINGS = [("dtd", "dinov2_vits14"), ("flowers102", "resnet18")]


def _summary(dataset, encoder, method, accuracy, delta=None, std=0.005):
    return {
        "dataset": dataset, "encoder": encoder, "method": method, "k_shot": 10,
        "num_euler_steps": None if method == "linear_probe" else 4,
        "num_runs": 3, "mean_test_accuracy": accuracy, "std_test_accuracy": std,
        "mean_delta_accuracy": delta, "std_delta_accuracy": None if delta is None else 0.002,
    }


def test_the_comparison_table_lists_the_baseline_and_both_strategies():
    # part_3.pdf: "Report top-1 test accuracy for the Stage 1 linear probe and
    # both Stage 3 methods ... Also report the change relative to the
    # corresponding linear-probe baseline."
    summaries = [
        _summary("dtd", "dinov2_vits14", "linear_probe", 0.6858),
        _summary("dtd", "dinov2_vits14", "fm_cls_rolled", 0.6878, 0.0020),
        _summary("dtd", "dinov2_vits14", "fm_cls_guided", 0.6963, 0.0105),
    ]

    lines = format_stage3_comparison_table(
        summaries, [("dtd", "dinov2_vits14")]
    ).strip().splitlines()

    assert lines[0] == (
        "| Dataset | Encoder | Method | Test accuracy "
        "| Change vs. Stage 1 linear probe |"
    )
    assert len(lines) == 2 + 3  # header, separator, one row per method
    assert "Stage 1 linear probe (baseline)" in lines[2] and "68.58%" in lines[2]
    assert "Strategy 1" in lines[3] and "+0.20%" in lines[3]
    assert "Strategy 2" in lines[4] and "+1.05%" in lines[4]


def test_the_comparison_table_has_a_row_per_method_per_setting():
    summaries = [
        _summary("dtd", "dinov2_vits14", "linear_probe", 0.6858),
        _summary("flowers102", "resnet18", "linear_probe", 0.8322),
    ]

    lines = format_stage3_comparison_table(summaries, SETTINGS).strip().splitlines()
    body = lines[2:]

    assert len(body) == 2 * 3  # two settings, three methods each
    assert all(row.startswith("| dtd |") for row in body[:3])
    assert all(row.startswith("| flowers102 |") for row in body[3:])


def test_missing_methods_render_as_not_available():
    # A partially-completed sweep must still produce a readable table: the
    # missing methods keep their rows, with both numeric cells marked.
    summaries = [_summary("dtd", "dinov2_vits14", "linear_probe", 0.6858)]

    body = format_stage3_comparison_table(
        summaries, [("dtd", "dinov2_vits14")]
    ).strip().splitlines()[2:]

    assert len(body) == 3
    assert "n/a" not in body[0]
    assert body[1].endswith("| n/a | n/a |")
    assert body[2].endswith("| n/a | n/a |")


def test_the_baseline_shows_no_delta():
    summaries = [_summary("dtd", "dinov2_vits14", "linear_probe", 0.6858)]

    row = format_stage3_comparison_table(
        summaries, [("dtd", "dinov2_vits14")]
    ).strip().splitlines()[2]

    assert "68.58%" in row
    assert "(+" not in row and "(-" not in row


def test_the_comparison_table_only_shows_the_requested_k_shot():
    summaries = [
        _summary("dtd", "dinov2_vits14", "linear_probe", 0.6858),
        {**_summary("dtd", "dinov2_vits14", "fm_cls_guided", 0.99, 0.30), "k_shot": 5},
    ]

    table = format_stage3_comparison_table(summaries, [("dtd", "dinov2_vits14")], k_shot=10)

    assert "99.00%" not in table


def _diagnostic(dataset, method, val_delta, displacement, epochs):
    return {
        "dataset": dataset, "encoder": "dinov2_vits14", "method": method, "k_shot": 10,
        "num_runs": len(epochs), "mean_val_delta": val_delta, "std_val_delta": 0.003,
        "mean_initial_val_accuracy": 0.6812, "mean_best_val_accuracy": 0.6812 + val_delta,
        "mean_displacement": displacement, "best_epochs": epochs,
    }


def test_the_validation_table_reports_accuracy_only():
    summaries = [
        _diagnostic("dtd", "fm_cls_rolled", 0.0044, 1.92, [150, 128, 105]),
        _diagnostic("dtd", "fm_cls_guided", 0.0161, 12.58, [17, 86, 48]),
    ]

    lines = format_stage3_diagnostics_table(summaries).strip().splitlines()

    assert lines[0] == "| Dataset | Method | Runs | Baseline val | Best val | Val delta |"
    assert "+0.44%" in lines[2]
    assert "+1.61%" in lines[3]
    # The two columns taken out: neither the epochs nor the displacement.
    assert "[150, 128, 105]" not in lines[2] and "1.92" not in lines[2]
    assert "Selected epochs" not in lines[0] and "displacement" not in lines[0]


def test_an_empty_diagnostics_table_says_so():
    assert "No Stage 3 runs" in format_stage3_diagnostics_table([])



# --- the Stage and Baseline columns ---


def test_the_accuracy_table_carries_a_stage_and_baseline_column():
    summaries = [
        _summary("dtd", "dinov2_vits14", "linear_probe", 0.6858),
        _summary("dtd", "dinov2_vits14", "fm_cls_rolled", 0.6878, 0.0020),
    ]

    lines = format_accuracy_table(summaries).strip().splitlines()

    assert "| Stage |" in lines[0]
    assert lines[0].rstrip().endswith("| Baseline |")


def test_stage_1_rows_declare_no_baseline():
    summaries = [_summary("dtd", "dinov2_vits14", "linear_probe", 0.6858)]

    row = format_accuracy_table(summaries).strip().splitlines()[2]

    assert "| 1 |" in row
    assert row.rstrip().endswith("| - |")


def test_each_stages_rows_name_the_baseline_its_delta_came_from():
    # The point of the column: these two rows share a dataset, T and K, but
    # their deltas are measured from different origins.
    summaries = [
        {**_summary("dtd", "dinov2_vits14", "fm_rolled", 0.6174, -0.0672),
         "num_euler_steps": 4},
        {**_summary("dtd", "dinov2_vits14", "fm_cls_rolled", 0.6878, 0.0020),
         "num_euler_steps": 4},
    ]

    rows = format_accuracy_table(summaries).strip().splitlines()[2:]
    stage2_row = next(r for r in rows if "fm_rolled" in r and "fm_cls" not in r)
    stage3_row = next(r for r in rows if "fm_cls_rolled" in r)

    assert stage2_row.rstrip().endswith("| prototype |")
    assert stage3_row.rstrip().endswith("| linear_probe |")
    assert "| 2 |" in stage2_row
    assert "| 3 |" in stage3_row


def test_the_extension_rows_are_labelled_separately():
    summaries = [_summary("dtd", "dinov2_vits14", "fm_cls_joint", 0.6941, 0.0083)]

    row = format_accuracy_table(summaries).strip().splitlines()[2]

    assert "| 3 ext |" in row
    assert row.rstrip().endswith("| linear_probe |")


def test_the_table_still_has_one_row_per_summary():
    summaries = [
        _summary("dtd", "dinov2_vits14", "linear_probe", 0.6858),
        _summary("dtd", "dinov2_vits14", "fm_cls_rolled", 0.6878, 0.0020),
        _summary("flowers102", "resnet18", "fm_cls_guided", 0.8424, 0.0102),
    ]

    lines = format_accuracy_table(summaries).strip().splitlines()

    assert len(lines) == 2 + 3  # header, separator, three rows


def test_the_change_column_carries_its_own_spread():
    # The paired delta has its own standard deviation, which is not the
    # difference of the accuracy spreads - and the old layout dropped it.
    summaries = [
        _summary("dtd", "dinov2_vits14", "linear_probe", 0.6858),
        _summary("dtd", "dinov2_vits14", "fm_cls_guided", 0.6963, 0.0105),
    ]

    row = format_stage3_comparison_table(
        summaries, [("dtd", "dinov2_vits14")], methods=["linear_probe", "fm_cls_guided"]
    ).strip().splitlines()[3]

    assert row.endswith("| +1.05% +/- 0.20% |")


def test_markdown_and_pdf_render_the_same_rows():
    # Both renderers consume stage3_comparison_rows, so the Markdown table and
    # the PDF table cannot disagree.
    from src.evaluation.tables import rows_to_markdown, stage3_comparison_rows

    summaries = [
        _summary("dtd", "dinov2_vits14", "linear_probe", 0.6858),
        _summary("dtd", "dinov2_vits14", "fm_cls_rolled", 0.6878, 0.0020),
    ]
    rows = stage3_comparison_rows(summaries, [("dtd", "dinov2_vits14")])

    assert rows_to_markdown(rows) == format_stage3_comparison_table(
        summaries, [("dtd", "dinov2_vits14")]
    )
    assert rows[0][-1] == "Change vs. Stage 1 linear probe"

