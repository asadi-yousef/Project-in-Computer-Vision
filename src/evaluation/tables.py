"""Render aggregated results (see aggregation.py) as Markdown tables."""

from typing import List, Optional, Sequence, Tuple

from src.evaluation.aggregation import (
    METHOD_DISPLAY_ORDER,
    STAGE3_COMPARISON_METHODS,
    method_baseline,
    method_label,
    method_stage,
)


def _format_percentage(mean: float, std: Optional[float], signed: bool = False) -> str:
    """Render "mean +/- std" as percentages, or just the mean when std is None.

    A single-run setting shows no "+/-": a "+/- 0" would misleadingly imply
    variance was measured and found to be zero.
    """
    sign = "+" if signed else ""
    text = f"{mean * 100:{sign}.2f}%"
    if std is None:
        return text
    return f"{text} +/- {std * 100:.2f}%"


def format_accuracy_table(summaries: List[dict]) -> str:
    """Render aggregated summaries as a GitHub-flavored Markdown table.

    Columns cover every stage: the Stage 1 methods have no Euler-step count
    and no baseline to compare against, so their T, delta and baseline cells
    show "-".

    The delta column is the mean of each run's *paired* delta against its own
    baseline (see `aggregate_results`), not a difference of column means. The
    Baseline column names which baseline that is, because it differs by
    stage - Stage 2 is measured against the prototype classifier and Stage 3
    against the linear probe - and without it two deltas on adjacent rows
    look comparable when they are distances from different origins.
    """
    header = (
        "| Dataset | Encoder | Stage | Method | T | K-shot | Runs "
        "| Test Accuracy | Delta vs baseline | Baseline |\n"
    )
    header += "|---" * 10 + "|\n"

    rows = []
    for summary in summaries:
        accuracy_text = _format_percentage(
            summary["mean_test_accuracy"], summary["std_test_accuracy"]
        )
        num_euler_steps = summary.get("num_euler_steps")
        euler_text = "-" if num_euler_steps is None else str(num_euler_steps)

        mean_delta = summary.get("mean_delta_accuracy")
        if mean_delta is None:
            delta_text = "-"
        else:
            delta_text = _format_percentage(
                mean_delta, summary.get("std_delta_accuracy"), signed=True
            )

        baseline = method_baseline(summary["method"])
        rows.append(
            f"| {summary['dataset']} | {summary['encoder']} "
            f"| {method_stage(summary['method'])} | {summary['method']} | "
            f"{euler_text} | {summary['k_shot']} | {summary['num_runs']} | "
            f"{accuracy_text} | {delta_text} | {baseline or '-'} |"
        )

    return header + "\n".join(rows) + "\n"


def format_flow_matching_comparison_table(summaries: List[dict], dataset: str, encoder: str) -> str:
    """Render the Stage 2 comparison for one dataset/encoder as one row per K.

    stage_2.pdf asks for a table that makes the baseline-versus-FM comparison
    clear at a glance. The general accuracy table has one row per setting,
    which spreads the five conditions for a given K across five rows; this
    puts them side by side instead, with the prototype baseline first.

    Settings absent from `summaries` render as "n/a" rather than raising, so
    a partially-completed sweep still produces a readable table.
    """
    relevant = [s for s in summaries if s["dataset"] == dataset and s["encoder"] == encoder]
    by_setting = {
        (s["method"], s.get("num_euler_steps"), s["k_shot"]): s for s in relevant
    }

    columns = [("prototype", None), ("fm_standard", 4), ("fm_standard", 12),
               ("fm_rolled", 4), ("fm_rolled", 12)]

    header = "| K-shot | " + " | ".join(method_label(m, t) for m, t in columns) + " |\n"
    header += "|---" * (len(columns) + 1) + "|\n"

    rows = []
    for k_shot in (5, 10, "full"):
        cells = []
        for method, num_euler_steps in columns:
            summary = by_setting.get((method, num_euler_steps, k_shot))
            if summary is None:
                cells.append("n/a")
                continue
            text = _format_percentage(
                summary["mean_test_accuracy"], summary["std_test_accuracy"]
            )
            mean_delta = summary.get("mean_delta_accuracy")
            if mean_delta is not None:
                text += f" ({mean_delta * 100:+.2f})"
            cells.append(text)
        rows.append(f"| {k_shot} | " + " | ".join(cells) + " |")

    return header + "\n".join(rows) + "\n"


# How each method is named in the Stage 3 comparison tables. part_3.pdf
# describes the comparison in words - "Stage 1 linear probe; end-to-end
# rolled-out classification training; classifier-guided FM training" - so
# the rows use those words, with the code name kept alongside so a reader can
# match a row to the rest of the report. The baseline is labelled as such,
# because a column or row headed only `linear_probe` does not say that every
# delta in the table is measured from it.
STAGE3_METHOD_LABELS = {
    "linear_probe": "Stage 1 linear probe (baseline)",
    "fm_cls_rolled": "Strategy 1: end-to-end rolled-out (fm_cls_rolled)",
    "fm_cls_guided": "Strategy 2: classifier-guided FM (fm_cls_guided)",
    "cls_finetune": "Control: classifier fine-tuned alone (cls_finetune)",
    "fm_cls_joint": "Extension: joint FM + classifier (fm_cls_joint)",
}

STAGE3_COMPARISON_HEADER = [
    "Dataset", "Encoder", "Method", "Test accuracy", "Change vs. Stage 1 linear probe",
]


def stage3_comparison_rows(
    summaries: List[dict],
    settings: Sequence[Tuple[str, str]],
    k_shot=10,
    methods: Sequence[str] = STAGE3_COMPARISON_METHODS,
) -> List[List[str]]:
    """part_3.pdf's main comparison, as row data: header first, then one row
    per method per dataset.

    Built as data rather than as Markdown so the Markdown table and the PDF
    table render from one source and cannot disagree.

    One row per method rather than one column per method, so that the
    baseline is a labelled row of its own and the change column can name
    what it is relative to - neither of which a column headed `linear_probe`
    manages. The change is each run's *paired* delta against its own Stage 1
    checkpoint, averaged, with its own standard deviation; it is not the
    difference of the two accuracy columns.

    Settings or methods absent from `summaries` render as "n/a" rather than
    raising, so a partially-completed sweep still produces a readable table.

    Args:
        summaries: aggregated summaries from `aggregate_results`.
        settings: the (dataset, encoder) pairs to show.
        k_shot: the training-set size Stage 3 used.
        methods: which conditions to show, in order. Defaults to the three
            part_3.pdf's main comparison names; the optional extension passes
            its own set.

    Returns:
        A list of rows, each a list of cell strings; the first is the header.
    """
    by_setting = {
        (s["dataset"], s["encoder"], s["method"]): s
        for s in summaries
        if s["k_shot"] == k_shot
    }

    rows = [list(STAGE3_COMPARISON_HEADER)]
    for dataset, encoder in settings:
        for method in methods:
            label = STAGE3_METHOD_LABELS.get(method, method)
            summary = by_setting.get((dataset, encoder, method))
            if summary is None:
                rows.append([dataset, encoder, label, "n/a", "n/a"])
                continue

            accuracy = _format_percentage(
                summary["mean_test_accuracy"], summary["std_test_accuracy"]
            )
            mean_delta = summary.get("mean_delta_accuracy")
            change = (
                "-"
                if mean_delta is None
                else _format_percentage(
                    mean_delta, summary.get("std_delta_accuracy"), signed=True
                )
            )
            rows.append([dataset, encoder, label, accuracy, change])

    return rows


def rows_to_markdown(rows: List[List[str]]) -> str:
    """Render row data (header first) as a GitHub-flavoured Markdown table."""
    header, *body = rows
    lines = ["| " + " | ".join(header) + " |", "|---" * len(header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines) + "\n"


def format_stage3_comparison_table(
    summaries: List[dict],
    settings: Sequence[Tuple[str, str]],
    k_shot=10,
    methods: Sequence[str] = STAGE3_COMPARISON_METHODS,
) -> str:
    """Render part_3.pdf's main comparison as Markdown.

    See `stage3_comparison_rows` for the layout and why.
    """
    return rows_to_markdown(stage3_comparison_rows(summaries, settings, k_shot, methods))


def format_stage3_diagnostics_table(stage3_summaries: List[dict]) -> str:
    """Render validation accuracy for the Stage 3 runs.

    The validation split is the one checkpoints and hyperparameters were
    selected on, so this is the view that selection saw; the main comparison
    reports the held-out test split. "Baseline val" is the untrained
    pipeline's validation accuracy, which equals the Stage 1 linear probe's
    because the flow starts as the exact identity.
    """
    if not stage3_summaries:
        return "_No Stage 3 runs._"

    header = "| Dataset | Method | Runs | Baseline val | Best val | Val delta |\n"
    header += "|---" * 6 + "|\n"

    rows = []
    for summary in stage3_summaries:
        rows.append(
            f"| {summary['dataset']} | {summary['method']} | {summary['num_runs']} "
            f"| {summary['mean_initial_val_accuracy'] * 100:.2f}% "
            f"| {summary['mean_best_val_accuracy'] * 100:.2f}% "
            f"| {_format_percentage(summary['mean_val_delta'], summary['std_val_delta'], signed=True)} |"
        )

    return header + "\n".join(rows) + "\n"


__all__ = [
    "METHOD_DISPLAY_ORDER",
    "format_accuracy_table",
    "format_flow_matching_comparison_table",
    "format_stage3_comparison_table",
    "format_stage3_diagnostics_table",
    "rows_to_markdown",
    "stage3_comparison_rows",
]
