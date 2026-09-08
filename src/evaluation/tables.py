"""Render aggregated results (see aggregation.py) as Markdown tables."""

from typing import List, Optional, Sequence, Tuple

from src.evaluation.aggregation import (
    METHOD_DISPLAY_ORDER,
    STAGE3_COMPARISON_METHODS,
    method_label,
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

    Columns cover both stages: the Stage 1 methods have no Euler-step count
    and no baseline to compare against, so their T and delta cells show "-".

    The delta column is the mean of each run's *paired* delta against its own
    baseline (see `aggregate_results`), not a difference of column means.
    """
    header = "| Dataset | Encoder | Method | T | K-shot | Runs | Test Accuracy | Delta vs baseline |\n"
    header += "|---|---|---|---|---|---|---|---|\n"

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

        rows.append(
            f"| {summary['dataset']} | {summary['encoder']} | {summary['method']} | "
            f"{euler_text} | {summary['k_shot']} | {summary['num_runs']} | "
            f"{accuracy_text} | {delta_text} |"
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


def format_stage3_comparison_table(
    summaries: List[dict],
    settings: Sequence[Tuple[str, str]],
    k_shot=10,
    methods: Sequence[str] = STAGE3_COMPARISON_METHODS,
) -> str:
    """Render part_3.pdf's main comparison: one row per dataset, three methods.

    part_3.pdf asks to compare, for each of the two datasets, the Stage 1
    linear probe against both Stage 3 methods, and to "also report the change
    relative to the corresponding linear-probe baseline". The general
    accuracy table spreads those three conditions across three rows per
    dataset; this puts them side by side, baseline first.

    Every method within a row uses the same encoder, training subset and
    pretrained classifier, as part_3.pdf requires - that is a property of how
    the runs were produced, not of this renderer, but it is what makes the
    row comparable at all.

    Settings absent from `summaries` render as "n/a" rather than raising, so
    a partially-completed sweep still produces a readable table.

    Args:
        summaries: aggregated summaries from `aggregate_results`.
        settings: the (dataset, encoder) pairs to show, one row each.
        k_shot: the training-set size Stage 3 used.
        methods: the columns, in order. Defaults to the three conditions
            part_3.pdf's main comparison names; the optional extension passes
            its own set.
    """
    by_setting = {
        (s["dataset"], s["encoder"], s["method"]): s
        for s in summaries
        if s["k_shot"] == k_shot
    }

    header = "| Dataset | Encoder | " + " | ".join(methods) + " |\n"
    header += "|---" * (len(methods) + 2) + "|\n"

    rows = []
    for dataset, encoder in settings:
        cells = []
        for method in methods:
            summary = by_setting.get((dataset, encoder, method))
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
        rows.append(f"| {dataset} | {encoder} | " + " | ".join(cells) + " |")

    return header + "\n".join(rows) + "\n"


def format_stage3_diagnostics_table(stage3_summaries: List[dict]) -> str:
    """Render the Stage 3-only measurements from `summarize_stage3_runs`.

    Supporting evidence for the observations rather than a required
    deliverable. The displacement column is the one to read alongside the
    accuracy: a flow that gains accuracy while moving features a small
    fraction of their own norm is doing something different from one that
    moves them further than they are long.
    """
    if not stage3_summaries:
        return "_No Stage 3 runs._"

    header = (
        "| Dataset | Method | Runs | Baseline val | Best val | Val delta "
        "| Selected epochs | Mean displacement |\n"
    )
    header += "|---" * 8 + "|\n"

    rows = []
    for summary in stage3_summaries:
        rows.append(
            f"| {summary['dataset']} | {summary['method']} | {summary['num_runs']} "
            f"| {summary['mean_initial_val_accuracy'] * 100:.2f}% "
            f"| {summary['mean_best_val_accuracy'] * 100:.2f}% "
            f"| {_format_percentage(summary['mean_val_delta'], summary['std_val_delta'], signed=True)} "
            f"| {summary['best_epochs']} "
            f"| {summary['mean_displacement']:.2f} |"
        )

    return header + "\n".join(rows) + "\n"


__all__ = [
    "METHOD_DISPLAY_ORDER",
    "format_accuracy_table",
    "format_flow_matching_comparison_table",
    "format_stage3_comparison_table",
    "format_stage3_diagnostics_table",
]
