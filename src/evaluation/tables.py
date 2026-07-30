"""Render aggregated results (see aggregation.py) as a Markdown accuracy table."""

from typing import List


def format_accuracy_table(summaries: List[dict]) -> str:
    """Render aggregated summaries as a GitHub-flavored Markdown table.

    Standard deviation is shown as "mean +/- std"; for single-run settings
    (std is None) only the mean is shown, since a "+/- 0" would misleadingly
    imply variance was measured.
    """
    header = "| Dataset | Encoder | Method | K-shot | Runs | Test Accuracy |\n"
    header += "|---|---|---|---|---|---|\n"

    rows = []
    for summary in summaries:
        if summary["std_test_accuracy"] is None:
            accuracy_text = f"{summary['mean_test_accuracy'] * 100:.2f}%"
        else:
            accuracy_text = (
                f"{summary['mean_test_accuracy'] * 100:.2f}% "
                f"+/- {summary['std_test_accuracy'] * 100:.2f}%"
            )
        rows.append(
            f"| {summary['dataset']} | {summary['encoder']} | {summary['method']} | "
            f"{summary['k_shot']} | {summary['num_runs']} | {accuracy_text} |"
        )

    return header + "\n".join(rows) + "\n"
