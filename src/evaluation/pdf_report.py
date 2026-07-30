"""Render aggregated results as a PDF report (accuracy table + plots).

Built directly from the same summary data used for RESULTS.md, rather than
converting the Markdown file itself - this avoids adding a markdown-to-PDF
dependency (e.g. pandoc/weasyprint) for what is fundamentally the same
table and images, and keeps this project's dependencies minimal.
"""

from pathlib import Path
from typing import List, Tuple, Union

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _format_accuracy_cell(summary: dict) -> str:
    if summary["std_test_accuracy"] is None:
        return f"{summary['mean_test_accuracy'] * 100:.2f}%"
    return f"{summary['mean_test_accuracy'] * 100:.2f}% +/- {summary['std_test_accuracy'] * 100:.2f}%"


def generate_pdf_report(
    summaries: List[dict],
    figure_paths: List[Tuple[str, str, Union[str, Path]]],
    save_path: Union[str, Path],
) -> None:
    """Build a single-file PDF report: accuracy table, then one
    accuracy-vs-training-size plot per (dataset, encoder) pair.

    Args:
        summaries: aggregated results from aggregation.aggregate_results().
        figure_paths: (dataset, encoder, png_path) tuples, one per plot to embed.
        save_path: where to save the PDF.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    story = [Paragraph("Stage 1 Results", styles["Title"]), Spacer(1, 12)]

    table_data = [["Dataset", "Encoder", "Method", "K-shot", "Runs", "Test Accuracy"]]
    for summary in summaries:
        table_data.append(
            [
                summary["dataset"],
                summary["encoder"],
                summary["method"],
                str(summary["k_shot"]),
                str(summary["num_runs"]),
                _format_accuracy_cell(summary),
            ]
        )

    table = Table(table_data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 24))

    story.append(Paragraph("Accuracy vs. training-set size", styles["Heading1"]))
    for dataset, encoder, figure_path in figure_paths:
        story.append(Paragraph(f"{dataset} / {encoder}", styles["Heading2"]))
        story.append(Image(str(figure_path), width=5 * inch, height=3.75 * inch))
        story.append(Spacer(1, 18))

    SimpleDocTemplate(str(save_path), pagesize=letter).build(story)
