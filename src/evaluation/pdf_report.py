"""Render aggregated results as a PDF report (accuracy table + plots).

Built directly from the same summary data used for RESULTS.md, rather than
converting the Markdown file itself - this avoids adding a markdown-to-PDF
dependency (e.g. pandoc/weasyprint) for what is fundamentally the same
table and images, and keeps this project's dependencies minimal.

Pages are landscape. The Stage 2 figures are wide multi-panel comparisons
(the per-step and feature-space figures are about 3.4:1), and every figure is
scaled to fit the page while preserving its own aspect ratio, so nothing is
stretched. Each figure gets its own page: at these aspect ratios two figures
per portrait page left them too small to read, which defeats the point of
including them.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PAGE_SIZE = landscape(letter)
MARGIN = 0.5 * inch
# Room left under the figure for its heading and the page margins.
HEADING_ALLOWANCE = 1.1 * inch


def _format_accuracy_cell(summary: dict) -> str:
    if summary["std_test_accuracy"] is None:
        return f"{summary['mean_test_accuracy'] * 100:.2f}%"
    return f"{summary['mean_test_accuracy'] * 100:.2f}% +/- {summary['std_test_accuracy'] * 100:.2f}%"


def _format_delta_cell(summary: dict) -> str:
    """Signed change against the run's own paired baseline, or "-" for the
    Stage 1 methods, which have no baseline to compare against."""
    mean_delta = summary.get("mean_delta_accuracy")
    if mean_delta is None:
        return "-"
    text = f"{mean_delta * 100:+.2f}%"
    std_delta = summary.get("std_delta_accuracy")
    if std_delta is not None:
        text += f" +/- {std_delta * 100:.2f}%"
    return text


def _fit_image(figure_path: Union[str, Path], max_width: float, max_height: float) -> Image:
    """Scale a PNG to fit the given box without distorting it.

    The previous version forced every figure into a fixed 4:3 box, which
    squashed the wide Stage 2 comparisons by up to 2.6x and stretched the
    square confusion matrices the other way. Reading the real pixel
    dimensions and preserving the ratio is what stops that.
    """
    pixel_width, pixel_height = PILImage.open(figure_path).size
    scale = min(max_width / pixel_width, max_height / pixel_height)
    return Image(str(figure_path), width=pixel_width * scale, height=pixel_height * scale)


def _add_figure_section(
    story: list, styles, heading: str, figure_paths, page_size=PAGE_SIZE
) -> None:
    """Add one page per figure, each headed and scaled to fit."""
    if not figure_paths:
        return
    max_width = page_size[0] - 2 * MARGIN
    max_height = page_size[1] - 2 * MARGIN - HEADING_ALLOWANCE

    for dataset, encoder, figure_path in figure_paths:
        image = _fit_image(figure_path, max_width, max_height)
        story.append(PageBreak())
        story.append(Paragraph(heading, styles["Heading2"]))
        story.append(Paragraph(f"{dataset} / {encoder}", styles["Heading3"]))
        story.append(Spacer(1, 6))
        # A wide figure is limited by page width, so it never fills the page
        # height. Centre it vertically rather than stranding it at the top
        # above a half page of white space.
        leftover = max_height - image.drawHeight
        if leftover > 0:
            story.append(Spacer(1, leftover / 2))
        story.append(image)


def generate_pdf_report(
    summaries: List[dict],
    figure_paths: List[Tuple[str, str, Union[str, Path]]],
    save_path: Union[str, Path],
    loss_curve_figure_paths: List[Tuple[str, str, Union[str, Path]]] = None,
    confusion_matrix_figure_paths: List[Tuple[str, str, Union[str, Path]]] = None,
    feature_space_figure_paths: List[Tuple[str, str, Union[str, Path]]] = None,
    extra_figure_sections: List[Tuple[str, List[Tuple[str, str, Union[str, Path]]]]] = None,
    title: str = "Stage 1 Results",
    summary_lines: Optional[List[str]] = None,
) -> None:
    """Build a single-file PDF report: accuracy table, accuracy-vs-shot
    plots, and (optionally) loss-curve, confusion-matrix, and feature-space
    plots.

    Args:
        summaries: aggregated results from aggregation.aggregate_results().
        figure_paths: (dataset, encoder, png_path) tuples, accuracy-vs-shot plots.
        save_path: where to save the PDF.
        loss_curve_figure_paths: (dataset, encoder, png_path) tuples, loss-curve
            plots; section omitted entirely if not provided.
        confusion_matrix_figure_paths: (dataset, encoder, png_path) tuples,
            confusion-matrix plots; section omitted entirely if not provided.
        feature_space_figure_paths: (dataset, encoder, png_path) tuples,
            feature-space plots; section omitted entirely if not provided.
        extra_figure_sections: (heading, figure_paths) pairs appended after the
            sections above. Stage 2 supplies its figures this way rather than
            adding a named parameter per figure type.
        title: document title.
        summary_lines: optional plain-text bullet points placed on the first
            page, so the headline findings are visible before the tables.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]

    for line in summary_lines or []:
        story.append(Paragraph(line, styles["BodyText"]))
    if summary_lines:
        story.append(Spacer(1, 12))

    table_data = [
        ["Dataset", "Encoder", "Method", "T", "K-shot", "Runs", "Test Accuracy", "Delta"]
    ]
    for summary in summaries:
        num_euler_steps = summary.get("num_euler_steps")
        table_data.append(
            [
                summary["dataset"],
                summary["encoder"],
                summary["method"],
                "-" if num_euler_steps is None else str(num_euler_steps),
                str(summary["k_shot"]),
                str(summary["num_runs"]),
                _format_accuracy_cell(summary),
                _format_delta_cell(summary),
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

    _add_figure_section(story, styles, "Accuracy vs. training-set size", figure_paths)
    _add_figure_section(
        story, styles, "Training / validation loss (10-shot, seed 0)", loss_curve_figure_paths or []
    )
    _add_figure_section(
        story, styles, "Confusion matrices (row-normalized)", confusion_matrix_figure_paths or []
    )
    _add_figure_section(
        story, styles, "Feature-space visualizations (t-SNE)", feature_space_figure_paths or []
    )
    for heading, section_figure_paths in extra_figure_sections or []:
        _add_figure_section(story, styles, heading, section_figure_paths)

    SimpleDocTemplate(
        str(save_path),
        pagesize=PAGE_SIZE,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
    ).build(story)
