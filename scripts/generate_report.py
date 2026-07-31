"""Aggregate all experiment results, generate an accuracy table and one
accuracy-vs-training-size plot per (dataset, encoder) pair, and write it all
into RESULTS.md at the project root.

Usage:
    python scripts/generate_report.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.datasets import get_class_names, load_dataset_splits
from src.evaluation.aggregation import aggregate_results, load_all_results
from src.evaluation.confusion_matrix import compute_confusion_matrix, row_normalize
from src.evaluation.pdf_report import generate_pdf_report
from src.evaluation.predictions import get_linear_probe_test_predictions
from src.evaluation.tables import format_accuracy_table
from src.visualization.accuracy_vs_shot import plot_accuracy_vs_shot
from src.visualization.confusion_matrix_plot import plot_confusion_matrix
from src.visualization.loss_curves import load_history, plot_loss_curve

# Per-project convention: seed 0's 10-shot run is "the representative run"
# stage_1.pdf asks for when plotting linear-probe loss curves.
REPRESENTATIVE_LOSS_CURVE_K_SHOT = 10
REPRESENTATIVE_LOSS_CURVE_SEED = 0

# Per-project convention: one representative linear-probe setting per
# dataset for the required confusion matrix - full training data, seed 0,
# and (for DTD) the encoder this project pairs with DTD specifically.
REPRESENTATIVE_CONFUSION_MATRIX_SETTINGS = {
    "dtd": {"encoder": "dinov2_vits14", "k_shot": "full", "seed": 0},
    "flowers102": {"encoder": "resnet18", "k_shot": "full", "seed": 0},
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs"
    reports_dir = PROJECT_ROOT / "reports"
    figures_dir = reports_dir / "figures"

    records = load_all_results(output_dir)
    print(f"Loaded {len(records)} individual run results from {output_dir}")
    if not records:
        raise RuntimeError(
            f"No result.json files found under {output_dir}. Run scripts/run_linear_probe.py "
            "and/or scripts/run_prototype.py first."
        )

    summaries = aggregate_results(records)
    print(f"Aggregated into {len(summaries)} (dataset, encoder, method, k_shot) settings")

    table_markdown = format_accuracy_table(summaries)

    dataset_encoder_pairs = sorted({(s["dataset"], s["encoder"]) for s in summaries})
    figure_paths = []  # (dataset, encoder, absolute_path) - for the PDF
    for dataset, encoder in dataset_encoder_pairs:
        figure_path = figures_dir / f"accuracy_vs_shot_{dataset}_{encoder}.png"
        plot_accuracy_vs_shot(summaries, dataset, encoder, figure_path)
        figure_paths.append((dataset, encoder, figure_path))
        print(f"Saved {figure_path}")

    linear_probe_pairs = sorted(
        {(s["dataset"], s["encoder"]) for s in summaries if s["method"] == "linear_probe"}
    )
    loss_curve_figure_paths = []  # (dataset, encoder, absolute_path)
    for dataset, encoder in linear_probe_pairs:
        history_path = (
            output_dir / "linear_probe" / dataset / encoder
            / f"k{REPRESENTATIVE_LOSS_CURVE_K_SHOT}" / f"seed{REPRESENTATIVE_LOSS_CURVE_SEED}"
            / "history.json"
        )
        if not history_path.exists():
            print(f"  Skipping loss curve for {dataset}/{encoder}: no run found at {history_path}")
            continue
        history = load_history(history_path)
        figure_path = figures_dir / f"loss_curve_{dataset}_{encoder}.png"
        plot_loss_curve(history, dataset, encoder, figure_path)
        loss_curve_figure_paths.append((dataset, encoder, figure_path))
        print(f"Saved {figure_path}")

    cache_dir = PROJECT_ROOT / "cache"
    data_dir = PROJECT_ROOT / "data"
    confusion_matrix_figure_paths = []  # (dataset, encoder, absolute_path)
    for dataset, setting in REPRESENTATIVE_CONFUSION_MATRIX_SETTINGS.items():
        encoder, k_shot, seed = setting["encoder"], setting["k_shot"], setting["seed"]
        try:
            true_labels, predicted_labels = get_linear_probe_test_predictions(
                cache_dir, output_dir, dataset, encoder, k_shot, seed, device=torch.device("cpu"),
            )
        except FileNotFoundError as error:
            print(f"  Skipping confusion matrix for {dataset}: {error}")
            continue

        class_names = get_class_names(
            load_dataset_splits(dataset, data_dir, download=False)["train"]
        )
        raw_matrix = compute_confusion_matrix(true_labels, predicted_labels, len(class_names))
        normalized_matrix = row_normalize(raw_matrix)

        figure_path = figures_dir / f"confusion_matrix_{dataset}_{encoder}.png"
        title = f"{dataset} / {encoder} (linear probe, full, seed {seed})"
        plot_confusion_matrix(normalized_matrix, class_names, title, figure_path)
        confusion_matrix_figure_paths.append((dataset, encoder, figure_path))
        print(f"Saved {figure_path}")

    report_lines = [
        "# Stage 1 Results\n",
        "Auto-generated by `scripts/generate_report.py` from `outputs/`. "
        "Re-run that script after new experiments to refresh this file.\n",
        "## Accuracy table\n",
        table_markdown,
        "## Accuracy vs. training-set size\n",
    ]
    for dataset, encoder, figure_path in figure_paths:
        relative_path = figure_path.relative_to(PROJECT_ROOT)
        report_lines.append(f"### {dataset} / {encoder}\n")
        report_lines.append(f"![{dataset} {encoder} accuracy vs shot]({relative_path.as_posix()})\n")

    if loss_curve_figure_paths:
        report_lines.append("## Training / validation loss (10-shot, seed 0)\n")
        for dataset, encoder, figure_path in loss_curve_figure_paths:
            relative_path = figure_path.relative_to(PROJECT_ROOT)
            report_lines.append(f"### {dataset} / {encoder}\n")
            report_lines.append(f"![{dataset} {encoder} loss curve]({relative_path.as_posix()})\n")

    if confusion_matrix_figure_paths:
        report_lines.append("## Confusion matrices (row-normalized)\n")
        for dataset, encoder, figure_path in confusion_matrix_figure_paths:
            relative_path = figure_path.relative_to(PROJECT_ROOT)
            report_lines.append(f"### {dataset} / {encoder}\n")
            report_lines.append(f"![{dataset} {encoder} confusion matrix]({relative_path.as_posix()})\n")

    report_path = PROJECT_ROOT / "RESULTS.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Saved report to {report_path}")

    pdf_path = PROJECT_ROOT / "RESULTS.pdf"
    generate_pdf_report(
        summaries, figure_paths, pdf_path, loss_curve_figure_paths, confusion_matrix_figure_paths
    )
    print(f"Saved report to {pdf_path}")


if __name__ == "__main__":
    main()
