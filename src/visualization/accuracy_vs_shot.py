"""Plot test accuracy vs. training-set size (5-shot / 10-shot / full), with
error bars from the sample standard deviation across seeds.
"""

from pathlib import Path
from typing import List, Union

import matplotlib

matplotlib.use("Agg")  # renders to file without needing a display
import matplotlib.pyplot as plt

_K_SHOT_X_POSITIONS = {5: 0, 10: 1, "full": 2}
_K_SHOT_TICK_LABELS = ["5-shot", "10-shot", "full"]


def plot_accuracy_vs_shot(
    summaries: List[dict], dataset: str, encoder: str, save_path: Union[str, Path]
) -> None:
    """Plot mean (+/- std) test accuracy vs. k_shot, one line per method, for
    a single (dataset, encoder) pair, saved as a high-resolution PNG.

    Args:
        summaries: aggregated results from aggregation.aggregate_results().
        dataset: which dataset's rows to plot (e.g. "dtd").
        encoder: which encoder's rows to plot (e.g. "resnet18").
        save_path: where to save the PNG.
    """
    relevant_summaries = [s for s in summaries if s["dataset"] == dataset and s["encoder"] == encoder]
    if not relevant_summaries:
        raise ValueError(f"No results found for dataset={dataset!r}, encoder={encoder!r}")

    methods = sorted({s["method"] for s in relevant_summaries})

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    for method in methods:
        method_summaries = sorted(
            (s for s in relevant_summaries if s["method"] == method),
            key=lambda s: _K_SHOT_X_POSITIONS[s["k_shot"]],
        )
        x_positions = [_K_SHOT_X_POSITIONS[s["k_shot"]] for s in method_summaries]
        mean_accuracies = [s["mean_test_accuracy"] * 100 for s in method_summaries]
        std_errors = [(s["std_test_accuracy"] or 0.0) * 100 for s in method_summaries]
        ax.errorbar(
            x_positions, mean_accuracies, yerr=std_errors,
            marker="o", capsize=4, label=method,
        )

    ax.set_xticks(list(_K_SHOT_X_POSITIONS.values()))
    ax.set_xticklabels(_K_SHOT_TICK_LABELS)
    ax.set_xlabel("Training-set size")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"{dataset} / {encoder}: accuracy vs. training-set size")
    ax.legend()
    ax.grid(alpha=0.3)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
