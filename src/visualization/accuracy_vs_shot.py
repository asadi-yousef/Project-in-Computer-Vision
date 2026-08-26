"""Plot test accuracy vs. training-set size (5-shot / 10-shot / full), with
error bars from the sample standard deviation across seeds.

Stage 2 puts up to five series on one panel - the prototype baseline plus
standard and rolled-out FM at T in {4, 12} - so the encoding is deliberately
two-dimensional: colour identifies the method, line style and marker
identify T. That keeps "which method" and "which T" separately readable
instead of asking the eye to distinguish five arbitrary colours.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")  # renders to file without needing a display
import matplotlib.pyplot as plt

from src.evaluation.aggregation import METHOD_DISPLAY_ORDER, method_label
from src.visualization.style import euler_style, method_color

_K_SHOT_X_POSITIONS = {5: 0, 10: 1, "full": 2}
_K_SHOT_TICK_LABELS = ["5-shot", "10-shot", "full"]

def _series_sort_key(method: str, num_euler_steps: Optional[int]):
    """Order legend entries the same way the tables order rows."""
    method_index = (
        METHOD_DISPLAY_ORDER.index(method) if method in METHOD_DISPLAY_ORDER else len(METHOD_DISPLAY_ORDER)
    )
    return (method_index, -1 if num_euler_steps is None else num_euler_steps)


def plot_accuracy_vs_shot(
    summaries: List[dict],
    dataset: str,
    encoder: str,
    save_path: Union[str, Path],
    methods: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
) -> None:
    """Plot mean (+/- std) test accuracy vs. k_shot for a single
    (dataset, encoder) pair, one line per (method, T), saved as a PNG.

    Args:
        summaries: aggregated results from aggregation.aggregate_results().
        dataset: which dataset's rows to plot (e.g. "dtd").
        encoder: which encoder's rows to plot (e.g. "resnet18").
        save_path: where to save the PNG.
        methods: restrict to these methods, in case the caller wants a
            focused comparison (e.g. the prototype baseline against the FM
            variants, leaving the linear probe out). None plots every method
            present.
        title: override the default plot title.

    Raises:
        ValueError: if no summaries match the requested dataset/encoder (and
            `methods` filter, when given).
    """
    relevant_summaries = [
        s for s in summaries if s["dataset"] == dataset and s["encoder"] == encoder
    ]
    if methods is not None:
        relevant_summaries = [s for s in relevant_summaries if s["method"] in methods]
    if not relevant_summaries:
        raise ValueError(
            f"No results found for dataset={dataset!r}, encoder={encoder!r}"
            + (f", methods={list(methods)!r}" if methods is not None else "")
        )

    series_keys = sorted(
        {(s["method"], s.get("num_euler_steps")) for s in relevant_summaries},
        key=lambda key: _series_sort_key(*key),
    )

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    for method, num_euler_steps in series_keys:
        series_summaries = sorted(
            (
                s for s in relevant_summaries
                if s["method"] == method and s.get("num_euler_steps") == num_euler_steps
            ),
            key=lambda s: _K_SHOT_X_POSITIONS[s["k_shot"]],
        )
        x_positions = [_K_SHOT_X_POSITIONS[s["k_shot"]] for s in series_summaries]
        mean_accuracies = [s["mean_test_accuracy"] * 100 for s in series_summaries]
        # A single-run setting has no measured variance; drawing a zero-length
        # bar is the honest rendering, since there is nothing to show.
        std_errors = [(s["std_test_accuracy"] or 0.0) * 100 for s in series_summaries]

        linestyle, marker = euler_style(num_euler_steps)
        ax.errorbar(
            x_positions,
            mean_accuracies,
            yerr=std_errors,
            color=method_color(method),
            linestyle=linestyle,
            marker=marker,
            capsize=4,
            linewidth=2.0 if method == "prototype" else 1.5,
            label=method_label(method, num_euler_steps),
        )

    ax.set_xticks(list(_K_SHOT_X_POSITIONS.values()))
    ax.set_xticklabels(_K_SHOT_TICK_LABELS)
    ax.set_xlim(-0.2, 2.2)
    ax.set_xlabel("Training-set size")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(title or f"{dataset} / {encoder}: accuracy vs. training-set size")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_delta_vs_shot(
    summaries: List[dict],
    dataset: str,
    encoder: str,
    save_path: Union[str, Path],
    title: Optional[str] = None,
) -> None:
    """Plot the change in accuracy relative to the prototype baseline.

    The companion to `plot_accuracy_vs_shot`: because every FM series tracks
    the baseline's overall shape, absolute-accuracy curves mostly show how
    the baseline itself varies with K, and the FM effect is a small vertical
    offset. Plotting the paired delta directly puts that effect on its own
    axis, against a zero reference line.

    Only settings that carry a delta (the FM methods) are plotted.

    Args:
        summaries: aggregated results from aggregation.aggregate_results().
        dataset, encoder: which pair to plot.
        save_path: where to save the PNG.
        title: override the default plot title.

    Raises:
        ValueError: if no flow-matching summaries match the request.
    """
    relevant_summaries = [
        s for s in summaries
        if s["dataset"] == dataset
        and s["encoder"] == encoder
        and s.get("mean_delta_accuracy") is not None
    ]
    if not relevant_summaries:
        raise ValueError(
            f"No flow-matching results found for dataset={dataset!r}, encoder={encoder!r}"
        )

    series_keys = sorted(
        {(s["method"], s.get("num_euler_steps")) for s in relevant_summaries},
        key=lambda key: _series_sort_key(*key),
    )

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.axhline(0.0, color="black", linewidth=1.2, label="prototype baseline")

    for method, num_euler_steps in series_keys:
        series_summaries = sorted(
            (
                s for s in relevant_summaries
                if s["method"] == method and s.get("num_euler_steps") == num_euler_steps
            ),
            key=lambda s: _K_SHOT_X_POSITIONS[s["k_shot"]],
        )
        x_positions = [_K_SHOT_X_POSITIONS[s["k_shot"]] for s in series_summaries]
        mean_deltas = [s["mean_delta_accuracy"] * 100 for s in series_summaries]
        std_errors = [(s.get("std_delta_accuracy") or 0.0) * 100 for s in series_summaries]

        linestyle, marker = euler_style(num_euler_steps)
        ax.errorbar(
            x_positions,
            mean_deltas,
            yerr=std_errors,
            color=method_color(method),
            linestyle=linestyle,
            marker=marker,
            capsize=4,
            label=method_label(method, num_euler_steps),
        )

    ax.set_xticks(list(_K_SHOT_X_POSITIONS.values()))
    ax.set_xticklabels(_K_SHOT_TICK_LABELS)
    ax.set_xlim(-0.2, 2.2)
    ax.set_xlabel("Training-set size")
    ax.set_ylabel("Change in test accuracy vs. baseline (percentage points)")
    ax.set_title(title or f"{dataset} / {encoder}: FM effect relative to the prototype baseline")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
