"""Training and validation curves for the Stage 3 strategies (part_3.pdf).

part_3.pdf: "Training behavior. Show representative training and validation
curves for both Stage 3 methods."

Stage 3 needs its own figure rather than reusing `plot_loss_curve` for one
reason: **the two strategies minimize different quantities.** Strategy 1's
training loss is a classification cross-entropy plus a displacement penalty;
Strategy 2's is a squared velocity error against a target that itself moves
during training. Their magnitudes differ by orders of magnitude and neither
says anything about the other.

What *is* comparable across both strategies, and against the Stage 1
baseline, is the pipeline's validation cross-entropy and its accuracy - both
measured identically by `evaluate_pipeline`, on the same frozen classifier.
So the figure is laid out as:

    row 1  each strategy's own training objective (left axis, own units)
           with the pipeline's validation cross-entropy on a twin axis
    row 2  training and validation accuracy, against two references:
           a horizontal line at the untrained pipeline's accuracy, and a
           vertical line at the epoch validation selected

The horizontal reference line is the point of the whole figure. Because the
flow is initialized to the exact identity, that line *is* the Stage 1 linear
probe's accuracy - so the vertical distance between it and the validation
curve is precisely what Stage 3 contributed, readable directly off the plot.
"""

import dataclasses
from pathlib import Path
from typing import List, Sequence, Union

import matplotlib

matplotlib.use("Agg")  # renders to file without needing a display
import matplotlib.pyplot as plt

from src.visualization.style import method_color

# What each strategy's `train_loss` column actually measures. Spelled out on
# the axis because the two are not comparable and a reader should not have to
# remember which panel is which.
TRAIN_OBJECTIVE_LABELS = {
    "fm_cls_rolled": r"$CE(W\hat{z}_T + b,\, y)$ + penalty",
    "fm_cls_guided": r"$\|v_\theta(z_t,t) - (\hat{z}' - z)\|^2$",
}
DEFAULT_TRAIN_OBJECTIVE_LABEL = "training objective"

_TRAIN_STYLE = "-"
_VAL_STYLE = "--"
_VAL_LOSS_COLOR = "tab:gray"


@dataclasses.dataclass
class Stage3CurveRun:
    """One run's curves, plus the two references the accuracy panel needs.

    Attributes:
        method: the Stage 3 method, used for the panel title and colour.
        history: per-epoch dicts as written to history.json.
        initial_val_accuracy: the untrained pipeline's validation accuracy.
            Equal to the frozen Stage 1 probe's own validation accuracy,
            since the flow starts as the exact identity.
        best_epoch: the epoch validation selected, marked on the plot.
    """

    method: str
    history: List[dict]
    initial_val_accuracy: float
    best_epoch: int


def plot_stage3_curves(
    runs: Sequence[Stage3CurveRun],
    dataset: str,
    encoder: str,
    k_shot,
    seed: int,
    save_path: Union[str, Path],
) -> None:
    """Plot representative training curves for the Stage 3 strategies.

    Args:
        runs: one entry per strategy, drawn as a column in the given order.
        dataset, encoder, k_shot, seed: identify the runs, for the title.
        save_path: where to save the PNG.

    Raises:
        ValueError: if `runs` is empty, or any run has an empty history.
    """
    if not runs:
        raise ValueError("runs is empty; nothing to plot")
    for run in runs:
        if not run.history:
            raise ValueError(f"history for {run.method!r} is empty; nothing to plot")

    fig, axes = plt.subplots(
        2, len(runs), figsize=(5.8 * len(runs), 7.5), dpi=150, squeeze=False
    )

    for column, run in enumerate(runs):
        color = method_color(run.method)
        epochs = [entry["epoch"] for entry in run.history]

        # --- Row 1: the strategy's own objective, plus validation CE ---
        loss_axis = axes[0][column]
        loss_axis.plot(
            epochs, [entry["train_loss"] for entry in run.history],
            linestyle=_TRAIN_STYLE, color=color, label="train objective",
        )
        loss_axis.set_ylabel(
            TRAIN_OBJECTIVE_LABELS.get(run.method, DEFAULT_TRAIN_OBJECTIVE_LABEL),
            color=color,
        )
        loss_axis.tick_params(axis="y", labelcolor=color)

        # A twin axis, not a shared one: validation cross-entropy is in
        # different units from Strategy 2's velocity loss entirely.
        val_axis = loss_axis.twinx()
        val_axis.plot(
            epochs, [entry["val_loss"] for entry in run.history],
            linestyle=_VAL_STYLE, color=_VAL_LOSS_COLOR, label="validation CE",
        )
        val_axis.set_ylabel("validation cross-entropy", color=_VAL_LOSS_COLOR)
        val_axis.tick_params(axis="y", labelcolor=_VAL_LOSS_COLOR)

        loss_axis.set_title(f"{run.method}: training objective", fontsize=10)
        loss_axis.set_xlabel("Epoch")
        loss_axis.grid(alpha=0.3)

        handles = loss_axis.get_lines() + val_axis.get_lines()
        loss_axis.legend(handles, [line.get_label() for line in handles], fontsize=8)

        # --- Row 2: accuracy, against the baseline and the selected epoch ---
        accuracy_axis = axes[1][column]
        accuracy_axis.plot(
            epochs, [entry["train_accuracy"] * 100 for entry in run.history],
            linestyle=_TRAIN_STYLE, color=color, label="train accuracy",
        )
        accuracy_axis.plot(
            epochs, [entry["val_accuracy"] * 100 for entry in run.history],
            linestyle=_VAL_STYLE, color=color, label="validation accuracy",
        )
        accuracy_axis.axhline(
            run.initial_val_accuracy * 100,
            color="black", linestyle=":", linewidth=1.2,
            label="Stage 1 probe (untrained flow)",
        )
        accuracy_axis.axvline(
            run.best_epoch,
            color="tab:green", linestyle="-.", linewidth=1.0,
            label=f"selected epoch ({run.best_epoch})",
        )

        accuracy_axis.set_title(f"{run.method}: accuracy", fontsize=10)
        accuracy_axis.set_xlabel("Epoch")
        accuracy_axis.set_ylabel("Accuracy (%)")
        accuracy_axis.legend(fontsize=8, loc="lower right")
        accuracy_axis.grid(alpha=0.3)

    fig.suptitle(
        f"{dataset} / {encoder}: Stage 3 training ({k_shot}-shot, seed {seed})"
        "\nthe two training objectives are different quantities and are not comparable to each other;"
        " the accuracy panels are",
        fontsize=10,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
