"""Plot per-epoch training/validation loss curves.

Covers the linear probe (stage_1.pdf) and both flow-matching variants
(stage_2.pdf). Both specs ask for a "representative" run; this project uses
the 10-shot, seed 0 run for each dataset/encoder pair (a documented
convention, not a spec requirement of which seed).
"""

import json
from pathlib import Path
from typing import Dict, List, Union

import matplotlib

matplotlib.use("Agg")  # renders to file without needing a display
import matplotlib.pyplot as plt


def load_history(path: Union[str, Path]) -> List[dict]:
    """Load a run's per-epoch history.json (written by any experiment runner)."""
    with open(path) as f:
        return json.load(f)


def plot_loss_curve(
    history: List[dict], dataset: str, encoder: str, save_path: Union[str, Path]
) -> None:
    """Plot train/val cross-entropy loss vs. epoch for one linear-probe run.

    Args:
        history: list of per-epoch dicts (as in history.json), each with at
            least "epoch", "train_loss", "val_loss".
        dataset, encoder: used only for the plot title.
        save_path: where to save the PNG.
    """
    epochs = [entry["epoch"] for entry in history]
    train_losses = [entry["train_loss"] for entry in history]
    val_losses = [entry["val_loss"] for entry in history]

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    ax.plot(epochs, train_losses, label="train loss")
    ax.plot(epochs, val_losses, label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"{dataset} / {encoder}: linear-probe loss (10-shot, seed 0)")
    ax.legend()
    ax.grid(alpha=0.3)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# Solid = training, dashed = validation, within either panel.
_TRAIN_STYLE = "-"
_VAL_STYLE = "--"

# Colour distinguishes T in the rolled-out panel (standard FM has a single
# T-independent training curve, so it needs no such distinction).
_EULER_COLORS = {4: "tab:blue", 12: "tab:red"}
_DEFAULT_EULER_COLOR = "tab:green"


def plot_flow_matching_loss_curves(
    standard_history: List[dict],
    rolled_histories: Dict[int, List[dict]],
    dataset: str,
    encoder: str,
    k_shot,
    seed: int,
    save_path: Union[str, Path],
) -> None:
    """Plot representative training curves for both Stage 2 variants.

    Drawn as two panels with **independent y-axes**, deliberately. The two
    objectives measure different quantities in different units - standard FM
    regresses a velocity (||v_theta - u||^2) while rolled-out training
    measures a squared distance from the final transported point to the
    prototype (||z_hat_T - p||^2) - and their magnitudes differ by roughly an
    order of magnitude. Sharing one axis would invite reading rolled-out's
    smaller numbers as "better training", which they are not: the two are not
    comparable to each other, only to themselves over epochs.

    stage_2.pdf asks these curves show that training is stable and that both
    approaches reach reasonable solutions, so validation loss is drawn
    alongside training loss. Neither is used for model selection - this
    project trains a fixed epoch budget and keeps the final weights.

    The standard-FM panel holds a single curve because its objective never
    discretizes the path: one training serves every T. The rolled-out panel
    holds one curve per T, because T is baked into those weights.

    Args:
        standard_history: per-epoch history for the standard-FM run.
        rolled_histories: T -> per-epoch history for each rolled-out run.
        dataset, encoder, k_shot, seed: identify the run, for the title.
        save_path: where to save the PNG.

    Raises:
        ValueError: if `standard_history` is empty or `rolled_histories` is.
    """
    if not standard_history:
        raise ValueError("standard_history is empty; nothing to plot")
    if not rolled_histories:
        raise ValueError("rolled_histories is empty; nothing to plot")

    fig, (standard_axis, rolled_axis) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    epochs = [entry["epoch"] for entry in standard_history]
    standard_axis.plot(
        epochs, [entry["train_loss"] for entry in standard_history],
        linestyle=_TRAIN_STYLE, color="tab:blue", label="train",
    )
    validation_losses = [entry["val_loss"] for entry in standard_history]
    if not all(value != value for value in validation_losses):  # skip an all-NaN series
        standard_axis.plot(
            epochs, validation_losses,
            linestyle=_VAL_STYLE, color="tab:blue", label="validation",
        )
    standard_axis.set_title("standard FM (one training serves every T)")
    standard_axis.set_ylabel(r"velocity loss  $\|v_\theta(z_t,t) - u\|^2$")

    for num_euler_steps in sorted(rolled_histories):
        history = rolled_histories[num_euler_steps]
        color = _EULER_COLORS.get(num_euler_steps, _DEFAULT_EULER_COLOR)
        rolled_epochs = [entry["epoch"] for entry in history]
        rolled_axis.plot(
            rolled_epochs, [entry["train_loss"] for entry in history],
            linestyle=_TRAIN_STYLE, color=color, label=f"train (T={num_euler_steps})",
        )
        rolled_validation = [entry["val_loss"] for entry in history]
        if not all(value != value for value in rolled_validation):
            rolled_axis.plot(
                rolled_epochs, rolled_validation,
                linestyle=_VAL_STYLE, color=color, label=f"validation (T={num_euler_steps})",
            )
    rolled_axis.set_title("rolled-out FM (T is baked into the weights)")
    rolled_axis.set_ylabel(r"endpoint loss  $\|\hat{z}_T - p_y\|^2$")

    for axis in (standard_axis, rolled_axis):
        axis.set_xlabel("Epoch")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)

    fig.suptitle(
        f"{dataset} / {encoder}: flow-matching training ({k_shot}-shot, seed {seed})"
        "\nnote: the two panels measure different quantities and are not comparable to each other",
        fontsize=10,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
