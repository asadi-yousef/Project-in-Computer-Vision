"""Plot per-epoch training/validation loss for a representative linear-probe
run. stage_1.pdf requires this for "representative 10-shot runs"; this
project uses seed 0 as that representative run for each dataset/encoder
pair (a documented convention, not a spec requirement of which seed).
"""

import json
from pathlib import Path
from typing import List, Union

import matplotlib

matplotlib.use("Agg")  # renders to file without needing a display
import matplotlib.pyplot as plt


def load_history(path: Union[str, Path]) -> List[dict]:
    """Load a run's per-epoch history.json (written by the linear-probe experiment runner)."""
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
