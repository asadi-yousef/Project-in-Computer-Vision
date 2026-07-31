import json

from src.visualization.loss_curves import load_history, plot_loss_curve


def _synthetic_history():
    return [
        {"epoch": 1, "train_loss": 1.5, "val_loss": 1.6, "train_accuracy": 0.2, "val_accuracy": 0.15, "learning_rate": 1e-3},
        {"epoch": 2, "train_loss": 1.2, "val_loss": 1.3, "train_accuracy": 0.4, "val_accuracy": 0.35, "learning_rate": 1e-3},
        {"epoch": 3, "train_loss": 0.9, "val_loss": 1.1, "train_accuracy": 0.6, "val_accuracy": 0.5, "learning_rate": 1e-3},
    ]


def test_load_history_reads_json_list(tmp_path):
    history = _synthetic_history()
    path = tmp_path / "history.json"
    with open(path, "w") as f:
        json.dump(history, f)

    loaded = load_history(path)

    assert loaded == history


def test_plot_loss_curve_is_saved_to_disk(tmp_path):
    save_path = tmp_path / "loss.png"
    plot_loss_curve(_synthetic_history(), "dtd", "resnet18", save_path)

    assert save_path.exists()
    assert save_path.stat().st_size > 0
