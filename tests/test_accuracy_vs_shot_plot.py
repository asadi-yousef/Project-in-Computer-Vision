import pytest

from src.visualization.accuracy_vs_shot import plot_accuracy_vs_shot


def _synthetic_summaries():
    return [
        {"dataset": "dtd", "encoder": "resnet18", "method": "linear_probe", "k_shot": 5,
         "num_runs": 3, "mean_test_accuracy": 0.3, "std_test_accuracy": 0.05, "seed_accuracies": {}},
        {"dataset": "dtd", "encoder": "resnet18", "method": "linear_probe", "k_shot": 10,
         "num_runs": 3, "mean_test_accuracy": 0.5, "std_test_accuracy": 0.04, "seed_accuracies": {}},
        {"dataset": "dtd", "encoder": "resnet18", "method": "linear_probe", "k_shot": "full",
         "num_runs": 3, "mean_test_accuracy": 0.7, "std_test_accuracy": 0.02, "seed_accuracies": {}},
        {"dataset": "dtd", "encoder": "resnet18", "method": "prototype", "k_shot": 5,
         "num_runs": 3, "mean_test_accuracy": 0.25, "std_test_accuracy": 0.03, "seed_accuracies": {}},
        {"dataset": "dtd", "encoder": "resnet18", "method": "prototype", "k_shot": "full",
         "num_runs": 1, "mean_test_accuracy": 0.4, "std_test_accuracy": None, "seed_accuracies": {}},
    ]


def test_plot_is_saved_to_disk(tmp_path):
    save_path = tmp_path / "plot.png"
    plot_accuracy_vs_shot(_synthetic_summaries(), "dtd", "resnet18", save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_raises_for_missing_dataset_encoder_combination():
    with pytest.raises(ValueError, match="flowers102"):
        plot_accuracy_vs_shot(_synthetic_summaries(), "flowers102", "resnet18", "unused.png")
