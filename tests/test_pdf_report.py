import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.pdf_report import generate_pdf_report


def _make_tiny_png(path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.savefig(path)
    plt.close(fig)


def test_generate_pdf_report_creates_a_valid_pdf_file(tmp_path):
    summaries = [
        {
            "dataset": "dtd", "encoder": "resnet18", "method": "linear_probe", "k_shot": 10,
            "num_runs": 3, "mean_test_accuracy": 0.5, "std_test_accuracy": 0.05, "seed_accuracies": {},
        },
        {
            "dataset": "dtd", "encoder": "resnet18", "method": "prototype", "k_shot": "full",
            "num_runs": 1, "mean_test_accuracy": 0.6, "std_test_accuracy": None, "seed_accuracies": {},
        },
    ]
    png_path = tmp_path / "plot.png"
    _make_tiny_png(png_path)

    pdf_path = tmp_path / "results.pdf"
    generate_pdf_report(summaries, [("dtd", "resnet18", png_path)], pdf_path)

    assert pdf_path.exists()
    with open(pdf_path, "rb") as f:
        header = f.read(5)
    assert header == b"%PDF-"
