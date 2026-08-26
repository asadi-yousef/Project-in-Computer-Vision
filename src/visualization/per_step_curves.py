"""Plot how accuracy and class separation evolve along the flow.

Three panels, all against normalized flow time t = k/T so that runs with
different T are directly comparable:

  1. accuracy - what the classifier would report if the flow were stopped here;
  2. similarity to the sample's own prototype - how far the flow has
     contracted the space;
  3. margin (own prototype minus best competitor) - what actually decides the
     prediction.

Panels 2 and 3 are separated deliberately. A flow can raise similarity to
every prototype at once, which contracts the feature space without improving
class separation at all; only the margin distinguishes those cases.
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")  # renders to file without needing a display
import matplotlib.pyplot as plt

from src.evaluation.aggregation import method_label
from src.flow_matching.per_step import PerStepMetrics
from src.visualization.style import euler_style, method_color


def plot_per_step_curves(
    series: Sequence[Tuple[str, Optional[int], PerStepMetrics]],
    dataset: str,
    encoder: str,
    k_shot,
    save_path: Union[str, Path],
    baseline_accuracy: Optional[float] = None,
) -> None:
    """Draw accuracy, own-prototype similarity and margin against flow time.

    Args:
        series: (method, num_euler_steps, metrics) per curve, e.g. standard
            and rolled-out FM at each T.
        dataset, encoder, k_shot: identify the setting, for the title.
        save_path: where to save the PNG.
        baseline_accuracy: drawn as a horizontal reference. Every curve
            starts from it by construction - at k=0 the state is the
            untransported feature, so step 0 reproduces the Stage 1
            prototype accuracy exactly.

    Raises:
        ValueError: if no series are given.
    """
    if not series:
        raise ValueError("series is empty; nothing to plot")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=150)
    accuracy_axis, similarity_axis, margin_axis = axes

    if baseline_accuracy is not None:
        accuracy_axis.axhline(
            baseline_accuracy * 100, color="black", linewidth=1.2,
            linestyle=":", label="prototype baseline",
        )

    for method, num_euler_steps, metrics in series:
        linestyle, marker = euler_style(num_euler_steps)
        style = dict(
            color=method_color(method),
            linestyle=linestyle,
            marker=marker,
            markersize=4,
            label=method_label(method, num_euler_steps),
        )
        accuracy_axis.plot(
            metrics.times, [value * 100 for value in metrics.accuracies], **style
        )
        similarity_axis.plot(metrics.times, metrics.mean_own_similarity, **style)
        margin_axis.plot(metrics.times, metrics.mean_margin, **style)

    accuracy_axis.set_ylabel("Test accuracy (%)")
    accuracy_axis.set_title("accuracy if the flow stopped here", fontsize=10)
    similarity_axis.set_ylabel("Mean cosine to own prototype")
    similarity_axis.set_title("contraction toward the prototype", fontsize=10)
    margin_axis.set_ylabel("Mean margin (own - best competitor)")
    margin_axis.set_title("class separation", fontsize=10)

    for axis in axes:
        axis.set_xlabel("Flow time  t = k / T")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    fig.suptitle(
        f"{dataset} / {encoder}: metrics along the flow (K={k_shot})", fontsize=11
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(save_path)
    plt.close(fig)
