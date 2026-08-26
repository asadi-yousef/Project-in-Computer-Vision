"""Measure the transported features at every intermediate Euler step, not
only at the endpoint (stage_2.pdf's optional "comparing samples and
prototypes at intermediate flow times").

The headline accuracy table reports only z_hat_T. That hides the shape of
the transport: whether accuracy improves steadily along the flow, peaks
early and then decays, or is decided in the first step. Evaluating the same
cosine rule at each state z_hat_k answers that directly, over the whole test
split rather than the handful of examples a trajectory plot can show.

Three quantities are tracked per step:
  - accuracy: the classification the flow would produce if stopped here.
  - own-prototype similarity: how close a sample has been pulled to its own
    class prototype. This measures contraction, not correctness.
  - margin: own-prototype similarity minus the best competing prototype's.
    This is what actually decides the prediction - a flow can increase
    similarity to every prototype at once, contracting the space without
    separating the classes any better.
"""

import dataclasses
from typing import List

import torch
import torch.nn.functional as F


@dataclasses.dataclass
class PerStepMetrics:
    """Per-Euler-step measurements along one trajectory.

    All lists have length T+1: index 0 is the untransported feature and
    index T the final transported one. `times` records k/T so that runs with
    different T can be compared on one axis - comparing them by step index
    would put step 4 of a 4-step flow (the end) alongside step 4 of a 12-step
    flow (a third of the way).
    """

    steps: List[int]
    times: List[float]
    accuracies: List[float]
    mean_own_similarity: List[float]
    mean_margin: List[float]


def compute_per_step_metrics(
    trajectory: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor
) -> PerStepMetrics:
    """Evaluate the Stage 1 cosine rule at every state along a trajectory.

    Args:
        trajectory: (T+1, N, D) states from `euler_trajectory`, index 0 being
            the original features.
        labels: (N,) true class labels.
        prototypes: (C, D) class prototypes, the same ones the run used.

    Returns:
        A `PerStepMetrics` with one entry per state.

    Raises:
        ValueError: on shape mismatches between the trajectory, labels and
            prototypes.
    """
    if trajectory.dim() != 3:
        raise ValueError(f"trajectory must be (T+1, N, D), got {trajectory.dim()} dimensions")
    if trajectory.shape[1] != labels.shape[0]:
        raise ValueError(
            f"trajectory has {trajectory.shape[1]} samples but labels has {labels.shape[0]}"
        )
    if trajectory.shape[2] != prototypes.shape[1]:
        raise ValueError(
            f"trajectory features have dimension {trajectory.shape[2]} but "
            f"prototypes have dimension {prototypes.shape[1]}"
        )

    num_steps = trajectory.shape[0] - 1
    sample_indices = torch.arange(labels.shape[0])

    steps, times, accuracies, own_similarities, margins = [], [], [], [], []
    for step in range(trajectory.shape[0]):
        # The same rule Stage 1 classifies with: normalize, then cosine
        # similarity against the (already unit-norm) prototypes.
        similarities = F.normalize(trajectory[step], dim=1) @ prototypes.T

        predictions = similarities.argmax(dim=1)
        own = similarities[sample_indices, labels]

        competitors = similarities.clone()
        competitors[sample_indices, labels] = float("-inf")
        best_competitor = competitors.max(dim=1).values

        steps.append(step)
        times.append(step / num_steps if num_steps else 0.0)
        accuracies.append((predictions == labels).float().mean().item())
        own_similarities.append(own.mean().item())
        margins.append((own - best_competitor).mean().item())

    return PerStepMetrics(
        steps=steps,
        times=times,
        accuracies=accuracies,
        mean_own_similarity=own_similarities,
        mean_margin=margins,
    )
