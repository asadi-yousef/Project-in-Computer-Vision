"""Load a completed Stage 1 linear probe and freeze it, for use as Stage 3's
fixed classifier (part_3.pdf).

part_3.pdf: "Train the linear classifier first, exactly as in Stage 1, and
then keep it frozen." This project reads that literally - the classifier is
not retrained, it is the checkpoint Stage 1 already produced and reported.
That makes every Stage 3 delta a paired comparison against a specific,
already-published baseline number rather than against a freshly trained
probe that would differ from it by seed noise.

"Frozen" here means exactly one thing: W and b never receive updates.
Gradients must still flow *through* the classifier, because Strategy 1
backpropagates the classification loss through it and the whole Euler
rollout to reach the velocity network. Setting `requires_grad=False` on the
parameters gives precisely that behaviour - the input gradient is computed,
the parameter gradients are not - which is why this module freezes that way
rather than by detaching or by wrapping calls in `torch.no_grad()`, either of
which would cut Strategy 1's gradient path entirely.
"""

import dataclasses
import json
from pathlib import Path
from typing import Union

import torch

from src.classifiers.linear_probe import LinearProbe
from src.classifiers.linear_probe_runner import linear_probe_run_dir

# How far the recomputed test accuracy may sit from the number Stage 1
# stored before `verify_stored_test_accuracy` treats it as a real mismatch.
# Loading the wrong checkpoint, or feeding it features prepared differently
# from Stage 1's, moves accuracy by whole percentage points; a single
# borderline sample flipping because the run was originally evaluated on a
# different device moves it by 1/N, which is about 0.05% on DTD's test
# split. This sits between the two.
ACCURACY_TOLERANCE = 1e-3


@dataclasses.dataclass
class FrozenLinearProbe:
    """A Stage 1 linear probe, loaded and frozen, with its provenance.

    Attributes:
        model: the probe itself, on the requested device, in eval mode, with
            every parameter's `requires_grad` set to False.
        feature_dim: D, read from the checkpoint rather than passed in, so a
            mismatch with the cached features fails loudly at the first
            forward pass instead of being silently accepted.
        num_classes: C, read from the checkpoint for the same reason.
        stored_test_accuracy: the top-1 test accuracy Stage 1 recorded for
            this exact run. This is the Stage 3 baseline that every delta is
            measured against.
        run_dir: where the checkpoint came from, for error messages and for
            recording in the Stage 3 run's metadata.
    """

    model: LinearProbe
    feature_dim: int
    num_classes: int
    stored_test_accuracy: float
    run_dir: Path


def load_frozen_linear_probe(
    output_dir: Union[str, Path],
    dataset: str,
    encoder: str,
    k_shot,
    seed: int,
    device: torch.device,
) -> FrozenLinearProbe:
    """Load the Stage 1 linear-probe run for this setting and freeze it.

    Args:
        output_dir: the project outputs root (the directory holding
            `linear_probe/`), not the run directory itself.
        dataset, encoder, k_shot, seed: identify which Stage 1 run to load.
            Stage 3 pairs each of its runs with the Stage 1 run of the same
            seed, so that seed's subset, classifier and baseline number all
            belong together.
        device: device to place the probe on.

    Returns:
        A `FrozenLinearProbe` with the loaded model and its stored accuracy.

    Raises:
        FileNotFoundError: if the run's checkpoint or result.json is absent,
            meaning Stage 1 has not been run for this setting.
        KeyError: if the checkpoint is not a linear-probe state dict.
    """
    run_dir = linear_probe_run_dir(output_dir, dataset, encoder, k_shot, seed)
    checkpoint_path = run_dir / "checkpoint.pt"
    result_path = run_dir / "result.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No Stage 1 checkpoint at {checkpoint_path}. Stage 3 reuses Stage 1's "
            "trained classifier, so that run must be completed first "
            "(scripts/run_all_experiments.py)."
        )
    if not result_path.exists():
        raise FileNotFoundError(
            f"No Stage 1 result.json at {result_path}. Stage 3 needs the stored "
            "test accuracy as its baseline, and to verify the loaded checkpoint "
            "is the one that produced it."
        )

    state_dict = torch.load(checkpoint_path, weights_only=True)
    if "linear.weight" not in state_dict:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} has keys {sorted(state_dict)}; "
            "expected a LinearProbe state dict containing 'linear.weight'."
        )

    num_classes, feature_dim = state_dict["linear.weight"].shape

    model = LinearProbe(feature_dim, num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with open(result_path) as f:
        stored_test_accuracy = json.load(f)["result"]["test_accuracy"]

    return FrozenLinearProbe(
        model=model,
        feature_dim=int(feature_dim),
        num_classes=int(num_classes),
        stored_test_accuracy=float(stored_test_accuracy),
        run_dir=run_dir,
    )


def frozen_probe_accuracy(
    frozen: FrozenLinearProbe,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> float:
    """Top-1 accuracy of the frozen probe on `features`, with no transport.

    Used both to verify the loaded checkpoint and, in the Stage 3 runner, to
    compute the baseline each result is compared against.
    """
    with torch.no_grad():
        logits = frozen.model(features.to(device))
        predictions = logits.argmax(dim=1)
        return (predictions == labels.to(device)).float().mean().item()


def verify_stored_test_accuracy(
    frozen: FrozenLinearProbe,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
    tolerance: float = ACCURACY_TOLERANCE,
) -> float:
    """Check the loaded probe still produces the accuracy Stage 1 recorded.

    This is the integrity check that makes Stage 3's deltas trustworthy. It
    catches the failure modes that would otherwise be invisible and would
    quietly invalidate every number in the stage: the wrong seed's
    checkpoint being loaded, a cache rebuilt with different features since
    Stage 1 ran, or features being prepared differently here than they were
    then - notably L2-normalizing them, which Stage 2 does and Stage 3 must
    not, and which costs several accuracy points on its own.

    Args:
        frozen: the loaded probe.
        test_features: the full official test split, prepared exactly as
            Stage 1 prepared it (i.e. raw cached features, not normalized).
        test_labels: matching labels.
        device: device to evaluate on.
        tolerance: allowed absolute difference; see `ACCURACY_TOLERANCE`.

    Returns:
        The recomputed test accuracy.

    Raises:
        ValueError: if the recomputed accuracy differs from the stored one by
            more than `tolerance`.
    """
    recomputed = frozen_probe_accuracy(frozen, test_features, test_labels, device)
    difference = abs(recomputed - frozen.stored_test_accuracy)

    if difference > tolerance:
        raise ValueError(
            f"Frozen probe from {frozen.run_dir} scores {recomputed:.4f} on the test "
            f"split but Stage 1 recorded {frozen.stored_test_accuracy:.4f} "
            f"(difference {difference:.4f} > tolerance {tolerance}). The checkpoint, "
            "the cached features, or the way they are prepared has changed since "
            "Stage 1 ran; Stage 3's comparison against this baseline would be invalid."
        )

    return recomputed
