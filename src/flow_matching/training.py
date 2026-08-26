"""The two flow-matching training objectives from stage_2.pdf.

Standard FM. For each training feature z_i with class prototype p_{y_i},
sample t ~ U(0, 1), form the interpolated state

    z_t = (1 - t) z_i + t p_{y_i},

and regress the velocity network toward the constant target velocity
u_i = p_{y_i} - z_i:

    L_FM = || v_theta(z_t, t) - u_i ||_2^2.

Supervision is pointwise along the *ideal* straight-line path; the network
never sees its own predictions while training. T never appears, because the
path is never discretized - so one trained network serves every T at
inference.

Rolled-out FM. Starting from z_hat_0 = z_i, apply the same full T-step Euler
sequence used at inference, and supervise only where it ends up:

    L_roll = || z_hat_T - p_{y_i} ||_2^2,

backpropagating through all T velocity predictions. This exposes the network
to the self-generated intermediate states it will actually meet at inference,
at the cost of leaving those intermediate states otherwise unconstrained. T
is baked into the learned weights, so training and inference must use the
same T.

Both objectives share `_train_velocity_network` - same architecture, same
optimizer, same schedule, differing only in the per-batch loss. That is a
requirement, not a convenience: stage_2.pdf asks that the network
architecture and main training choices stay fixed when comparing the two.

Features are expected to be L2-normalized by the caller (the experiment
runner is the single place that normalizes, so training, inference, and the
visualizations all operate on the same representation).
"""

import dataclasses
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.flow_matching.inference import euler_transport
from src.flow_matching.velocity_net import VelocityNetwork
from src.utils.config import FlowMatchingHyperparams
from src.utils.seeding import set_seed


@dataclasses.dataclass
class FlowMatchingEpochLog:
    """One epoch of the logging stage_2.pdf's training curves need.

    Field names match the linear probe's `EpochLog` so the same history.json
    loader reads both.
    """

    epoch: int
    train_loss: float
    val_loss: float


@dataclasses.dataclass
class FlowMatchingTrainResult:
    """Outcome of one flow-matching training run.

    The *final* state dict, not a best-epoch checkpoint: this project trains
    the FM layer for a fixed epoch budget with no validation-based selection
    (stage_2.pdf asks only for stable training and a fair comparison, and
    standard FM has no single T to validate at). `history` is kept for the
    required training-curve figures.
    """

    final_state_dict: Dict[str, torch.Tensor]
    history: List[FlowMatchingEpochLog]


def flow_matching_loss(
    predicted_velocity: torch.Tensor, target_velocity: torch.Tensor
) -> torch.Tensor:
    """L_FM: mean over samples of the per-sample squared velocity error.

    Written as a squared norm summed over the feature dimension (then
    averaged over the batch) to match stage_2.pdf's formula literally,
    rather than the element-wise mean `nn.MSELoss` would give. With AdamW
    the two differ only by a constant factor that the optimizer's per-
    parameter scaling absorbs, so this is about reported losses being
    comparable to the spec, not about optimization behaviour.

    Args:
        predicted_velocity, target_velocity: (N, D) tensors.

    Returns:
        Scalar loss.
    """
    return (predicted_velocity - target_velocity).pow(2).sum(dim=1).mean()


def rolled_out_loss(
    transported_features: torch.Tensor, target_prototypes: torch.Tensor
) -> torch.Tensor:
    """L_roll: mean over samples of the per-sample squared distance from the
    final transported point to its class prototype.

    The same reduction as `flow_matching_loss`, but applied to positions
    rather than velocities. Kept as a separate named function because the
    two are distinct objectives in the spec and appear on separate training
    curves - their numeric values are not comparable to each other.

    Args:
        transported_features: (N, D) final states z_hat_T.
        target_prototypes: (N, D) prototype p_{y_i} for each sample.

    Returns:
        Scalar loss.
    """
    return (transported_features - target_prototypes).pow(2).sum(dim=1).mean()


def _validate_inputs(
    features: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, name: str
) -> None:
    """Fail loudly on shape/label mismatches that would otherwise silently
    train against the wrong targets."""
    if features.dim() != 2:
        raise ValueError(f"{name}_features must be 2-D (N, D), got {features.dim()} dimensions")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{name}_features has {features.shape[0]} rows but "
            f"{name}_labels has {labels.shape[0]}"
        )
    if features.shape[1] != prototypes.shape[1]:
        raise ValueError(
            f"{name}_features have dimension {features.shape[1]} but "
            f"prototypes have dimension {prototypes.shape[1]}"
        )
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= prototypes.shape[0]):
        raise ValueError(
            f"{name}_labels must index the {prototypes.shape[0]} prototypes, "
            f"got range [{int(labels.min())}, {int(labels.max())}]"
        )


def _fixed_validation_times(num_samples: int, seed: int) -> torch.Tensor:
    """Draw the validation split's times once, deterministically.

    Re-drawing t every epoch would make the validation curve jitter for
    reasons unrelated to the model, obscuring exactly what the curve exists
    to show (that training is stable). A dedicated generator is used so this
    draw does not perturb the global RNG stream that seeds initialization
    and batch shuffling.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(num_samples, generator=generator)


def _train_velocity_network(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    prototypes: torch.Tensor,
    hyperparams: FlowMatchingHyperparams,
    seed: int,
    device: torch.device,
    batch_loss_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
    val_loss_fn: Optional[Callable[[nn.Module], float]],
) -> FlowMatchingTrainResult:
    """Shared training loop for both FM objectives.

    Everything the spec requires to stay fixed across the comparison -
    architecture, optimizer, learning rate, weight decay, batch size, epoch
    budget, seeding, and the final-weights convention - lives here. The two
    variants supply only `batch_loss_fn`.

    Args:
        train_features, train_labels, prototypes: as in the public wrappers.
        hyperparams: architecture and optimizer settings.
        seed: drives initialization, batch shuffling, and any sampled times.
        device: device to train on.
        batch_loss_fn: called as `batch_loss_fn(model, features, targets)`
            with the batch's features and its per-sample target prototypes,
            returning the scalar loss to backpropagate.
        val_loss_fn: called as `val_loss_fn(model)` once per epoch inside
            eval mode and `no_grad`, or None to skip validation logging.

    Returns:
        A `FlowMatchingTrainResult` with the final weights and per-epoch history.
    """
    if hyperparams.optimizer != "adamw":
        raise ValueError(f"Only 'adamw' is currently supported, got {hyperparams.optimizer!r}")

    set_seed(seed)

    model = VelocityNetwork(train_features.shape[1], hyperparams.hidden_dims).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hyperparams.learning_rate, weight_decay=hyperparams.weight_decay
    )

    prototypes = prototypes.to(device)
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=hyperparams.batch_size,
        shuffle=True,
    )

    history: List[FlowMatchingEpochLog] = []
    for epoch in range(1, hyperparams.max_epochs + 1):
        model.train()
        running_loss = 0.0
        num_samples = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = prototypes[batch_labels.to(device)]

            optimizer.zero_grad()
            loss = batch_loss_fn(model, batch_features, batch_targets)
            loss.backward()
            optimizer.step()

            batch_size = batch_features.shape[0]
            running_loss += loss.item() * batch_size
            num_samples += batch_size

        train_loss = running_loss / num_samples
        if val_loss_fn is None:
            val_loss = float("nan")
        else:
            model.eval()
            with torch.no_grad():
                val_loss = val_loss_fn(model)
        history.append(FlowMatchingEpochLog(epoch, train_loss, val_loss))

    final_state_dict = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    return FlowMatchingTrainResult(final_state_dict=final_state_dict, history=history)


def train_standard_flow_matching(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    prototypes: torch.Tensor,
    hyperparams: FlowMatchingHyperparams,
    seed: int,
    device: torch.device,
    val_features: Optional[torch.Tensor] = None,
    val_labels: Optional[torch.Tensor] = None,
) -> FlowMatchingTrainResult:
    """Train a velocity network with the standard flow-matching objective.

    Args:
        train_features: (N, D) L2-normalized training features, already
            restricted to the K-shot subset if applicable.
        train_labels: (N,) class labels indexing `prototypes`.
        prototypes: (C, D) class prototypes from Stage 1's
            `compute_class_prototypes`, computed on this same subset. These
            are the flow's fixed targets and are never updated.
        hyperparams: architecture and optimizer settings.
            `num_euler_steps` is ignored - standard FM does not discretize.
        seed: drives initialization, batch shuffling, and the sampled times.
        device: device to train on.
        val_features, val_labels: optional validation split, used only to
            log a validation loss for the training-stability figures. Never
            used for checkpoint or hyperparameter selection.

    Returns:
        A `FlowMatchingTrainResult` with the final weights and per-epoch history.

    Raises:
        ValueError: on shape/label mismatches, or an unsupported optimizer.
    """
    _validate_inputs(train_features, train_labels, prototypes, "train")
    has_validation = val_features is not None and val_labels is not None
    if has_validation:
        _validate_inputs(val_features, val_labels, prototypes, "val")

    def batch_loss_fn(model, features, targets):
        # One independent t per sample, not one per batch: the objective is
        # an expectation over t for each individual training point.
        times = torch.rand(features.shape[0], device=features.device)
        interpolated = (1.0 - times.unsqueeze(1)) * features + times.unsqueeze(1) * targets
        return flow_matching_loss(model(interpolated, times), targets - features)

    val_loss_fn = None
    if has_validation:
        val_features_on_device = val_features.to(device)
        val_targets = prototypes.to(device)[val_labels.to(device)]
        val_times = _fixed_validation_times(val_features.shape[0], seed).to(device)

        def val_loss_fn(model):  # noqa: F811 - defined only when validating
            interpolated = (
                1.0 - val_times.unsqueeze(1)
            ) * val_features_on_device + val_times.unsqueeze(1) * val_targets
            target_velocity = val_targets - val_features_on_device
            return flow_matching_loss(model(interpolated, val_times), target_velocity).item()

    return _train_velocity_network(
        train_features, train_labels, prototypes, hyperparams, seed, device,
        batch_loss_fn, val_loss_fn,
    )


def train_rolled_out_flow_matching(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    prototypes: torch.Tensor,
    hyperparams: FlowMatchingHyperparams,
    seed: int,
    device: torch.device,
    val_features: Optional[torch.Tensor] = None,
    val_labels: Optional[torch.Tensor] = None,
) -> FlowMatchingTrainResult:
    """Train a velocity network by unrolling the full T-step Euler sequence.

    Each batch is integrated exactly as at inference, and only the endpoint
    is supervised, with gradients flowing back through all T velocity
    predictions.

    Args:
        train_features: (N, D) L2-normalized training features, already
            restricted to the K-shot subset if applicable.
        train_labels: (N,) class labels indexing `prototypes`.
        prototypes: (C, D) class prototypes from Stage 1's
            `compute_class_prototypes`, computed on this same subset.
        hyperparams: architecture and optimizer settings.
            `num_euler_steps` (T) is used here and must match the T used at
            inference for this run.
        seed: drives initialization and batch shuffling. No times are
            sampled - the rollout visits the fixed schedule k/T.
        device: device to train on.
        val_features, val_labels: optional validation split, used only to
            log a validation loss. Note this loss is a squared *distance to
            the prototype* and is not comparable to standard FM's velocity
            loss.

    Returns:
        A `FlowMatchingTrainResult` with the final weights and per-epoch history.

    Raises:
        ValueError: on shape/label mismatches, or an unsupported optimizer.
    """
    _validate_inputs(train_features, train_labels, prototypes, "train")
    has_validation = val_features is not None and val_labels is not None
    if has_validation:
        _validate_inputs(val_features, val_labels, prototypes, "val")

    num_steps = hyperparams.num_euler_steps

    def batch_loss_fn(model, features, targets):
        transported = euler_transport(model, features, num_steps)
        return rolled_out_loss(transported, targets)

    val_loss_fn = None
    if has_validation:
        val_features_on_device = val_features.to(device)
        val_targets = prototypes.to(device)[val_labels.to(device)]

        def val_loss_fn(model):  # noqa: F811 - defined only when validating
            transported = euler_transport(model, val_features_on_device, num_steps)
            return rolled_out_loss(transported, val_targets).item()

    return _train_velocity_network(
        train_features, train_labels, prototypes, hyperparams, seed, device,
        batch_loss_fn, val_loss_fn,
    )
