"""Stage 3 training objectives: FM before a frozen linear classifier (part_3.pdf).

The pipeline is

    z --FM--> z_hat --frozen linear classifier--> s,

where the FM transformation is the same T-step Euler rollout as Stage 2 and
the classifier is Stage 1's trained linear probe, loaded and frozen by
`src.classifiers.frozen_probe`.

Both of part_3.pdf's strategies live here, and both reuse Stage 2's velocity
network and Euler integrator unchanged. What differs is the objective.

**Strategy 1** (`train_rolled_out_classification`) reuses Stage 2's rolled-out
mechanics - the same full T-step rollout, the same backpropagation through
every velocity prediction - but scores the endpoint with the frozen
classifier instead of against a class prototype:

    L_cls = CE(W z_hat_T + b, y).

**Strategy 2** (`train_classifier_guided_fm`) reuses Stage 2's *standard* FM
mechanics - one random t per sample, a straight-line interpolation, a
constant target velocity - but replaces Stage 2's fixed class prototype with
a target constructed per example from the classifier's own gradient:

    z_hat  = FM(z)                      (the current transported feature)
    z_hat' = z_hat - eta * grad_{z_hat} CE(W z_hat + b, y)
    L_FM   = || v_theta(z_t, t) - (z_hat' - z) ||^2,   z_t = (1-t) z + t z_hat'.

Only the FM parameters are updated in either case. W and b are frozen
upstream, which lets gradients flow through them without accumulating
parameter gradients.

Three departures from `src.flow_matching.training` are deliberate:

  - **Raw feature space.** Stage 2 L2-normalized before the flow because its
    classifier was cosine similarity, which is scale-invariant. Stage 1's
    linear probe is not, and was fitted to unnormalized features with mean
    norms of roughly 24 (ResNet-18) and 48 (DINOv2), so Stage 3 must leave
    them alone. `verify_stored_test_accuracy` enforces this.

  - **Best-validation-accuracy checkpointing.** Stage 2 kept the final
    weights, because standard FM has no single T to validate at and the spec
    asked only for stable training. Stage 3's pipeline has a meaningful
    validation *accuracy*, so it follows Stage 1's convention instead. This
    also guards against the failure Stage 2 documented for its own rolled-out
    variant, where training loss fell for 200 epochs while validation loss
    rose after about 25.

  - **A shared training skeleton.** part_3.pdf and stage_2.pdf both require
    the architecture and main training choices to stay fixed across the
    variants being compared. `_train_stage3_velocity_network` holds every one
    of those - seeding, initialization, optimizer, epoch budget, validation
    protocol, checkpoint selection - so the two strategies cannot drift apart
    on anything except the objective they supply.

Selection runs over the trained epochs only. The untrained network is exactly
the identity (see `build_near_identity_velocity_network`), so including it as
a selection candidate would clamp every reported delta at >= 0 and hide a
result where the flow genuinely hurts. Its accuracy is recorded separately as
`initial_val_accuracy` so the curves and the report can show where training
started.

A note for reading the curves: `train_loss` is whatever objective the
strategy minimizes, so it is a cross-entropy for Strategy 1 and a squared
velocity error for Strategy 2. Those are not comparable to each other, or to
`val_loss`, which is always the pipeline's plain cross-entropy. The
accuracies are comparable throughout.
"""

import dataclasses
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.flow_matching.inference import euler_trajectory, euler_transport
from src.flow_matching.training import flow_matching_loss
from src.flow_matching.velocity_net import build_near_identity_velocity_network
from src.utils.config import Stage3Hyperparams
from src.utils.seeding import set_seed

# Guards the per-sample division when a gradient is numerically zero, which
# is a real possibility here: the frozen probe already classifies its own
# K-shot training subset perfectly, so some examples carry essentially no
# classification gradient at all.
_MIN_GRADIENT_NORM = 1e-12


@dataclasses.dataclass
class Stage3EpochLog:
    """One epoch of the logging part_3.pdf's training curves need.

    The `train_loss`/`val_loss` field names match the linear probe's
    `EpochLog` and Stage 2's `FlowMatchingEpochLog`, so the existing
    `load_history` and loss-curve plotting read this without changes.

    `mean_displacement` is Stage 3-specific: the mean ||z_hat_T - z|| over
    the validation split, in raw feature units. It exists because the one
    genuine risk of optimizing through a frozen classifier is that the
    cheapest descent direction is to inflate the feature magnitude along the
    logit direction rather than to improve the representation. That shows up
    here as displacement growing without validation accuracy following.
    """

    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    mean_displacement: float


@dataclasses.dataclass
class Stage3TrainResult:
    """Outcome of one Stage 3 training run.

    Attributes:
        best_epoch: the epoch whose weights are returned.
        best_val_accuracy: that epoch's validation accuracy.
        best_state_dict: the velocity network's weights at that epoch.
        history: per-epoch logs, starting at epoch 1.
        initial_val_accuracy: validation accuracy of the untrained,
            near-identity network. Equal to the frozen probe's own
            validation accuracy, since the rollout is exactly the identity
            before training - so it is the baseline the curves start from.
        initial_val_loss: the matching validation loss.
    """

    best_epoch: int
    best_val_accuracy: float
    best_state_dict: Dict[str, torch.Tensor]
    history: List[Stage3EpochLog]
    initial_val_accuracy: float
    initial_val_loss: float


def _validate_inputs(
    features: torch.Tensor, labels: torch.Tensor, num_classes: int, name: str
) -> None:
    """Fail loudly on shape/label mismatches rather than training on nonsense."""
    if features.dim() != 2:
        raise ValueError(
            f"{name}_features must be 2-D (N, D), got {features.dim()} dimensions"
        )
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{name}_features has {features.shape[0]} rows but "
            f"{name}_labels has {labels.shape[0]}"
        )
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= num_classes):
        raise ValueError(
            f"{name}_labels must lie in [0, {num_classes - 1}], "
            f"got range [{int(labels.min())}, {int(labels.max())}]"
        )


def rollout_with_velocities(
    velocity_net: nn.Module, features: torch.Tensor, num_steps: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the Euler rollout, returning the endpoint and the velocities used.

    The velocities are recovered from consecutive states rather than
    collected inside the integrator, since the Euler update

        z_hat_{k+1} = z_hat_k + (1/T) v_k    =>    v_k = T (z_hat_{k+1} - z_hat_k)

    makes them exact. That keeps `euler_transport`/`euler_trajectory` - the
    functions Stage 2's completed runs depend on - untouched.

    `euler_trajectory` is used unconditionally rather than only when the
    velocity penalty is active. Backpropagating through the rollout already
    keeps every intermediate state alive, so stacking them adds an allocation
    and nothing else, and T is 4.

    Args:
        velocity_net: the velocity network v_theta.
        features: (N, D) starting points z.
        num_steps: T.

    Returns:
        (transported, velocities) where transported is (N, D) and velocities
        is (T, N, D), ordered by step.
    """
    trajectory = euler_trajectory(velocity_net, features, num_steps)
    velocities = (trajectory[1:] - trajectory[:-1]) * num_steps
    return trajectory[-1], velocities


def classification_rollout_loss(
    velocity_net: nn.Module,
    classifier: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    num_steps: int,
    displacement_penalty: float = 0.0,
    velocity_penalty: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Strategy 1's objective: cross-entropy on the transported features.

    L = CE(W z_hat_T + b, y)
        + lambda_d * mean||z_hat_T - z||^2 / mean||z||^2
        + lambda_v * mean_k mean_i ||v_k,i||^2 / mean||z||^2

    Both penalties are part_3.pdf's optional regularization ("penalizing the
    displacement between z and z_hat, or the magnitude of the predicted
    velocities").

    They are divided by the batch's mean squared feature norm on purpose.
    Raw features differ in scale by roughly a factor of two between this
    project's two encoders, so an absolute penalty weight would mean
    different things on DTD and Flowers-102. Normalized this way, the terms
    are dimensionless - 0 at initialization, 1 when the displacement is
    comparable to the feature norm itself - and one lambda transfers across
    both settings.

    Args:
        velocity_net: the velocity network being trained.
        classifier: the frozen linear probe. Its parameters must already
            have `requires_grad=False`; gradients still flow through it to
            reach the rollout.
        features: (N, D) raw, unnormalized features z.
        labels: (N,) true class labels.
        num_steps: T.
        displacement_penalty, velocity_penalty: the lambdas above.

    Returns:
        (loss, transported) - the scalar loss to backpropagate, and the
        (N, D) endpoint z_hat_T, which the caller reuses for accuracy and
        displacement logging rather than recomputing the rollout.
    """
    transported, velocities = rollout_with_velocities(velocity_net, features, num_steps)
    loss = nn.functional.cross_entropy(classifier(transported), labels)

    if displacement_penalty > 0.0 or velocity_penalty > 0.0:
        mean_squared_norm = features.pow(2).sum(dim=1).mean()
        if displacement_penalty > 0.0:
            displacement = (transported - features).pow(2).sum(dim=1).mean()
            loss = loss + displacement_penalty * displacement / mean_squared_norm
        if velocity_penalty > 0.0:
            magnitude = velocities.pow(2).sum(dim=2).mean()
            loss = loss + velocity_penalty * magnitude / mean_squared_norm

    return loss, transported


def build_classifier_guided_targets(
    velocity_net: nn.Module,
    classifier: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    num_steps: int,
    step_size: float,
    num_target_steps: int = 1,
    normalize: bool = True,
) -> torch.Tensor:
    """Strategy 2's target construction: steps 1-3 of part_3.pdf's recipe.

    Run z through the current FM to get z_hat, then take one or more gradient
    steps on z_hat in *feature space* to build a nearby representation the
    frozen classifier handles better:

        z_hat' = z_hat - eta * grad_{z_hat} CE(W z_hat + b, y)     repeated M times.

    Normalization is what makes this work here, and it is not cosmetic.
    Stage 1's probe already reaches 100% accuracy and a cross-entropy of
    about 0.002 on the very subset Stage 3 trains on (part_3.pdf requires
    reusing it), so the classification gradient is minuscule for nearly every
    training example. An unnormalized step would therefore leave essentially
    every target sitting on top of z_hat, and the FM would be trained to
    reproduce what it already does - measured on synthetic data of the same
    scale, the unnormalized step is about three orders of magnitude shorter
    than the normalized one. Dividing by the gradient's own norm discards
    that vanishing magnitude and keeps only the direction, which is the part
    that still carries information.

    The step is also measured in units of the batch's mean feature norm
    rather than absolute distance, for the same reason the Strategy 1
    penalties are: one `step_size` then means the same thing on both
    encoders despite their feature scales differing by about a factor of two.

    One consequence of clamping the divisor rather than the result: a sample
    whose gradient underflows below `_MIN_GRADIENT_NORM` gets a step shorter
    than eta, shrinking smoothly to nothing as the gradient vanishes, instead
    of a full-length step in a direction that is numerically meaningless.
    That is the desired behaviour - if the frozen classifier offers no
    improvement direction for an example, the target should stay where it is
    - and it happens in practice, not just in theory, since the probe already
    fits this subset perfectly.

    Args:
        velocity_net: the current FM. Used in eval mode under `no_grad`; the
            returned targets never carry gradient back to its parameters,
            since they are training data for the FM loss, not part of it.
        classifier: the frozen linear probe.
        features: (N, D) raw source features z.
        labels: (N,) true labels.
        num_steps: T, for the rollout that produces z_hat.
        step_size: eta. In units of the mean feature norm when `normalize` is
            True, and an ordinary gradient-step size when it is False.
        num_target_steps: M, how many feature-space steps to take.
        normalize: whether to L2-normalize each sample's gradient before
            stepping.

    Returns:
        (N, D) targets z_hat', detached.
    """
    was_training = velocity_net.training
    velocity_net.eval()

    with torch.no_grad():
        target = euler_transport(velocity_net, features, num_steps)

    scale = features.norm(dim=1).mean()

    for _ in range(num_target_steps):
        target = target.detach().requires_grad_(True)
        loss = nn.functional.cross_entropy(classifier(target), labels)
        (gradient,) = torch.autograd.grad(loss, target)

        if normalize:
            norms = gradient.norm(dim=1, keepdim=True).clamp_min(_MIN_GRADIENT_NORM)
            step = step_size * scale * (gradient / norms)
        else:
            step = step_size * gradient

        target = target.detach() - step

    if was_training:
        velocity_net.train()
    return target.detach()


def classifier_guided_fm_loss(
    velocity_net: nn.Module, features: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Strategy 2's objective: Stage 2's standard FM loss, new targets.

    Steps 4-5 of part_3.pdf's recipe. Identical in form to
    `src.flow_matching.training.train_standard_flow_matching` - one
    independent t per sample, a straight-line interpolation, a constant
    target velocity - and it reuses the same `flow_matching_loss` reduction
    so the two stages' losses are computed the same way. The only difference
    is what plays the role of the endpoint: Stage 2 used the fixed class
    prototype p_y, Stage 3 uses the per-example, drifting z_hat'.

    Args:
        velocity_net: the velocity network being trained.
        features: (N, D) source features z.
        targets: (N, D) classifier-guided targets z_hat'.

    Returns:
        Scalar loss.
    """
    times = torch.rand(features.shape[0], device=features.device)
    interpolated = (
        1.0 - times.unsqueeze(1)
    ) * features + times.unsqueeze(1) * targets
    return flow_matching_loss(velocity_net(interpolated, times), targets - features)


def evaluate_pipeline(
    velocity_net: nn.Module,
    classifier: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    num_steps: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Score the complete z -> FM -> frozen classifier pipeline on one split.

    Reports plain cross-entropy and accuracy, never the strategy's own
    training objective. That is what makes the two strategies and the Stage 1
    baseline comparable at all: Strategy 2's objective is a velocity
    regression whose value says nothing about classification, and Strategy
    1's includes a regularization term that is a training device.

    Returns:
        (loss, accuracy, mean_displacement), where mean_displacement is the
        mean ||z_hat_T - z|| in raw feature units.
    """
    was_training = velocity_net.training
    velocity_net.eval()
    features = features.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        transported = euler_transport(velocity_net, features, num_steps)
        logits = classifier(transported)
        loss = nn.functional.cross_entropy(logits, labels).item()
        accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
        displacement = (transported - features).norm(dim=1).mean().item()

    if was_training:
        velocity_net.train()
    return loss, accuracy, displacement


def _train_stage3_velocity_network(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    classifier: nn.Module,
    num_classes: int,
    hyperparams: Stage3Hyperparams,
    seed: int,
    device: torch.device,
    epoch_fn_factory: Callable[
        [nn.Module, torch.optim.Optimizer], Callable[[int], Tuple[float, float]]
    ],
    progress: Optional[Callable[[Stage3EpochLog], None]] = None,
) -> Stage3TrainResult:
    """Shared training skeleton for both Stage 3 strategies.

    Everything part_3.pdf and stage_2.pdf require to stay fixed across the
    compared variants lives here - the near-identity initialization, the
    optimizer, the learning rate and weight decay, the epoch budget, the
    seeding convention, the validation protocol, and best-validation-accuracy
    checkpoint selection. A strategy supplies only its objective.

    Args:
        train_features, train_labels: the K-shot subset, raw features.
        val_features, val_labels: the full official validation split.
        classifier: the frozen Stage 1 linear probe, on `device`.
        num_classes: C, for label validation.
        hyperparams: architecture, optimizer and strategy settings.
        seed: drives initialization and batch shuffling.
        device: device to train on.
        epoch_fn_factory: called once, after the network and optimizer are
            built, as `epoch_fn_factory(velocity_net, optimizer)`. It returns
            a callable invoked as `epoch_fn(epoch)` for each epoch, which
            performs that epoch's updates and returns
            `(train_loss, train_accuracy)`. Taking a factory rather than a
            plain callback preserves the seeding order the earlier stages
            use - seed, then initialize, then build the data loader.
        progress: optional per-epoch callback for console output.

    Returns:
        A `Stage3TrainResult` with the best-validation-accuracy weights.

    Raises:
        ValueError: on shape/label mismatches, or an unsupported optimizer.
    """
    if hyperparams.optimizer != "adamw":
        raise ValueError(f"Only 'adamw' is currently supported, got {hyperparams.optimizer!r}")

    _validate_inputs(train_features, train_labels, num_classes, "train")
    _validate_inputs(val_features, val_labels, num_classes, "val")

    num_steps = hyperparams.num_euler_steps
    set_seed(seed)

    velocity_net = build_near_identity_velocity_network(
        train_features.shape[1], hyperparams.hidden_dims
    ).to(device)
    optimizer = torch.optim.AdamW(
        velocity_net.parameters(),
        lr=hyperparams.learning_rate,
        weight_decay=hyperparams.weight_decay,
    )

    # The identity starting point: equal to the frozen probe's own accuracy,
    # and the reference every delta is read against.
    initial_val_loss, initial_val_accuracy, _ = evaluate_pipeline(
        velocity_net, classifier, val_features, val_labels, num_steps, device
    )

    run_epoch = epoch_fn_factory(velocity_net, optimizer)

    history: List[Stage3EpochLog] = []
    best_val_accuracy = -1.0
    best_epoch = -1
    best_state_dict: Dict[str, torch.Tensor] = {}

    for epoch in range(1, hyperparams.max_epochs + 1):
        velocity_net.train()
        train_loss, train_accuracy = run_epoch(epoch)

        val_loss, val_accuracy, mean_displacement = evaluate_pipeline(
            velocity_net, classifier, val_features, val_labels, num_steps, device
        )
        epoch_log = Stage3EpochLog(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            mean_displacement=mean_displacement,
        )
        history.append(epoch_log)
        if progress is not None:
            progress(epoch_log)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_state_dict = {
                k: v.detach().clone().cpu() for k, v in velocity_net.state_dict().items()
            }

    return Stage3TrainResult(
        best_epoch=best_epoch,
        best_val_accuracy=best_val_accuracy,
        best_state_dict=best_state_dict,
        history=history,
        initial_val_accuracy=initial_val_accuracy,
        initial_val_loss=initial_val_loss,
    )


def train_rolled_out_classification(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    classifier: nn.Module,
    num_classes: int,
    hyperparams: Stage3Hyperparams,
    seed: int,
    device: torch.device,
    progress: Optional[Callable[[Stage3EpochLog], None]] = None,
) -> Stage3TrainResult:
    """Strategy 1: train the FM end to end through the frozen classifier.

    Each batch runs the complete T-step Euler rollout from the raw training
    feature, scores the endpoint with the frozen classifier, and
    backpropagates through all T velocity predictions. Only the velocity
    network is updated.

    `train_loss` in the returned history is the regularized objective that is
    actually minimized, and `train_accuracy` is a running average over the
    epoch's batches, matching Stage 1's linear-probe convention.

    Args:
        train_features: (N, D) raw features, already restricted to the K-shot
            subset by the caller using Stage 1's own sampler and seed.
        train_labels: (N,) labels.
        val_features, val_labels: the full official validation split, used
            for checkpoint selection exactly as in Stage 1. Never trained on.
        classifier: the frozen Stage 1 linear probe, on `device`.
        num_classes: C, for label validation.
        hyperparams: architecture, optimizer and regularization settings.
        seed: drives initialization and batch shuffling.
        device: device to train on.
        progress: optional per-epoch callback.

    Returns:
        A `Stage3TrainResult` with the best-validation-accuracy weights.
    """
    num_steps = hyperparams.num_euler_steps

    def epoch_fn_factory(velocity_net, optimizer):
        loader = DataLoader(
            TensorDataset(train_features, train_labels),
            batch_size=hyperparams.batch_size,
            shuffle=True,
        )

        def run_epoch(epoch: int) -> Tuple[float, float]:
            running_loss = 0.0
            running_correct = 0
            num_samples = 0

            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()
                loss, transported = classification_rollout_loss(
                    velocity_net,
                    classifier,
                    batch_features,
                    batch_labels,
                    num_steps,
                    hyperparams.displacement_penalty,
                    hyperparams.velocity_penalty,
                )
                loss.backward()
                optimizer.step()

                batch_size = batch_labels.shape[0]
                running_loss += loss.item() * batch_size
                with torch.no_grad():
                    predictions = classifier(transported).argmax(dim=1)
                    running_correct += (predictions == batch_labels).sum().item()
                num_samples += batch_size

            return running_loss / num_samples, running_correct / num_samples

        return run_epoch

    return _train_stage3_velocity_network(
        train_features, train_labels, val_features, val_labels, classifier,
        num_classes, hyperparams, seed, device, epoch_fn_factory, progress,
    )


def train_classifier_guided_fm(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    classifier: nn.Module,
    num_classes: int,
    hyperparams: Stage3Hyperparams,
    seed: int,
    device: torch.device,
    progress: Optional[Callable[[Stage3EpochLog], None]] = None,
) -> Stage3TrainResult:
    """Strategy 2: standard FM training toward classifier-guided targets.

    Implements part_3.pdf's six-step recipe. Targets are rebuilt for the
    whole training subset every `target_refresh_epochs` epochs (step 6,
    "recompute the targets as the FM changes during training"), then held
    fixed within that block so the FM regresses against a stable objective
    rather than one moving under it batch by batch.

    `train_loss` in the returned history is the FM velocity-regression loss -
    the objective actually minimized, and *not* comparable to Strategy 1's
    cross-entropy. `train_accuracy` is measured by a separate end-of-epoch
    pass of the training subset through the full pipeline, because this
    strategy's batches never produce a classification of their own. It is
    therefore an end-of-epoch figure where Strategy 1's is a running average;
    both are honest, and the validation columns - measured identically for
    both - are the ones to compare across strategies.

    Args:
        train_features: (N, D) raw features, already restricted to the K-shot
            subset by the caller using Stage 1's own sampler and seed.
        train_labels: (N,) labels.
        val_features, val_labels: the full official validation split.
        classifier: the frozen Stage 1 linear probe, on `device`.
        num_classes: C, for label validation.
        hyperparams: architecture, optimizer and target-construction settings.
        seed: drives initialization, batch shuffling, and the sampled times.
        device: device to train on.
        progress: optional per-epoch callback.

    Returns:
        A `Stage3TrainResult` with the best-validation-accuracy weights.
    """
    num_steps = hyperparams.num_euler_steps

    def epoch_fn_factory(velocity_net, optimizer):
        features_on_device = train_features.to(device)
        labels_on_device = train_labels.to(device)
        indices = torch.arange(features_on_device.shape[0])
        loader = DataLoader(
            TensorDataset(indices), batch_size=hyperparams.batch_size, shuffle=True
        )
        targets = {"value": None}

        def run_epoch(epoch: int) -> Tuple[float, float]:
            # Step 6: refresh the targets against the FM as it is now.
            if (epoch - 1) % hyperparams.target_refresh_epochs == 0:
                targets["value"] = build_classifier_guided_targets(
                    velocity_net,
                    classifier,
                    features_on_device,
                    labels_on_device,
                    num_steps,
                    hyperparams.target_step_size,
                    hyperparams.target_num_steps,
                    hyperparams.normalize_target_update,
                )

            running_loss = 0.0
            num_samples = 0
            for (batch_indices,) in loader:
                batch_indices = batch_indices.to(device)
                batch_features = features_on_device[batch_indices]
                batch_targets = targets["value"][batch_indices]

                optimizer.zero_grad()
                loss = classifier_guided_fm_loss(
                    velocity_net, batch_features, batch_targets
                )
                loss.backward()
                optimizer.step()

                batch_size = batch_indices.shape[0]
                running_loss += loss.item() * batch_size
                num_samples += batch_size

            _, train_accuracy, _ = evaluate_pipeline(
                velocity_net, classifier, features_on_device, labels_on_device,
                num_steps, device,
            )
            return running_loss / num_samples, train_accuracy

        return run_epoch

    return _train_stage3_velocity_network(
        train_features, train_labels, val_features, val_labels, classifier,
        num_classes, hyperparams, seed, device, epoch_fn_factory, progress,
    )
