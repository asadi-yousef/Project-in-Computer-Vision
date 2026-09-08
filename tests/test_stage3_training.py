"""Tests for Stage 3's training objectives (part_3.pdf).

Synthetic data throughout: the point is that the mechanics are right, not
that a 16-dimensional random problem is learnable.
"""

import pytest
import torch
import torch.nn as nn

from src.classifiers.linear_probe import LinearProbe
from src.flow_matching.inference import euler_transport
from src.flow_matching import stage3_training
from src.flow_matching.stage3_training import (
    build_classifier_guided_targets,
    classification_rollout_loss,
    classifier_guided_fm_loss,
    evaluate_pipeline,
    rollout_with_velocities,
    train_classifier_guided_fm,
    train_joint_finetuning,
    train_rolled_out_classification,
)
from src.flow_matching.velocity_net import (
    VelocityNetwork,
    build_near_identity_velocity_network,
)
from src.utils.config import Stage3Hyperparams
from src.utils.seeding import set_seed

CPU = torch.device("cpu")
FEATURE_DIM = 16
NUM_CLASSES = 5
FEATURE_SCALE = 24.0  # raw-scale features, like this project's real caches


class ProportionalVelocity(nn.Module):
    """A stand-in velocity field v(z, t) = alpha * z.

    Used where a test needs a *non-identity* flow whose behaviour scales
    exactly with the feature magnitude, which a trained network would not.
    """

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, features, time):
        return self.alpha * features


def _frozen_classifier(seed: int = 0) -> LinearProbe:
    set_seed(seed)
    classifier = LinearProbe(FEATURE_DIM, NUM_CLASSES)
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    return classifier


def _synthetic_split(num_samples: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(num_samples, FEATURE_DIM, generator=generator) * FEATURE_SCALE
    labels = torch.randint(0, NUM_CLASSES, (num_samples,), generator=generator)
    return features, labels


# --- rollout_with_velocities ---


def test_endpoint_matches_euler_transport():
    set_seed(0)
    net = VelocityNetwork(FEATURE_DIM, [32, 32])
    features, _ = _synthetic_split(7)

    with torch.no_grad():
        endpoint, _ = rollout_with_velocities(net, features, num_steps=4)
        expected = euler_transport(net, features, num_steps=4)

    assert torch.equal(endpoint, expected)


def test_recovered_velocities_are_the_ones_the_integrator_used():
    # v_k = T (z_{k+1} - z_k) is exact for explicit Euler, so the recovered
    # velocities must equal a direct evaluation of the network at each state.
    set_seed(0)
    net = VelocityNetwork(FEATURE_DIM, [32, 32])
    features, _ = _synthetic_split(6)
    num_steps = 4

    with torch.no_grad():
        _, velocities = rollout_with_velocities(net, features, num_steps)

        state = features
        for step in range(num_steps):
            expected = net(state, step / num_steps)
            assert torch.allclose(velocities[step], expected, atol=1e-4)
            state = state + expected / num_steps


def test_velocities_have_one_entry_per_step():
    net = build_near_identity_velocity_network(FEATURE_DIM, [32, 32])
    features, _ = _synthetic_split(5)

    _, velocities = rollout_with_velocities(net, features, num_steps=12)

    assert velocities.shape == (12, 5, FEATURE_DIM)


# --- classification_rollout_loss ---


def test_loss_at_initialization_is_the_frozen_classifiers_own_loss():
    # The pipeline starts as the identity, so Strategy 1's objective must
    # start at exactly the Stage 1 linear probe's cross-entropy.
    classifier = _frozen_classifier()
    net = build_near_identity_velocity_network(FEATURE_DIM, [32, 32])
    features, labels = _synthetic_split(32)

    loss, transported = classification_rollout_loss(
        net, classifier, features, labels, num_steps=4
    )
    expected = nn.functional.cross_entropy(classifier(features), labels)

    assert torch.equal(transported, features)
    assert loss.item() == pytest.approx(expected.item())


def test_penalties_are_zero_at_initialization():
    # Nothing has moved yet, so neither penalty may shift the objective.
    classifier = _frozen_classifier()
    net = build_near_identity_velocity_network(FEATURE_DIM, [32, 32])
    features, labels = _synthetic_split(32)

    plain, _ = classification_rollout_loss(net, classifier, features, labels, 4)
    penalized, _ = classification_rollout_loss(
        net, classifier, features, labels, 4,
        displacement_penalty=10.0, velocity_penalty=10.0,
    )

    assert penalized.item() == pytest.approx(plain.item())


def test_penalties_increase_the_loss_once_the_flow_moves():
    classifier = _frozen_classifier()
    net = ProportionalVelocity(alpha=0.5)
    features, labels = _synthetic_split(32)

    plain, _ = classification_rollout_loss(net, classifier, features, labels, 4)
    with_displacement, _ = classification_rollout_loss(
        net, classifier, features, labels, 4, displacement_penalty=1.0
    )
    with_velocity, _ = classification_rollout_loss(
        net, classifier, features, labels, 4, velocity_penalty=1.0
    )

    assert with_displacement.item() > plain.item()
    assert with_velocity.item() > plain.item()


@pytest.mark.parametrize("scale", [0.5, 2.0, 10.0])
def test_penalty_terms_are_invariant_to_the_feature_scale(scale):
    # The reason both penalties are divided by the mean squared feature norm:
    # this project's two encoders differ in feature scale by about 2x, and one
    # lambda has to mean the same thing for both.
    classifier = _frozen_classifier()
    net = ProportionalVelocity(alpha=0.5)
    features, labels = _synthetic_split(32)

    def penalty_term(z):
        plain, _ = classification_rollout_loss(net, classifier, z, labels, 4)
        penalized, _ = classification_rollout_loss(
            net, classifier, z, labels, 4, displacement_penalty=1.0, velocity_penalty=1.0
        )
        return (penalized - plain).item()

    assert penalty_term(features * scale) == pytest.approx(penalty_term(features), rel=1e-4)


def test_gradients_reach_the_flow_but_never_the_classifier():
    classifier = _frozen_classifier()
    net = build_near_identity_velocity_network(FEATURE_DIM, [32, 32])
    features, labels = _synthetic_split(32)

    loss, _ = classification_rollout_loss(net, classifier, features, labels, 4)
    loss.backward()

    assert classifier.linear.weight.grad is None
    assert classifier.linear.bias.grad is None
    assert net.net[-1].weight.grad.abs().sum() > 0


# --- evaluate_pipeline ---


def test_evaluate_at_initialization_matches_the_frozen_classifier():
    classifier = _frozen_classifier()
    net = build_near_identity_velocity_network(FEATURE_DIM, [32, 32])
    features, labels = _synthetic_split(64)

    loss, accuracy, displacement = evaluate_pipeline(
        net, classifier, features, labels, 4, CPU
    )

    with torch.no_grad():
        logits = classifier(features)
        expected_loss = nn.functional.cross_entropy(logits, labels).item()
        expected_accuracy = (logits.argmax(dim=1) == labels).float().mean().item()

    assert loss == pytest.approx(expected_loss)
    assert accuracy == pytest.approx(expected_accuracy)
    assert displacement == pytest.approx(0.0)


def test_evaluate_restores_the_networks_training_mode():
    # It is called from inside the training loop, so it must not silently
    # leave the network in eval mode for the following epoch.
    classifier = _frozen_classifier()
    net = build_near_identity_velocity_network(FEATURE_DIM, [32, 32])
    features, labels = _synthetic_split(16)

    net.train()
    evaluate_pipeline(net, classifier, features, labels, 4, CPU)
    assert net.training

    net.eval()
    evaluate_pipeline(net, classifier, features, labels, 4, CPU)
    assert not net.training


# --- train_rolled_out_classification ---


def _train(seed=0, **overrides):
    classifier = _frozen_classifier()
    train_features, train_labels = _synthetic_split(48, seed=1)
    val_features, val_labels = _synthetic_split(32, seed=2)
    settings = dict(hidden_dims=[16, 16], num_euler_steps=2, max_epochs=4, batch_size=16)
    settings.update(overrides)
    hyperparams = Stage3Hyperparams(**settings)
    result = train_rolled_out_classification(
        train_features, train_labels, val_features, val_labels,
        classifier, NUM_CLASSES, hyperparams, seed=seed, device=CPU,
    )
    return result, classifier, (val_features, val_labels)


def test_training_returns_a_full_history_and_a_usable_checkpoint():
    result, _, _ = _train()

    assert [log.epoch for log in result.history] == [1, 2, 3, 4]
    assert 1 <= result.best_epoch <= 4
    assert result.best_val_accuracy == result.history[result.best_epoch - 1].val_accuracy

    rebuilt = build_near_identity_velocity_network(FEATURE_DIM, [16, 16])
    rebuilt.load_state_dict(result.best_state_dict)  # must not raise


def test_initial_accuracy_is_the_frozen_classifiers_own_accuracy():
    # part_3.pdf's near-identity requirement, at the level of the training
    # loop: before any update the pipeline is the linear probe.
    result, classifier, (val_features, val_labels) = _train()

    with torch.no_grad():
        expected = (
            classifier(val_features).argmax(dim=1) == val_labels
        ).float().mean().item()

    assert result.initial_val_accuracy == pytest.approx(expected)


def test_selection_ignores_the_untrained_identity_network():
    # Deliberate: including epoch 0 as a candidate would clamp every reported
    # delta at >= 0 and hide a genuinely negative result.
    result, _, _ = _train()

    assert result.best_epoch >= 1
    assert result.best_val_accuracy in [log.val_accuracy for log in result.history]


def test_the_classifier_is_never_updated():
    classifier = _frozen_classifier()
    before = classifier.linear.weight.detach().clone()

    train_features, train_labels = _synthetic_split(48, seed=1)
    val_features, val_labels = _synthetic_split(32, seed=2)
    train_rolled_out_classification(
        train_features, train_labels, val_features, val_labels,
        classifier, NUM_CLASSES,
        Stage3Hyperparams(hidden_dims=[16, 16], num_euler_steps=2, max_epochs=3),
        seed=0, device=CPU,
    )

    assert torch.equal(classifier.linear.weight, before)
    assert classifier.linear.weight.grad is None


def test_training_is_reproducible_given_a_seed():
    first, _, _ = _train(seed=3)
    second, _, _ = _train(seed=3)
    different, _, _ = _train(seed=4)

    assert [log.train_loss for log in first.history] == [
        log.train_loss for log in second.history
    ]
    assert [log.train_loss for log in first.history] != [
        log.train_loss for log in different.history
    ]


def test_a_large_displacement_penalty_keeps_the_flow_near_identity():
    unregularized, _, _ = _train(learning_rate=1e-2, displacement_penalty=0.0)
    regularized, _, _ = _train(learning_rate=1e-2, displacement_penalty=100.0)

    assert (
        regularized.history[-1].mean_displacement
        < unregularized.history[-1].mean_displacement
    )


def test_the_progress_callback_sees_every_epoch():
    seen = []
    classifier = _frozen_classifier()
    train_features, train_labels = _synthetic_split(48, seed=1)
    val_features, val_labels = _synthetic_split(32, seed=2)

    train_rolled_out_classification(
        train_features, train_labels, val_features, val_labels,
        classifier, NUM_CLASSES,
        Stage3Hyperparams(hidden_dims=[16, 16], num_euler_steps=2, max_epochs=3),
        seed=0, device=CPU, progress=seen.append,
    )

    assert [log.epoch for log in seen] == [1, 2, 3]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"train_labels": torch.zeros(3, dtype=torch.long)}, "train_labels"),
        ({"val_labels": torch.zeros(3, dtype=torch.long)}, "val_labels"),
    ],
)
def test_mismatched_label_counts_raise(kwargs, message):
    classifier = _frozen_classifier()
    train_features, train_labels = _synthetic_split(16, seed=1)
    val_features, val_labels = _synthetic_split(16, seed=2)
    arguments = {
        "train_features": train_features, "train_labels": train_labels,
        "val_features": val_features, "val_labels": val_labels,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        train_rolled_out_classification(
            **arguments, classifier=classifier, num_classes=NUM_CLASSES,
            hyperparams=Stage3Hyperparams(hidden_dims=[8], max_epochs=1),
            seed=0, device=CPU,
        )


def test_out_of_range_labels_raise():
    classifier = _frozen_classifier()
    train_features, train_labels = _synthetic_split(16, seed=1)
    val_features, val_labels = _synthetic_split(16, seed=2)
    train_labels[0] = NUM_CLASSES  # one past the last valid class

    with pytest.raises(ValueError, match="train_labels must lie in"):
        train_rolled_out_classification(
            train_features, train_labels, val_features, val_labels,
            classifier, NUM_CLASSES,
            Stage3Hyperparams(hidden_dims=[8], max_epochs=1), seed=0, device=CPU,
        )


def test_unsupported_optimizer_raises():
    classifier = _frozen_classifier()
    train_features, train_labels = _synthetic_split(16, seed=1)
    val_features, val_labels = _synthetic_split(16, seed=2)

    with pytest.raises(ValueError, match="adamw"):
        train_rolled_out_classification(
            train_features, train_labels, val_features, val_labels,
            classifier, NUM_CLASSES,
            Stage3Hyperparams(hidden_dims=[8], max_epochs=1, optimizer="sgd"),
            seed=0, device=CPU,
        )


# --- Strategy 2: classifier-guided targets (part_3.pdf) ---


def test_a_normalized_target_moves_by_one_step_and_never_further():
    # The whole point of normalizing. With M=1 the displacement is exactly
    # eta * mean||z|| regardless of how confident the classifier is - except
    # for samples whose gradient has underflowed, which move less rather than
    # being flung in a meaningless direction. Even in this 16-dimensional
    # synthetic problem a sample or two lands in that regime, which is the
    # same effect that dominates on the real K-shot subsets.
    classifier = _frozen_classifier()
    net = build_near_identity_velocity_network(FEATURE_DIM, [32, 32])
    features, labels = _synthetic_split(64)
    step_size = 0.1

    targets = build_classifier_guided_targets(
        net, classifier, features, labels, num_steps=4,
        step_size=step_size, num_target_steps=1, normalize=True,
    )

    # z_hat == z at initialization, so the move is measured from z itself.
    distances = (targets - features).norm(dim=1)
    one_step = (step_size * features.norm(dim=1).mean()).item()

    assert distances.max().item() <= one_step * (1 + 1e-4)
    at_full_step = torch.isclose(
        distances, torch.full_like(distances, one_step), rtol=1e-4
    )
    assert at_full_step.float().mean() > 0.9


def test_an_unnormalized_target_barely_moves_at_all():
    # The failure mode normalization exists to prevent. Because the frozen
    # classifier already fits these points, the raw gradient is minuscule for
    # essentially every example, so an unnormalized step leaves the target on
    # top of z_hat and trains the flow to reproduce what it already does.
    classifier = _frozen_classifier()
    net = build_near_identity_velocity_network(FEATURE_DIM, [32, 32])
    features, labels = _synthetic_split(64)
    step_size = 0.1

    def mean_distance(normalize):
        targets = build_classifier_guided_targets(
            net, classifier, features, labels, 4, step_size=step_size,
            num_target_steps=1, normalize=normalize,
        )
        return (targets - features).norm(dim=1).mean().item()

    feature_norm = features.norm(dim=1).mean().item()

    # Unnormalized: far below a thousandth of a feature norm - no useful target.
    assert mean_distance(normalize=False) < 1e-3 * feature_norm
    # Normalized: the intended step_size fraction of a feature norm.
    assert mean_distance(normalize=True) == pytest.approx(
        step_size * feature_norm, rel=0.05
    )


def test_the_target_reduces_the_frozen_classifiers_loss():
    # Steps 2-3 of the recipe: z_hat' must be a *better* representation than
    # z_hat as far as the frozen classifier is concerned.
    classifier = _frozen_classifier()
    net = build_near_identity_velocity_network(FEATURE_DIM, [32, 32])
    features, labels = _synthetic_split(64)

    targets = build_classifier_guided_targets(
        net, classifier, features, labels, 4, step_size=0.05,
        num_target_steps=1, normalize=True,
    )

    with torch.no_grad():
        before = nn.functional.cross_entropy(classifier(features), labels)
        after = nn.functional.cross_entropy(classifier(targets), labels)

    assert after.item() < before.item()


def test_more_target_steps_move_further():
    classifier = _frozen_classifier()
    net = build_near_identity_velocity_network(FEATURE_DIM, [32, 32])
    features, labels = _synthetic_split(64)

    def distance(num_target_steps):
        targets = build_classifier_guided_targets(
            net, classifier, features, labels, 4, step_size=0.05,
            num_target_steps=num_target_steps, normalize=True,
        )
        return (targets - features).norm(dim=1).mean().item()

    assert distance(3) > distance(1)


def test_a_zero_gradient_leaves_the_target_where_it_was():
    # Guards the division in the normalization step. A classifier with zero
    # weights gives every class the same logit, so the gradient with respect
    # to the feature is exactly zero - which is not far from what the real
    # frozen probe produces on a training subset it already fits perfectly.
    classifier = LinearProbe(FEATURE_DIM, NUM_CLASSES)
    nn.init.zeros_(classifier.linear.weight)
    nn.init.zeros_(classifier.linear.bias)
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)

    net = build_near_identity_velocity_network(FEATURE_DIM, [32, 32])
    features, labels = _synthetic_split(32)

    targets = build_classifier_guided_targets(
        net, classifier, features, labels, 4, step_size=0.1,
        num_target_steps=1, normalize=True,
    )

    assert torch.isfinite(targets).all()
    assert torch.allclose(targets, features)


def test_targets_are_detached_and_restore_the_training_mode():
    classifier = _frozen_classifier()
    net = build_near_identity_velocity_network(FEATURE_DIM, [32, 32])
    features, labels = _synthetic_split(16)

    net.train()
    targets = build_classifier_guided_targets(
        net, classifier, features, labels, 4, 0.1, 1, True
    )

    assert targets.grad_fn is None
    assert not targets.requires_grad
    assert net.training  # not left in eval mode for the next epoch


def test_the_fm_loss_is_zero_when_the_target_is_the_source():
    # A near-identity network predicts zero velocity, and a target equal to
    # the source asks for exactly that - so the objective starts at zero.
    net = build_near_identity_velocity_network(FEATURE_DIM, [32, 32])
    features, _ = _synthetic_split(32)

    loss = classifier_guided_fm_loss(net, features, features)

    assert loss.item() == pytest.approx(0.0)


def test_the_fm_loss_is_positive_when_the_target_differs():
    net = build_near_identity_velocity_network(FEATURE_DIM, [32, 32])
    features, _ = _synthetic_split(32)
    targets = features + 1.0

    loss = classifier_guided_fm_loss(net, features, targets)

    assert loss.item() > 0


def _train_guided(seed=0, **overrides):
    classifier = _frozen_classifier()
    train_features, train_labels = _synthetic_split(48, seed=1)
    val_features, val_labels = _synthetic_split(32, seed=2)
    settings = dict(hidden_dims=[16, 16], num_euler_steps=2, max_epochs=4, batch_size=16)
    settings.update(overrides)
    hyperparams = Stage3Hyperparams(**settings)
    result = train_classifier_guided_fm(
        train_features, train_labels, val_features, val_labels,
        classifier, NUM_CLASSES, hyperparams, seed=seed, device=CPU,
    )
    return result, classifier, (val_features, val_labels)


def test_guided_training_returns_a_full_history_and_a_usable_checkpoint():
    result, _, _ = _train_guided()

    assert [log.epoch for log in result.history] == [1, 2, 3, 4]
    assert 1 <= result.best_epoch <= 4

    rebuilt = build_near_identity_velocity_network(FEATURE_DIM, [16, 16])
    rebuilt.load_state_dict(result.best_state_dict)  # must not raise


def test_guided_training_starts_from_the_frozen_classifiers_accuracy():
    result, classifier, (val_features, val_labels) = _train_guided()

    with torch.no_grad():
        expected = (
            classifier(val_features).argmax(dim=1) == val_labels
        ).float().mean().item()

    assert result.initial_val_accuracy == pytest.approx(expected)


def test_guided_training_never_updates_the_classifier():
    classifier = _frozen_classifier()
    before = classifier.linear.weight.detach().clone()

    train_features, train_labels = _synthetic_split(48, seed=1)
    val_features, val_labels = _synthetic_split(32, seed=2)
    train_classifier_guided_fm(
        train_features, train_labels, val_features, val_labels,
        classifier, NUM_CLASSES,
        Stage3Hyperparams(hidden_dims=[16, 16], num_euler_steps=2, max_epochs=3),
        seed=0, device=CPU,
    )

    assert torch.equal(classifier.linear.weight, before)
    assert classifier.linear.weight.grad is None


def test_guided_training_is_reproducible_given_a_seed():
    first, _, _ = _train_guided(seed=3)
    second, _, _ = _train_guided(seed=3)
    different, _, _ = _train_guided(seed=4)

    assert [log.train_loss for log in first.history] == [
        log.train_loss for log in second.history
    ]
    assert [log.train_loss for log in first.history] != [
        log.train_loss for log in different.history
    ]


@pytest.mark.parametrize(
    "refresh_epochs, max_epochs, expected_refreshes", [(1, 4, 4), (2, 4, 2), (3, 7, 3)]
)
def test_targets_are_recomputed_on_the_configured_schedule(
    monkeypatch, refresh_epochs, max_epochs, expected_refreshes
):
    # Step 6 of the recipe. Counting the calls is the only way to see this
    # from outside, since a refreshed target is not distinguishable from a
    # stale one by inspection.
    calls = []
    original = stage3_training.build_classifier_guided_targets

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(stage3_training, "build_classifier_guided_targets", counting)

    _train_guided(target_refresh_epochs=refresh_epochs, max_epochs=max_epochs)

    assert len(calls) == expected_refreshes


def test_the_guided_progress_callback_sees_every_epoch():
    seen = []
    classifier = _frozen_classifier()
    train_features, train_labels = _synthetic_split(48, seed=1)
    val_features, val_labels = _synthetic_split(32, seed=2)

    train_classifier_guided_fm(
        train_features, train_labels, val_features, val_labels,
        classifier, NUM_CLASSES,
        Stage3Hyperparams(hidden_dims=[16, 16], num_euler_steps=2, max_epochs=3),
        seed=0, device=CPU, progress=seen.append,
    )

    assert [log.epoch for log in seen] == [1, 2, 3]


def test_guided_training_validates_its_inputs():
    classifier = _frozen_classifier()
    train_features, train_labels = _synthetic_split(16, seed=1)
    val_features, val_labels = _synthetic_split(16, seed=2)
    train_labels[0] = NUM_CLASSES

    with pytest.raises(ValueError, match="train_labels must lie in"):
        train_classifier_guided_fm(
            train_features, train_labels, val_features, val_labels,
            classifier, NUM_CLASSES,
            Stage3Hyperparams(hidden_dims=[8], max_epochs=1), seed=0, device=CPU,
        )


# --- The shared skeleton both strategies are required to agree on ---


def test_both_strategies_start_from_the_same_network():
    # part_3.pdf and stage_2.pdf both require the architecture and main
    # training choices to stay fixed across the compared variants. Same seed,
    # same hyperparameters: the two strategies must begin identically and
    # differ only in what they optimize.
    rolled, _, _ = _train(seed=7)
    guided, _, _ = _train_guided(seed=7)

    assert rolled.initial_val_accuracy == guided.initial_val_accuracy
    assert rolled.initial_val_loss == pytest.approx(guided.initial_val_loss)
    assert len(rolled.history) == len(guided.history)


# --- The optional extension: joint fine-tuning (part_3.pdf) ---


def _train_joint(seed=0, train_flow=True, **overrides):
    classifier = _frozen_classifier()
    train_features, train_labels = _synthetic_split(48, seed=1)
    val_features, val_labels = _synthetic_split(32, seed=2)
    settings = dict(hidden_dims=[16, 16], num_euler_steps=2, max_epochs=4, batch_size=16)
    settings.update(overrides)
    result = train_joint_finetuning(
        train_features, train_labels, val_features, val_labels,
        classifier, NUM_CLASSES, Stage3Hyperparams(**settings),
        seed=seed, device=CPU, train_flow=train_flow,
    )
    return result, classifier


def test_joint_training_returns_both_checkpoints():
    result, _ = _train_joint()

    assert result.best_classifier_state_dict is not None
    assert set(result.best_classifier_state_dict) == {"linear.weight", "linear.bias"}

    rebuilt = build_near_identity_velocity_network(FEATURE_DIM, [16, 16])
    rebuilt.load_state_dict(result.best_state_dict)  # must not raise
    LinearProbe(FEATURE_DIM, NUM_CLASSES).load_state_dict(
        result.best_classifier_state_dict
    )


def test_the_frozen_strategies_report_no_classifier_checkpoint():
    # How a caller tells the two kinds of run apart.
    frozen_result, _, _ = _train()

    assert frozen_result.best_classifier_state_dict is None


def test_joint_training_does_not_mutate_the_callers_classifier():
    # The loaded probe is still needed unmodified - it defines the baseline
    # this run is measured against, and the frozen runs share it.
    classifier = _frozen_classifier()
    before_weight = classifier.linear.weight.detach().clone()
    before_bias = classifier.linear.bias.detach().clone()

    train_features, train_labels = _synthetic_split(48, seed=1)
    val_features, val_labels = _synthetic_split(32, seed=2)
    result = train_joint_finetuning(
        train_features, train_labels, val_features, val_labels,
        classifier, NUM_CLASSES,
        Stage3Hyperparams(hidden_dims=[16, 16], num_euler_steps=2, max_epochs=4),
        seed=0, device=CPU,
    )

    assert torch.equal(classifier.linear.weight, before_weight)
    assert torch.equal(classifier.linear.bias, before_bias)
    assert all(not p.requires_grad for p in classifier.parameters())
    # ... and the copy really did move.
    assert not torch.equal(result.best_classifier_state_dict["linear.weight"], before_weight)


def test_joint_training_starts_from_the_same_place_as_a_frozen_run():
    # Only the unfreezing may differ, so the two must agree on where they began.
    frozen, _, _ = _train(seed=5)
    joint, _ = _train_joint(seed=5)

    assert joint.initial_val_accuracy == frozen.initial_val_accuracy
    assert joint.initial_val_loss == pytest.approx(frozen.initial_val_loss)


def test_the_classifier_only_control_leaves_the_flow_at_identity():
    # `cls_finetune`: the flow's learning rate is held at zero, so the
    # pipeline stays exactly the classifier and any gain is the classifier's.
    #
    # Asserted behaviourally rather than by comparing weights to a freshly
    # built network: the hidden layers are randomly initialized and a fresh
    # network draws different values. What matters is that the final layer is
    # still zero, which makes the rollout the identity whatever the hidden
    # layers hold.
    result, _ = _train_joint(train_flow=False)

    rebuilt = build_near_identity_velocity_network(FEATURE_DIM, [16, 16])
    rebuilt.load_state_dict(result.best_state_dict)
    final_layer = rebuilt.net[-1]

    assert torch.equal(final_layer.weight, torch.zeros_like(final_layer.weight))
    assert torch.equal(final_layer.bias, torch.zeros_like(final_layer.bias))

    features, _ = _synthetic_split(16, seed=9)
    with torch.no_grad():
        assert torch.equal(euler_transport(rebuilt, features, 2), features)

    assert all(log.mean_displacement == pytest.approx(0.0) for log in result.history)


def test_the_control_still_trains_the_classifier():
    result, classifier = _train_joint(train_flow=False)

    assert not torch.equal(
        result.best_classifier_state_dict["linear.weight"],
        classifier.linear.weight,
    )


def test_delayed_unfreezing_holds_the_classifier_still_at_first():
    # part_3.pdf suggests delayed unfreezing; with the flow also disabled,
    # nothing may change at all before the unfreeze epoch.
    result, classifier = _train_joint(
        train_flow=False, max_epochs=3, unfreeze_epoch=4
    )

    assert torch.equal(
        result.best_classifier_state_dict["linear.weight"], classifier.linear.weight
    )


def test_the_classifier_moves_once_its_epoch_arrives():
    early, classifier = _train_joint(train_flow=False, max_epochs=4, unfreeze_epoch=1)
    late, _ = _train_joint(train_flow=False, max_epochs=4, unfreeze_epoch=3)

    reference = classifier.linear.weight
    early_shift = (early.best_classifier_state_dict["linear.weight"] - reference).abs().sum()
    late_shift = (late.best_classifier_state_dict["linear.weight"] - reference).abs().sum()

    assert early_shift > 0
    assert late_shift >= 0


def test_the_two_learning_rates_are_applied_to_their_own_groups():
    captured = {}
    original = torch.optim.AdamW.__init__

    def spy(self, params, *args, **kwargs):
        groups = list(params)
        captured["lrs"] = [group["lr"] for group in groups]
        return original(self, groups, *args, **kwargs)

    torch.optim.AdamW.__init__ = spy
    try:
        _train_joint(learning_rate=1e-3, classifier_learning_rate=1e-5, max_epochs=1)
    finally:
        torch.optim.AdamW.__init__ = original

    assert captured["lrs"] == [1e-3, 1e-5]


def test_joint_training_is_reproducible_given_a_seed():
    first, _ = _train_joint(seed=3)
    second, _ = _train_joint(seed=3)
    different, _ = _train_joint(seed=4)

    assert [log.train_loss for log in first.history] == [
        log.train_loss for log in second.history
    ]
    assert [log.train_loss for log in first.history] != [
        log.train_loss for log in different.history
    ]

