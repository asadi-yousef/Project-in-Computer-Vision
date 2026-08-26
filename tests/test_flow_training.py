import math

import pytest
import torch
import torch.nn.functional as F

from src.flow_matching.inference import euler_transport
from src.flow_matching.training import (
    flow_matching_loss,
    rolled_out_loss,
    train_rolled_out_flow_matching,
    train_standard_flow_matching,
)
from src.flow_matching.velocity_net import VelocityNetwork
from src.utils.config import FlowMatchingHyperparams
from src.utils.seeding import set_seed

DEVICE = torch.device("cpu")


def make_synthetic_problem(num_classes=4, per_class=8, feature_dim=6, seed=0):
    """A small, well-separated classification problem on the unit sphere.

    Features are L2-normalized, matching what the experiment runner will
    hand the trainer.
    """
    set_seed(seed)
    centers = F.normalize(torch.randn(num_classes, feature_dim), dim=1)
    labels = torch.arange(num_classes).repeat_interleave(per_class)
    features = F.normalize(centers[labels] + 0.15 * torch.randn(len(labels), feature_dim), dim=1)
    prototypes = F.normalize(
        torch.stack([features[labels == c].mean(dim=0) for c in range(num_classes)]), dim=1
    )
    return features, labels, prototypes


def quick_hyperparams(**overrides):
    defaults = dict(hidden_dims=[32, 32], max_epochs=15, batch_size=16, learning_rate=1e-2)
    defaults.update(overrides)
    return FlowMatchingHyperparams(**defaults)


# --- the loss function itself ---


def test_loss_is_the_mean_of_per_sample_squared_norms():
    predicted = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0], [3.0, 4.0]])

    # per-sample squared norms are 1+4=5 and 9+16=25, mean 15
    assert flow_matching_loss(predicted, target).item() == pytest.approx(15.0)


def test_loss_is_zero_for_a_perfect_prediction():
    velocity = torch.randn(5, 4)
    assert flow_matching_loss(velocity, velocity).item() == pytest.approx(0.0)


def test_loss_sums_over_features_rather_than_averaging_over_them():
    # Guards the spec-literal reduction: nn.MSELoss would divide by D too.
    predicted = torch.ones(1, 10)
    target = torch.zeros(1, 10)

    assert flow_matching_loss(predicted, target).item() == pytest.approx(10.0)


# --- training behaviour ---


def test_history_has_one_entry_per_epoch():
    features, labels, prototypes = make_synthetic_problem()
    hyperparams = quick_hyperparams(max_epochs=7)

    result = train_standard_flow_matching(
        features, labels, prototypes, hyperparams, seed=0, device=DEVICE
    )

    assert len(result.history) == 7
    assert [entry.epoch for entry in result.history] == list(range(1, 8))


def test_training_loss_decreases():
    features, labels, prototypes = make_synthetic_problem()
    hyperparams = quick_hyperparams(max_epochs=40)

    result = train_standard_flow_matching(
        features, labels, prototypes, hyperparams, seed=0, device=DEVICE
    )

    first_losses = [entry.train_loss for entry in result.history[:5]]
    last_losses = [entry.train_loss for entry in result.history[-5:]]
    assert sum(last_losses) / 5 < sum(first_losses) / 5


def test_all_logged_losses_are_finite():
    features, labels, prototypes = make_synthetic_problem()
    result = train_standard_flow_matching(
        features, labels, prototypes, quick_hyperparams(), seed=0, device=DEVICE,
        val_features=features, val_labels=labels,
    )

    for entry in result.history:
        assert math.isfinite(entry.train_loss)
        assert math.isfinite(entry.val_loss)


def test_trained_network_transports_features_toward_their_prototypes():
    # The end-to-end point of the whole stage: after training, integrating a
    # feature must move it closer to its own class prototype.
    features, labels, prototypes = make_synthetic_problem()
    hyperparams = quick_hyperparams(max_epochs=60)

    result = train_standard_flow_matching(
        features, labels, prototypes, hyperparams, seed=0, device=DEVICE
    )
    model = VelocityNetwork(features.shape[1], hyperparams.hidden_dims)
    model.load_state_dict(result.final_state_dict)
    model.eval()

    targets = prototypes[labels]
    with torch.no_grad():
        transported = euler_transport(model, features, num_steps=4)

    distance_before = (features - targets).norm(dim=1).mean()
    distance_after = (transported - targets).norm(dim=1).mean()
    assert distance_after < distance_before


def test_returns_the_final_weights_not_a_best_checkpoint():
    # This project trains for a fixed budget with no validation-based
    # selection, so the returned weights must be the last epoch's.
    features, labels, prototypes = make_synthetic_problem()
    hyperparams = quick_hyperparams(max_epochs=5)

    short = train_standard_flow_matching(
        features, labels, prototypes, hyperparams, seed=0, device=DEVICE
    )
    longer = train_standard_flow_matching(
        features, labels, prototypes, quick_hyperparams(max_epochs=10), seed=0, device=DEVICE
    )

    differs = any(
        not torch.allclose(short.final_state_dict[k], longer.final_state_dict[k])
        for k in short.final_state_dict
    )
    assert differs


def test_state_dict_loads_into_a_matching_velocity_network():
    features, labels, prototypes = make_synthetic_problem()
    hyperparams = quick_hyperparams()

    result = train_standard_flow_matching(
        features, labels, prototypes, hyperparams, seed=0, device=DEVICE
    )
    model = VelocityNetwork(features.shape[1], hyperparams.hidden_dims)
    model.load_state_dict(result.final_state_dict)  # must not raise


# --- reproducibility ---


def test_same_seed_reproduces_the_run_exactly():
    features, labels, prototypes = make_synthetic_problem()

    first = train_standard_flow_matching(
        features, labels, prototypes, quick_hyperparams(), seed=0, device=DEVICE
    )
    second = train_standard_flow_matching(
        features, labels, prototypes, quick_hyperparams(), seed=0, device=DEVICE
    )

    assert [e.train_loss for e in first.history] == [e.train_loss for e in second.history]
    for key in first.final_state_dict:
        assert torch.equal(first.final_state_dict[key], second.final_state_dict[key])


def test_different_seeds_give_different_runs():
    features, labels, prototypes = make_synthetic_problem()

    first = train_standard_flow_matching(
        features, labels, prototypes, quick_hyperparams(), seed=0, device=DEVICE
    )
    second = train_standard_flow_matching(
        features, labels, prototypes, quick_hyperparams(), seed=1, device=DEVICE
    )

    assert [e.train_loss for e in first.history] != [e.train_loss for e in second.history]


# --- validation logging ---


def test_validation_loss_is_logged_when_a_split_is_given():
    features, labels, prototypes = make_synthetic_problem()

    result = train_standard_flow_matching(
        features, labels, prototypes, quick_hyperparams(), seed=0, device=DEVICE,
        val_features=features, val_labels=labels,
    )

    assert all(math.isfinite(entry.val_loss) for entry in result.history)


def test_validation_loss_is_nan_when_no_split_is_given():
    features, labels, prototypes = make_synthetic_problem()

    result = train_standard_flow_matching(
        features, labels, prototypes, quick_hyperparams(), seed=0, device=DEVICE
    )

    assert all(math.isnan(entry.val_loss) for entry in result.history)


def test_validation_times_are_fixed_across_epochs():
    # A freshly drawn t each epoch would make the val curve jitter for
    # reasons unrelated to the model. With a frozen model the val loss must
    # therefore be identical every epoch; here we check the weaker but
    # sufficient property that an untrained net (lr=0) gives a flat curve.
    features, labels, prototypes = make_synthetic_problem()
    hyperparams = quick_hyperparams(max_epochs=6, learning_rate=0.0)

    result = train_standard_flow_matching(
        features, labels, prototypes, hyperparams, seed=0, device=DEVICE,
        val_features=features, val_labels=labels,
    )

    val_losses = [entry.val_loss for entry in result.history]
    assert all(loss == pytest.approx(val_losses[0]) for loss in val_losses)


def test_omitting_validation_does_not_change_training():
    # The validation draw uses its own generator, so it must not perturb the
    # RNG stream driving initialization and shuffling.
    features, labels, prototypes = make_synthetic_problem()

    without = train_standard_flow_matching(
        features, labels, prototypes, quick_hyperparams(), seed=0, device=DEVICE
    )
    with_validation = train_standard_flow_matching(
        features, labels, prototypes, quick_hyperparams(), seed=0, device=DEVICE,
        val_features=features, val_labels=labels,
    )

    assert [e.train_loss for e in without.history] == [
        e.train_loss for e in with_validation.history
    ]


# --- T independence ---


def test_num_euler_steps_does_not_affect_standard_training():
    # Standard FM never discretizes the path, so one trained network serves
    # every T. This is what lets the runner train once and evaluate at both.
    features, labels, prototypes = make_synthetic_problem()

    t4 = train_standard_flow_matching(
        features, labels, prototypes, quick_hyperparams(num_euler_steps=4),
        seed=0, device=DEVICE,
    )
    t12 = train_standard_flow_matching(
        features, labels, prototypes, quick_hyperparams(num_euler_steps=12),
        seed=0, device=DEVICE,
    )

    for key in t4.final_state_dict:
        assert torch.equal(t4.final_state_dict[key], t12.final_state_dict[key])


# --- input validation ---


def test_unsupported_optimizer_raises():
    features, labels, prototypes = make_synthetic_problem()
    with pytest.raises(ValueError, match="adamw"):
        train_standard_flow_matching(
            features, labels, prototypes, quick_hyperparams(optimizer="sgd"),
            seed=0, device=DEVICE,
        )


def test_label_count_mismatch_raises():
    features, labels, prototypes = make_synthetic_problem()
    with pytest.raises(ValueError, match="rows"):
        train_standard_flow_matching(
            features, labels[:-1], prototypes, quick_hyperparams(), seed=0, device=DEVICE
        )


def test_prototype_dimension_mismatch_raises():
    features, labels, prototypes = make_synthetic_problem()
    with pytest.raises(ValueError, match="dimension"):
        train_standard_flow_matching(
            features, labels, prototypes[:, :-1], quick_hyperparams(), seed=0, device=DEVICE
        )


def test_label_outside_the_prototype_range_raises():
    features, labels, prototypes = make_synthetic_problem()
    with pytest.raises(ValueError, match="index"):
        train_standard_flow_matching(
            features, labels, prototypes[:2], quick_hyperparams(), seed=0, device=DEVICE
        )


# --- rolled-out training (Task 5) ---


def test_rolled_out_loss_is_the_mean_of_per_sample_squared_distances():
    transported = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    prototypes = torch.tensor([[0.0, 0.0], [3.0, 4.0]])

    assert rolled_out_loss(transported, prototypes).item() == pytest.approx(15.0)


def test_rolled_out_loss_is_zero_when_the_endpoint_is_the_prototype():
    prototypes = torch.randn(5, 4)
    assert rolled_out_loss(prototypes, prototypes).item() == pytest.approx(0.0)


def test_rolled_out_history_has_one_entry_per_epoch():
    features, labels, prototypes = make_synthetic_problem()

    result = train_rolled_out_flow_matching(
        features, labels, prototypes, quick_hyperparams(max_epochs=7), seed=0, device=DEVICE
    )

    assert len(result.history) == 7


def test_rolled_out_training_loss_decreases():
    features, labels, prototypes = make_synthetic_problem()

    result = train_rolled_out_flow_matching(
        features, labels, prototypes, quick_hyperparams(max_epochs=40), seed=0, device=DEVICE
    )

    first = sum(e.train_loss for e in result.history[:5]) / 5
    last = sum(e.train_loss for e in result.history[-5:]) / 5
    assert last < first


def test_rolled_out_training_drives_the_endpoint_onto_the_prototype():
    # The objective *is* the endpoint distance, so after training the
    # transported features must sit very close to their prototypes.
    features, labels, prototypes = make_synthetic_problem()
    hyperparams = quick_hyperparams(max_epochs=60)

    result = train_rolled_out_flow_matching(
        features, labels, prototypes, hyperparams, seed=0, device=DEVICE
    )
    model = VelocityNetwork(features.shape[1], hyperparams.hidden_dims)
    model.load_state_dict(result.final_state_dict)
    model.eval()

    targets = prototypes[labels]
    with torch.no_grad():
        transported = euler_transport(model, features, hyperparams.num_euler_steps)

    before = (features - targets).norm(dim=1).mean()
    after = (transported - targets).norm(dim=1).mean()
    assert after < before / 2


def test_rolled_out_final_train_loss_matches_its_own_objective():
    # The logged loss must be the endpoint distance actually achieved, not
    # some other quantity: recompute it independently and compare.
    features, labels, prototypes = make_synthetic_problem()
    hyperparams = quick_hyperparams(max_epochs=30)

    result = train_rolled_out_flow_matching(
        features, labels, prototypes, hyperparams, seed=0, device=DEVICE,
        val_features=features, val_labels=labels,
    )
    model = VelocityNetwork(features.shape[1], hyperparams.hidden_dims)
    model.load_state_dict(result.final_state_dict)
    model.eval()

    with torch.no_grad():
        transported = euler_transport(model, features, hyperparams.num_euler_steps)
        recomputed = rolled_out_loss(transported, prototypes[labels]).item()

    assert result.history[-1].val_loss == pytest.approx(recomputed, rel=1e-4)


def test_rolled_out_depends_on_the_step_count():
    # T is baked into the learned weights, unlike standard FM.
    features, labels, prototypes = make_synthetic_problem()

    t4 = train_rolled_out_flow_matching(
        features, labels, prototypes, quick_hyperparams(num_euler_steps=4),
        seed=0, device=DEVICE,
    )
    t12 = train_rolled_out_flow_matching(
        features, labels, prototypes, quick_hyperparams(num_euler_steps=12),
        seed=0, device=DEVICE,
    )

    differs = any(
        not torch.equal(t4.final_state_dict[k], t12.final_state_dict[k])
        for k in t4.final_state_dict
    )
    assert differs


def test_rolled_out_is_reproducible_given_a_seed():
    features, labels, prototypes = make_synthetic_problem()

    first = train_rolled_out_flow_matching(
        features, labels, prototypes, quick_hyperparams(), seed=0, device=DEVICE
    )
    second = train_rolled_out_flow_matching(
        features, labels, prototypes, quick_hyperparams(), seed=0, device=DEVICE
    )

    assert [e.train_loss for e in first.history] == [e.train_loss for e in second.history]
    for key in first.final_state_dict:
        assert torch.equal(first.final_state_dict[key], second.final_state_dict[key])


def test_rolled_out_validation_loss_is_deterministic_across_epochs_when_frozen():
    # Unlike standard FM, the rolled-out objective samples no times at all,
    # so a frozen model must give exactly the same val loss every epoch.
    features, labels, prototypes = make_synthetic_problem()

    result = train_rolled_out_flow_matching(
        features, labels, prototypes, quick_hyperparams(max_epochs=6, learning_rate=0.0),
        seed=0, device=DEVICE, val_features=features, val_labels=labels,
    )

    val_losses = [e.val_loss for e in result.history]
    assert all(loss == pytest.approx(val_losses[0]) for loss in val_losses)


def test_rolled_out_all_losses_finite():
    features, labels, prototypes = make_synthetic_problem()

    result = train_rolled_out_flow_matching(
        features, labels, prototypes, quick_hyperparams(num_euler_steps=12),
        seed=0, device=DEVICE, val_features=features, val_labels=labels,
    )

    for entry in result.history:
        assert math.isfinite(entry.train_loss)
        assert math.isfinite(entry.val_loss)


def test_both_variants_start_from_identical_weights():
    # stage_2.pdf requires the architecture and main training choices stay
    # fixed across the comparison; same seed must mean same initialization.
    features, labels, prototypes = make_synthetic_problem()
    hyperparams = quick_hyperparams(max_epochs=0)

    standard = train_standard_flow_matching(
        features, labels, prototypes, hyperparams, seed=0, device=DEVICE
    )
    rolled = train_rolled_out_flow_matching(
        features, labels, prototypes, hyperparams, seed=0, device=DEVICE
    )

    for key in standard.final_state_dict:
        assert torch.equal(standard.final_state_dict[key], rolled.final_state_dict[key])


def test_rolled_out_unsupported_optimizer_raises():
    features, labels, prototypes = make_synthetic_problem()
    with pytest.raises(ValueError, match="adamw"):
        train_rolled_out_flow_matching(
            features, labels, prototypes, quick_hyperparams(optimizer="sgd"),
            seed=0, device=DEVICE,
        )


def test_rolled_out_label_count_mismatch_raises():
    features, labels, prototypes = make_synthetic_problem()
    with pytest.raises(ValueError, match="rows"):
        train_rolled_out_flow_matching(
            features, labels[:-1], prototypes, quick_hyperparams(), seed=0, device=DEVICE
        )
