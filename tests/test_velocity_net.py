import pytest
import torch
import torch.nn as nn

from src.classifiers.linear_probe import LinearProbe
from src.flow_matching.inference import euler_trajectory, euler_transport
from src.flow_matching.velocity_net import (
    VelocityNetwork,
    build_near_identity_velocity_network,
)
from src.utils.seeding import set_seed


def test_output_has_the_same_shape_as_the_input_features():
    net = VelocityNetwork(feature_dim=16, hidden_dims=[32, 32])
    features = torch.randn(7, 16)

    velocity = net(features, 0.5)

    assert velocity.shape == features.shape


def test_architecture_matches_the_spec_suggestion():
    # stage_2.pdf: 2 hidden layers of width ~512, SiLU, scalar t concatenated
    # to the input, output dimension equal to the feature dimension.
    net = VelocityNetwork(feature_dim=512, hidden_dims=[512, 512])
    linear_layers = [layer for layer in net.net if isinstance(layer, nn.Linear)]
    activations = [layer for layer in net.net if isinstance(layer, nn.SiLU)]

    assert len(linear_layers) == 3
    assert len(activations) == 2
    assert linear_layers[0].in_features == 513  # 512 features + 1 time
    assert linear_layers[-1].out_features == 512


def test_final_layer_has_no_activation():
    # The velocity is unbounded, so the network must not squash its output.
    net = VelocityNetwork(feature_dim=8, hidden_dims=[16])
    assert isinstance(net.net[-1], nn.Linear)


def test_custom_hidden_dims_are_respected():
    net = VelocityNetwork(feature_dim=8, hidden_dims=[4, 5, 6])
    linear_layers = [layer for layer in net.net if isinstance(layer, nn.Linear)]

    assert [layer.out_features for layer in linear_layers] == [4, 5, 6, 8]


@pytest.mark.parametrize(
    "time",
    [0.25, torch.tensor(0.25), torch.full((6,), 0.25), torch.full((6, 1), 0.25)],
    ids=["float", "0d_tensor", "1d_tensor", "column_tensor"],
)
def test_equivalent_time_formats_give_identical_results(time):
    set_seed(0)
    net = VelocityNetwork(feature_dim=8, hidden_dims=[16, 16])
    features = torch.randn(6, 8)

    reference = net(features, 0.25)
    assert torch.allclose(net(features, time), reference)


def test_per_sample_times_are_applied_independently():
    # Standard FM training draws a different t for every sample in the batch,
    # so row i must be evaluated at time[i] and nothing else.
    set_seed(0)
    net = VelocityNetwork(feature_dim=8, hidden_dims=[16, 16])
    features = torch.randn(3, 8)
    times = torch.tensor([0.1, 0.5, 0.9])

    batched = net(features, times)
    row_by_row = torch.cat(
        [net(features[i : i + 1], times[i].item()) for i in range(3)], dim=0
    )

    assert torch.allclose(batched, row_by_row, atol=1e-6)


def test_time_actually_changes_the_prediction():
    set_seed(0)
    net = VelocityNetwork(feature_dim=8, hidden_dims=[16, 16])
    features = torch.randn(4, 8)

    assert not torch.allclose(net(features, 0.0), net(features, 1.0))


def test_samples_do_not_influence_each_other():
    # An MLP is applied row-wise; this guards against accidentally
    # introducing a batch-coupling op (e.g. batch norm) later.
    set_seed(0)
    net = VelocityNetwork(feature_dim=8, hidden_dims=[16, 16])
    features = torch.randn(5, 8)

    full_batch = net(features, 0.3)
    single = net(features[2:3], 0.3)

    assert torch.allclose(full_batch[2:3], single, atol=1e-6)


def test_gradients_reach_every_parameter():
    net = VelocityNetwork(feature_dim=8, hidden_dims=[16, 16])
    features = torch.randn(4, 8)

    net(features, 0.5).pow(2).sum().backward()

    for name, parameter in net.named_parameters():
        assert parameter.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} has non-finite gradients"


def test_initialization_is_reproducible_given_a_seed():
    set_seed(0)
    first = VelocityNetwork(feature_dim=8, hidden_dims=[16, 16])
    set_seed(0)
    second = VelocityNetwork(feature_dim=8, hidden_dims=[16, 16])

    for parameter_a, parameter_b in zip(first.parameters(), second.parameters()):
        assert torch.equal(parameter_a, parameter_b)


def test_different_seeds_give_different_initializations():
    # K=full FM runs rely on this being real stochasticity, not a no-op.
    set_seed(0)
    first = VelocityNetwork(feature_dim=8, hidden_dims=[16, 16])
    set_seed(1)
    second = VelocityNetwork(feature_dim=8, hidden_dims=[16, 16])

    assert not all(
        torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters())
    )


def test_state_dict_round_trips():
    set_seed(0)
    net = VelocityNetwork(feature_dim=8, hidden_dims=[16, 16])
    features = torch.randn(4, 8)
    expected = net(features, 0.4)

    reloaded = VelocityNetwork(feature_dim=8, hidden_dims=[16, 16])
    reloaded.load_state_dict(net.state_dict())

    assert torch.allclose(reloaded(features, 0.4), expected)


@pytest.mark.parametrize("feature_dim", [0, -4])
def test_non_positive_feature_dim_raises(feature_dim):
    with pytest.raises(ValueError, match="feature_dim"):
        VelocityNetwork(feature_dim=feature_dim)


def test_empty_hidden_dims_raises():
    with pytest.raises(ValueError, match="hidden_dims"):
        VelocityNetwork(feature_dim=8, hidden_dims=[])


def test_non_positive_hidden_dim_raises():
    with pytest.raises(ValueError, match="hidden_dims"):
        VelocityNetwork(feature_dim=8, hidden_dims=[16, -1])


def test_wrong_feature_dimension_raises():
    net = VelocityNetwork(feature_dim=8, hidden_dims=[16])
    with pytest.raises(ValueError, match="dimension"):
        net(torch.randn(4, 9), 0.5)


def test_non_2d_features_raise():
    net = VelocityNetwork(feature_dim=8, hidden_dims=[16])
    with pytest.raises(ValueError, match="2-D"):
        net(torch.randn(8), 0.5)


def test_time_length_mismatch_raises():
    net = VelocityNetwork(feature_dim=8, hidden_dims=[16])
    with pytest.raises(ValueError, match="time"):
        net(torch.randn(4, 8), torch.rand(3))


# --- Stage 3: near-identity initialization (part_3.pdf) ---


def test_near_identity_network_outputs_exactly_zero():
    net = build_near_identity_velocity_network(feature_dim=16, hidden_dims=[32, 32])
    features = torch.randn(9, 16) * 40.0  # raw-scale features, as Stage 3 uses

    for time in (0.0, 0.25, 0.5, 0.999):
        assert torch.equal(net(features, time), torch.zeros_like(features))


def test_near_identity_network_keeps_the_architecture_unchanged():
    # part_3.pdf requires the *same* velocity-network design as Stage 2;
    # only the initialization may differ.
    plain = VelocityNetwork(feature_dim=32, hidden_dims=[64, 64])
    near_identity = build_near_identity_velocity_network(
        feature_dim=32, hidden_dims=[64, 64]
    )

    assert type(near_identity) is VelocityNetwork
    assert [type(layer) for layer in near_identity.net] == [
        type(layer) for layer in plain.net
    ]
    assert near_identity.state_dict().keys() == plain.state_dict().keys()


@pytest.mark.parametrize("num_steps", [1, 2, 4, 12])
def test_euler_rollout_is_the_identity_at_initialization(num_steps):
    # The property part_3.pdf actually asks for, stated directly.
    net = build_near_identity_velocity_network(feature_dim=24, hidden_dims=[32, 32])
    features = torch.randn(11, 24) * 47.0

    with torch.no_grad():
        transported = euler_transport(net, features, num_steps)

    assert torch.equal(transported, features)


def test_every_intermediate_state_is_the_original_feature():
    # Not just the endpoint: nothing moves at any point along the rollout.
    net = build_near_identity_velocity_network(feature_dim=8, hidden_dims=[16, 16])
    features = torch.randn(5, 8) * 24.0

    with torch.no_grad():
        trajectory = euler_trajectory(net, features, num_steps=4)

    assert trajectory.shape == (5, 5, 8)
    for state in trajectory:
        assert torch.equal(state, features)


def test_only_the_final_layer_is_zeroed():
    # A network zeroed throughout would be permanently stuck; the hidden
    # layers must keep their ordinary initialization.
    net = build_near_identity_velocity_network(feature_dim=16, hidden_dims=[32, 32])
    linear_layers = [layer for layer in net.net if isinstance(layer, nn.Linear)]

    for layer in linear_layers[:-1]:
        assert layer.weight.abs().sum() > 0

    assert torch.equal(linear_layers[-1].weight, torch.zeros_like(linear_layers[-1].weight))
    assert torch.equal(linear_layers[-1].bias, torch.zeros_like(linear_layers[-1].bias))


def test_the_complete_pipeline_reproduces_the_frozen_classifier_exactly():
    # part_3.pdf's actual requirement: before Stage 3 training, the whole
    # system z -> FM -> frozen classifier must behave like the linear probe.
    set_seed(0)
    classifier = LinearProbe(feature_dim=16, num_classes=5)
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)

    net = build_near_identity_velocity_network(feature_dim=16, hidden_dims=[32, 32])
    features = torch.randn(20, 16) * 24.0

    with torch.no_grad():
        baseline_logits = classifier(features)
        pipeline_logits = classifier(euler_transport(net, features, num_steps=4))

    assert torch.equal(pipeline_logits, baseline_logits)


def test_the_network_starts_training_immediately():
    # The obvious worry about a zeroed output layer is that it cannot learn.
    # It can: the final layer gets a real gradient at step 0, and every layer
    # trains from step 1 onward.
    set_seed(0)
    net = build_near_identity_velocity_network(feature_dim=16, hidden_dims=[32, 32])
    classifier = LinearProbe(feature_dim=16, num_classes=5)
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)

    features = torch.randn(12, 16) * 24.0
    labels = torch.randint(0, 5, (12,))
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)

    first_layer = net.net[0]
    final_layer = net.net[-1]
    first_layer_grads = []

    for _ in range(2):
        loss = nn.functional.cross_entropy(
            classifier(euler_transport(net, features, num_steps=4)), labels
        )
        optimizer.zero_grad()
        loss.backward()
        first_layer_grads.append(first_layer.weight.grad.abs().sum().item())
        assert final_layer.weight.grad.abs().sum() > 0  # always a real gradient
        optimizer.step()

    assert first_layer_grads[0] == 0.0  # zeroed weight blocks the backward pass once
    assert first_layer_grads[1] > 0.0  # and only once

    with torch.no_grad():
        moved = euler_transport(net, features, num_steps=4)
    assert not torch.equal(moved, features)


def test_stage_2_networks_are_not_near_identity():
    # Guards the reason a separate builder exists: Stage 2's default
    # initialization perturbs features noticeably before any training, so it
    # could not have been reused for Stage 3.
    set_seed(0)
    net = VelocityNetwork(feature_dim=384, hidden_dims=[512, 512])
    features = torch.randn(32, 384) * 47.0

    with torch.no_grad():
        displacement = (euler_transport(net, features, 4) - features).norm(dim=1).mean()

    assert displacement > 0

