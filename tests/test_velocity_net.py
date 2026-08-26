import pytest
import torch
import torch.nn as nn

from src.flow_matching.velocity_net import VelocityNetwork
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
