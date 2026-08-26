import pytest
import torch
import torch.nn as nn

from src.flow_matching.inference import euler_trajectory, euler_transport
from src.flow_matching.velocity_net import VelocityNetwork
from src.utils.seeding import set_seed


class ConstantVelocity(nn.Module):
    """Returns the same velocity everywhere, ignoring state and time.

    Makes the integrator's arithmetic analytically checkable: T steps of
    size v/T must sum to exactly v, whatever T is.
    """

    def __init__(self, velocity: torch.Tensor):
        super().__init__()
        self.velocity = velocity

    def forward(self, features, time):
        return self.velocity.expand_as(features)


class TowardTarget(nn.Module):
    """The ideal constant FM velocity for a fixed start: u = target - start."""

    def __init__(self, start: torch.Tensor, target: torch.Tensor):
        super().__init__()
        self.displacement = target - start

    def forward(self, features, time):
        return self.displacement.expand_as(features)


class RecordingVelocity(nn.Module):
    """Records the times it was evaluated at; returns zero velocity."""

    def __init__(self):
        super().__init__()
        self.seen_times = []

    def forward(self, features, time):
        self.seen_times.append(float(time))
        return torch.zeros_like(features)


@pytest.mark.parametrize("num_steps", [1, 4, 12])
def test_constant_velocity_integrates_to_exactly_one_displacement(num_steps):
    # T steps of size v/T sum to v, independent of T.
    velocity = torch.tensor([[2.0, -3.0]])
    net = ConstantVelocity(velocity)
    features = torch.tensor([[1.0, 1.0], [0.0, 5.0]])

    transported = euler_transport(net, features, num_steps)

    assert torch.allclose(transported, features + velocity)


def test_ideal_velocity_field_lands_on_the_prototype():
    # Sanity check on the whole formulation: a network that has perfectly
    # learned u_i = p - z_i transports z_i exactly onto p.
    start = torch.tensor([[0.3, -0.9, 0.5]])
    prototype = torch.tensor([[1.0, 0.0, 0.0]])
    net = TowardTarget(start, prototype)

    assert torch.allclose(euler_transport(net, start, 4), prototype, atol=1e-6)
    assert torch.allclose(euler_transport(net, start, 12), prototype, atol=1e-6)


def test_zero_velocity_leaves_features_untouched():
    net = ConstantVelocity(torch.zeros(1, 3))
    features = torch.randn(5, 3)

    assert torch.allclose(euler_transport(net, features, 12), features)


def test_times_visited_follow_the_spec_schedule():
    # stage_2.pdf: v_theta(z_hat_k, k/T) for k = 0..T-1, so t = 1 is never
    # evaluated - the last step starts at (T-1)/T and lands on the endpoint.
    net = RecordingVelocity()
    euler_transport(net, torch.randn(2, 3), num_steps=4)

    assert net.seen_times == [0.0, 0.25, 0.5, 0.75]


def test_trajectory_has_one_more_state_than_steps():
    set_seed(0)
    net = VelocityNetwork(feature_dim=6, hidden_dims=[8])
    features = torch.randn(5, 6)

    trajectory = euler_trajectory(net, features, num_steps=4)

    assert trajectory.shape == (5, 5, 6)  # (T+1, N, D) with T=4, N=5, D=6


def test_trajectory_starts_at_the_original_feature():
    set_seed(0)
    net = VelocityNetwork(feature_dim=6, hidden_dims=[8])
    features = torch.randn(4, 6)

    trajectory = euler_trajectory(net, features, num_steps=12)

    assert torch.equal(trajectory[0], features)


def test_trajectory_endpoint_equals_euler_transport():
    # The visualizations and the reported accuracy must describe the same
    # integration, so these two entry points cannot drift apart.
    set_seed(0)
    net = VelocityNetwork(feature_dim=6, hidden_dims=[8, 8])
    features = torch.randn(7, 6)

    for num_steps in (4, 12):
        trajectory = euler_trajectory(net, features, num_steps)
        transported = euler_transport(net, features, num_steps)
        assert torch.allclose(trajectory[-1], transported, atol=1e-6)


def test_trajectory_steps_are_evenly_spaced_under_constant_velocity():
    velocity = torch.tensor([[4.0]])
    net = ConstantVelocity(velocity)
    features = torch.zeros(1, 1)

    trajectory = euler_trajectory(net, features, num_steps=4)

    assert torch.allclose(trajectory.squeeze(), torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]))


def test_single_step_is_one_full_velocity_application():
    set_seed(0)
    net = VelocityNetwork(feature_dim=4, hidden_dims=[8])
    features = torch.randn(3, 4)

    with torch.no_grad():
        expected = features + net(features, 0.0)
        assert torch.allclose(euler_transport(net, features, 1), expected)


def test_gradients_flow_back_through_every_step():
    # Rolled-out training backpropagates through the complete unroll, so the
    # integrator must not detach anything.
    set_seed(0)
    net = VelocityNetwork(feature_dim=4, hidden_dims=[8])
    features = torch.randn(3, 4)

    euler_transport(net, features, num_steps=12).pow(2).sum().backward()

    for name, parameter in net.named_parameters():
        assert parameter.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} has non-finite gradients"
        assert parameter.grad.abs().sum() > 0, f"{name} has an all-zero gradient"


def test_step_count_changes_the_gradient():
    # Guards against an implementation that silently unrolls a fixed number
    # of steps: T=4 and T=12 must produce genuinely different graphs.
    def gradient_for(num_steps):
        set_seed(0)
        net = VelocityNetwork(feature_dim=4, hidden_dims=[8])
        features = torch.ones(2, 4)
        euler_transport(net, features, num_steps).pow(2).sum().backward()
        return net.net[0].weight.grad.clone()

    assert not torch.allclose(gradient_for(4), gradient_for(12))


def test_input_features_are_not_modified_in_place():
    set_seed(0)
    net = VelocityNetwork(feature_dim=4, hidden_dims=[8])
    features = torch.randn(3, 4)
    original = features.clone()

    euler_transport(net, features, num_steps=4)
    euler_trajectory(net, features, num_steps=4)

    assert torch.equal(features, original)


def test_integration_is_deterministic():
    set_seed(0)
    net = VelocityNetwork(feature_dim=4, hidden_dims=[8])
    features = torch.randn(3, 4)

    with torch.no_grad():
        first = euler_transport(net, features, 12)
        second = euler_transport(net, features, 12)

    assert torch.equal(first, second)


@pytest.mark.parametrize("num_steps", [0, -1])
def test_non_positive_step_count_raises(num_steps):
    net = ConstantVelocity(torch.zeros(1, 3))
    features = torch.randn(2, 3)

    with pytest.raises(ValueError, match="num_steps"):
        euler_transport(net, features, num_steps)
    with pytest.raises(ValueError, match="num_steps"):
        euler_trajectory(net, features, num_steps)
