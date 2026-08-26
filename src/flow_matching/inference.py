"""Euler integration of the learned velocity field (stage_2.pdf).

Starting from a feature z, apply T Euler steps

    z_hat_0     = z
    z_hat_{k+1} = z_hat_k + (1/T) * v_theta(z_hat_k, k/T),   k = 0, ..., T-1

and take z_hat_T as the transported feature. Note the times visited are
k/T for k in 0..T-1, i.e. 0, 1/T, ..., (T-1)/T - the field is never
evaluated at t = 1, since the final step lands on the endpoint rather than
starting from it.

The same integrator is used in three places, which is why nothing here
detaches or wraps itself in `torch.no_grad()`:
  - inference for both FM variants (caller wraps in `no_grad`);
  - rolled-out *training*, which backpropagates through the whole unroll;
  - the trajectory visualizations and per-step analyses.
"""

from typing import Dict, Sequence

import torch
import torch.nn as nn

from src.flow_matching.velocity_net import VelocityNetwork


def euler_transport(
    velocity_net: nn.Module, features: torch.Tensor, num_steps: int
) -> torch.Tensor:
    """Integrate `features` forward through `num_steps` Euler steps.

    Gradients flow through every step, so this is also the forward pass of
    rolled-out training. Callers doing pure inference should wrap the call
    in `torch.no_grad()`.

    Args:
        velocity_net: the velocity network v_theta, called as
            `velocity_net(state, time)`.
        features: (N, D) starting points z_hat_0.
        num_steps: T, the number of Euler steps.

    Returns:
        (N, D) transported features z_hat_T.

    Raises:
        ValueError: if num_steps is not at least 1.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be at least 1, got {num_steps}")

    state = features
    for step in range(num_steps):
        state = state + velocity_net(state, step / num_steps) / num_steps
    return state


def euler_trajectory(
    velocity_net: nn.Module, features: torch.Tensor, num_steps: int
) -> torch.Tensor:
    """Integrate `features` and keep every intermediate state.

    Same integration as `euler_transport`, but returning the whole path
    rather than only its endpoint. Kept separate rather than folded into
    `euler_transport` behind a flag so the hot path (training and
    evaluation) never allocates the stacked trajectory it does not need.

    Args:
        velocity_net: the velocity network v_theta.
        features: (N, D) starting points z_hat_0.
        num_steps: T, the number of Euler steps.

    Returns:
        (T+1, N, D) tensor of states, where index 0 is the original feature
        and index T is the final transported feature. The leading dimension
        is T+1, not T, because both endpoints are included.

    Raises:
        ValueError: if num_steps is not at least 1.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be at least 1, got {num_steps}")

    state = features
    states = [state]
    for step in range(num_steps):
        state = state + velocity_net(state, step / num_steps) / num_steps
        states.append(state)
    return torch.stack(states)


def transport_with_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    hidden_dims: Sequence[int],
    features: torch.Tensor,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Rebuild a trained velocity network from a checkpoint and transport features.

    A convenience wrapper over `euler_transport` for every caller that has a
    saved run rather than a live model: test-set evaluation in the experiment
    runner, and the feature-space and per-step visualizations. Always runs
    under `no_grad` in eval mode, and returns a CPU tensor, since none of
    those callers want gradients or device-resident results.

    Args:
        state_dict: weights saved by a flow-matching training run.
        hidden_dims: the hidden widths the network was built with; must match
            the checkpoint.
        features: (N, D) starting points, L2-normalized by the caller.
        num_steps: T, the number of Euler steps.
        device: device to integrate on.

    Returns:
        (N, D) transported features, on CPU.
    """
    model = VelocityNetwork(features.shape[1], hidden_dims).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        return euler_transport(model, features.to(device), num_steps).cpu()


def trajectory_with_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    hidden_dims: Sequence[int],
    features: torch.Tensor,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Same as `transport_with_checkpoint`, but keeping every intermediate state.

    Returns:
        (T+1, N, D) trajectory, on CPU.
    """
    model = VelocityNetwork(features.shape[1], hidden_dims).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        return euler_trajectory(model, features.to(device), num_steps).cpu()


def reverse_euler_trajectory(
    velocity_net: nn.Module, features: torch.Tensor, num_steps: int
) -> torch.Tensor:
    """Integrate the learned field *backwards*, from t=1 down to t=0.

    stage_2.pdf's optional exploration: start from the class prototypes and
    run the flow in reverse to see what the model treats as a typical member
    of each class. This is explicit Euler applied to the reversed ODE,

        z_hat_{k-1} = z_hat_k - (1/T) * v_theta(z_hat_k, k/T),   k = T, ..., 1

    so the times visited are 1, (T-1)/T, ..., 1/T - the mirror of the forward
    schedule.

    One caveat worth knowing when reading the result: the forward integrator
    never evaluates at t=1 (it stops at (T-1)/T), so the first reverse step
    asks the network about a time the forward pass never visits. For a
    standard-FM network that is harmless, since its training samples t
    continuously from U(0,1) and so covers times arbitrarily close to 1. For
    a rolled-out network it is genuine extrapolation: that network only ever
    saw the discrete times k/T for k < T. Reverse trajectories from a
    rolled-out model should be read with that in mind.

    Unlike the forward direction there is no separate endpoint-only variant:
    reverse integration is used purely for visualizing a handful of
    prototypes, never in a training or evaluation hot path, so the memory
    argument that justified splitting `euler_transport` from
    `euler_trajectory` does not apply here.

    Args:
        velocity_net: the velocity network v_theta.
        features: (N, D) starting points, normally the class prototypes.
        num_steps: T, the number of Euler steps.

    Returns:
        (T+1, N, D) states in integration order: index 0 is the starting
        point (t=1) and index T is the final backward state (t=0).

    Raises:
        ValueError: if num_steps is not at least 1.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be at least 1, got {num_steps}")

    state = features
    states = [state]
    for step in range(num_steps, 0, -1):
        state = state - velocity_net(state, step / num_steps) / num_steps
        states.append(state)
    return torch.stack(states)


def reverse_trajectory_with_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    hidden_dims: Sequence[int],
    features: torch.Tensor,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Rebuild a trained velocity network and integrate it backwards.

    Returns:
        (T+1, N, D) reverse trajectory, on CPU.
    """
    model = VelocityNetwork(features.shape[1], hidden_dims).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        return reverse_euler_trajectory(model, features.to(device), num_steps).cpu()
