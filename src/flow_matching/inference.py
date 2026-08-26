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
