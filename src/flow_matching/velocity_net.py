"""The velocity network v_theta(z, t) used by both Stage 2 flow-matching
variants (stage_2.pdf).

Architecture follows the spec's suggestion: a small MLP with two hidden
layers of width ~512 and SiLU activations, the scalar time t concatenated to
the input feature, and output dimension equal to the feature dimension. The
output is an unbounded velocity vector, so the final layer has no activation.

The same network definition serves standard FM and rolled-out training - only
the training objective differs between them - and the same forward pass is
used at inference by the Euler integrator.

Stage 3 (part_3.pdf) reuses this exact architecture, as its spec requires,
but needs a different initialization: see
`build_near_identity_velocity_network` at the bottom of this module.
"""

from typing import Sequence, Union

import torch
import torch.nn as nn


def _as_time_column(
    time: Union[float, torch.Tensor], num_samples: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Normalize the many shapes `t` legitimately arrives in into (N, 1).

    Both call sites are supported deliberately:
      - standard FM training samples a *different* t per sample, arriving as
        a (N,) or (N, 1) tensor;
      - Euler integration evaluates the *whole batch* at one time k/T,
        arriving as a Python float or a 0-dim tensor.

    Args:
        time: scalar (float or 0-dim tensor) broadcast to every sample, or a
            per-sample tensor of shape (N,) or (N, 1).
        num_samples: N, the batch size the result must match.
        device, dtype: of the feature tensor `time` will be concatenated to.

    Returns:
        (N, 1) tensor of times.

    Raises:
        ValueError: if a per-sample tensor's length does not match
            `num_samples`, or it has more than 2 dimensions.
    """
    if not isinstance(time, torch.Tensor):
        time = torch.tensor(time, device=device, dtype=dtype)

    time = time.to(device=device, dtype=dtype)

    if time.dim() == 0:
        return time.expand(num_samples).unsqueeze(1)
    if time.dim() == 1:
        if time.shape[0] != num_samples:
            raise ValueError(
                f"time has {time.shape[0]} entries but there are {num_samples} samples"
            )
        return time.unsqueeze(1)
    if time.dim() == 2:
        if time.shape != (num_samples, 1):
            raise ValueError(
                f"2-D time must have shape ({num_samples}, 1), got {tuple(time.shape)}"
            )
        return time

    raise ValueError(f"time must be scalar, 1-D, or (N, 1); got {time.dim()} dimensions")


class VelocityNetwork(nn.Module):
    """v_theta(z, t): predicts a velocity in feature space at feature z, time t.

    Args:
        feature_dim: D, the frozen encoder's feature dimension. The network
            both consumes (alongside t) and produces vectors of this size.
        hidden_dims: widths of the hidden layers, in order.

    Raises:
        ValueError: if feature_dim is not positive, or hidden_dims is empty
            or contains a non-positive width.
    """

    def __init__(self, feature_dim: int, hidden_dims: Sequence[int] = (512, 512)):
        super().__init__()
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        hidden_dims = list(hidden_dims)
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one hidden layer width")
        if any(width <= 0 for width in hidden_dims):
            raise ValueError(f"hidden_dims widths must all be positive, got {hidden_dims}")

        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims

        layers: list = []
        input_dim = feature_dim + 1  # +1 for the concatenated scalar time
        for width in hidden_dims:
            layers.append(nn.Linear(input_dim, width))
            layers.append(nn.SiLU())
            input_dim = width
        layers.append(nn.Linear(input_dim, feature_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self, features: torch.Tensor, time: Union[float, torch.Tensor]
    ) -> torch.Tensor:
        """Predict the velocity at each (feature, time) pair.

        Args:
            features: (N, D) batch of features - either interpolated states
                z_t (standard FM training) or self-generated states z_hat_k
                (rolled-out training and inference).
            time: scalar broadcast to the whole batch, or per-sample times of
                shape (N,) or (N, 1).

        Returns:
            (N, D) predicted velocities, same shape as `features`.

        Raises:
            ValueError: if `features` is not 2-D, its width is not
                feature_dim, or `time` cannot be matched to the batch.
        """
        if features.dim() != 2:
            raise ValueError(f"features must be 2-D (N, D), got {features.dim()} dimensions")
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features have dimension {features.shape[1]}, "
                f"but this network was built for {self.feature_dim}"
            )

        time_column = _as_time_column(
            time, features.shape[0], features.device, features.dtype
        )
        return self.net(torch.cat([features, time_column], dim=1))


def build_near_identity_velocity_network(
    feature_dim: int, hidden_dims: Sequence[int] = (512, 512)
) -> VelocityNetwork:
    """Build a `VelocityNetwork` whose Euler rollout is exactly the identity.

    part_3.pdf requires: "Initialize the FM close to identity so that, before
    Stage 3 training, the complete system behaves approximately like the
    original linear probe."

    The network predicts a *velocity*, and each Euler step adds it to the
    current state:

        z_hat_{k+1} = z_hat_k + (1/T) * v_theta(z_hat_k, k/T)

    so a network that outputs zero contributes nothing at every step and
    leaves z_hat_T = z exactly - not approximately, for any T. Zeroing the
    final linear layer's weight and bias is sufficient, since every path
    through the network ends there.

    Only that layer is zeroed. The earlier layers keep PyTorch's default
    initialization, so the network already has varied hidden features to
    build on the moment its output layer moves off zero - which happens on
    the very first optimizer step. The gradient w.r.t. the final layer's
    weight is (upstream gradient) x (hidden activation), and the hidden
    activations are nonzero, so that layer updates immediately. Only the
    gradient flowing *back into* the earlier layers vanishes at step 0,
    because it passes through the zeroed weight; from step 1 onward every
    layer trains normally. A network zeroed throughout would instead be
    stuck permanently.

    Stage 2's networks are deliberately left as they were. `VelocityNetwork`
    keeps PyTorch's default initialization, which on DINOv2 features moves
    each feature by roughly 6% of its norm before any training - harmless
    when the flow's target is a fixed class prototype, but fatal to the
    property this function exists to provide. Constructing Stage 3's network
    through a separate builder rather than a flag on `VelocityNetwork` keeps
    every already-completed Stage 2 run bit-reproducible.

    Args:
        feature_dim: D, the frozen encoder's feature dimension.
        hidden_dims: widths of the hidden layers, in order.

    Returns:
        A `VelocityNetwork` that returns exactly zero for every input, so
        `euler_transport(net, z, T)` returns `z` unchanged.

    Raises:
        TypeError: if the constructed network does not end in a linear
            layer - which would mean the architecture changed underneath
            this function and the zeroing no longer guarantees anything.
        ValueError: propagated from `VelocityNetwork` for invalid dimensions.
    """
    model = VelocityNetwork(feature_dim, hidden_dims)

    final_layer = model.net[-1]
    if not isinstance(final_layer, nn.Linear):
        raise TypeError(
            "expected VelocityNetwork to end in nn.Linear, found "
            f"{type(final_layer).__name__}; the near-identity guarantee "
            "depends on zeroing that final layer"
        )

    nn.init.zeros_(final_layer.weight)
    nn.init.zeros_(final_layer.bias)
    return model
