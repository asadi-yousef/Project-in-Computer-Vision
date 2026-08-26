"""The velocity network v_theta(z, t) used by both Stage 2 flow-matching
variants (stage_2.pdf).

Architecture follows the spec's suggestion: a small MLP with two hidden
layers of width ~512 and SiLU activations, the scalar time t concatenated to
the input feature, and output dimension equal to the feature dimension. The
output is an unbounded velocity vector, so the final layer has no activation.

The same network definition serves standard FM and rolled-out training - only
the training objective differs between them - and the same forward pass is
used at inference by the Euler integrator.
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
