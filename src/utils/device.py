"""Compute device selection that works across the two target machines
(a Windows PC with a CUDA GPU, and a Mac laptop with Apple Silicon)."""

import torch


def get_device() -> torch.device:
    """Pick the best available compute device.

    Preference order: CUDA (Windows GPU) -> MPS (Apple Silicon) -> CPU.
    Keeping this in one place means no other module has to branch on
    platform or hardware.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
