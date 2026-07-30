"""Global random-seed control for reproducible experiments."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every source of randomness used in this project.

    Why it is needed: the spec requires that few-shot subset sampling and
    classifier initialization/training be exactly reproducible given a seed.
    PyTorch, NumPy, and Python's own `random` module each keep independent
    random state, so all three must be seeded together.

    Args:
        seed: non-negative integer seed.

    Note: `cudnn.deterministic = True` can make some GPU ops slower but is
    required for bit-for-bit reproducibility across runs on CUDA.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
