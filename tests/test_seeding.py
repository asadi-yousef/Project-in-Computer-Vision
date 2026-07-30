import random

import numpy as np
import torch

from src.utils.seeding import set_seed


def test_same_seed_reproduces_values_across_all_libraries():
    set_seed(0)
    python_value_1 = random.random()
    numpy_value_1 = np.random.rand()
    torch_value_1 = torch.rand(1).item()

    set_seed(0)
    python_value_2 = random.random()
    numpy_value_2 = np.random.rand()
    torch_value_2 = torch.rand(1).item()

    assert python_value_1 == python_value_2
    assert numpy_value_1 == numpy_value_2
    assert torch_value_1 == torch_value_2


def test_different_seeds_produce_different_values():
    set_seed(0)
    a = torch.rand(8)
    set_seed(1)
    b = torch.rand(8)
    assert not torch.allclose(a, b)
