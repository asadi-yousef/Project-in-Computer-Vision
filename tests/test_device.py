import torch

from src.utils.device import get_device


def test_get_device_returns_a_usable_device():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cuda", "mps", "cpu")

    # Sanity check: the returned device can actually hold a tensor.
    tensor = torch.zeros(2, device=device)
    assert tensor.device.type == device.type
