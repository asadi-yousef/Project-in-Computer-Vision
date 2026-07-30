"""Print which compute device this project will use, and why.

Useful to run before starting real training: confirms whether CUDA (or MPS
on a Mac) was actually detected, so a run doesn't silently fall back to CPU.

Usage:
    python scripts/check_device.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.utils.device import get_device


def main() -> None:
    device = get_device()
    print(f"Selected device: {device}")
    print(f"torch version: {torch.__version__}")

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  VRAM: {total_memory_gb:.1f} GB")

    mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    print(f"MPS available: {mps_available}")

    if device.type == "cpu":
        print("Warning: neither CUDA nor MPS was detected; training/extraction will run on CPU.")


if __name__ == "__main__":
    main()
