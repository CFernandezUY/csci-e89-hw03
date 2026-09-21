"""Configure reproducible PyTorch setup and report the selected compute device.

This establishes a consistent starting point for the Fashion-MNIST classifier.
"""

import platform
import random

import numpy as np
import torch


def set_seed(seed=42):
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    """Select CUDA, then MPS, and otherwise CPU when available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_environment(device):
    """Print the software versions and selected PyTorch device."""
    print(f"Python version: {platform.python_version()}")
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print_environment(device)
