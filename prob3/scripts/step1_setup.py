"""Step 1: environment setup for the Fashion-MNIST MLP assignment.

Reproduces the setup portion of "Building an Image Classifier with PyTorch"
(Chapter 10, Hands-On Machine Learning with Scikit-Learn and PyTorch, Geron 2025):
fix random seeds for reproducibility and pick the best available device.
"""

import platform
import random

import numpy as np
import torch

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Fix the seed for the random, numpy, and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Pick CUDA if available, else Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    set_seed(SEED)
    device = get_device()

    print(f"Python version:  {platform.python_version()}")
    print(f"NumPy version:   {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Seed:            {SEED}")
    print(f"Device:          {device}")
