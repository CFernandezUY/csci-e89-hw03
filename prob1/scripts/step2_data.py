"""Step 2: Fashion-MNIST data loading for the MLP classifier assignment.

Reproduces the data-loading portion of "Building an Image Classifier with
PyTorch" (Chapter 10, Hands-On ML with Scikit-Learn and PyTorch, Geron 2025):
load Fashion-MNIST, scale images to float32 in [0, 1], split the 60,000
training images into 55,000 train / 5,000 validation, and wrap everything
in DataLoaders.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2

SEED = 42
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BATCH_SIZE = 32
N_TRAIN = 55_000
N_VAL = 5_000

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

TRANSFORM = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)


def load_datasets(data_dir: Path = DATA_DIR):
    """Download (if needed) and return the raw train/test FashionMNIST datasets."""
    train_full = FashionMNIST(
        root=data_dir, train=True, download=True, transform=TRANSFORM
    )
    test = FashionMNIST(
        root=data_dir, train=False, download=True, transform=TRANSFORM
    )
    return train_full, test


def split_train_val(train_full, n_train: int = N_TRAIN, n_val: int = N_VAL, seed: int = SEED):
    """Split the full training set into train/validation subsets with a seeded generator."""
    generator = torch.Generator().manual_seed(seed)
    return random_split(train_full, [n_train, n_val], generator=generator)


def build_dataloaders(batch_size: int = BATCH_SIZE, data_dir: Path = DATA_DIR):
    """Load Fashion-MNIST, split it, and wrap the three subsets in DataLoaders."""
    train_full, test = load_datasets(data_dir)
    train, val = split_train_val(train_full)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = build_dataloaders()

    print(f"Train size:      {len(train_loader.dataset)}")
    print(f"Validation size: {len(val_loader.dataset)}")
    print(f"Test size:       {len(test_loader.dataset)}")

    images, labels = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}, dtype: {images.dtype}")
    print(f"Batch labels shape: {labels.shape}, dtype: {labels.dtype}")
    print(f"Pixel value range:  [{images.min().item():.3f}, {images.max().item():.3f}]")
    print(f"Class names: {CLASS_NAMES}")
