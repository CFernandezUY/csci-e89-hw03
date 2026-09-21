"""Load and inspect Fashion-MNIST data for the image classifier.

The images are scaled to float32 values in [0, 1] so the model receives
consistent inputs, and the training data is split for validation.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BATCH_SIZE = 32
TRAIN_SIZE = 55_000
SEED = 42


def get_transform():
    """Create the image transform used for Fashion-MNIST examples."""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def load_datasets(data_dir=DATA_DIR):
    """Download Fashion-MNIST and return its training and test datasets."""
    transform = get_transform()
    train_dataset = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


def make_dataloaders(train_dataset, test_dataset, batch_size=BATCH_SIZE, seed=SEED):
    """Split training examples and create train, validation, and test loaders."""
    validation_size = len(train_dataset) - TRAIN_SIZE
    generator = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset = random_split(
        train_dataset, [TRAIN_SIZE, validation_size], generator=generator
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader, test_loader


def print_data_summary(train_loader, validation_loader, test_loader, class_names):
    """Print split details and inspect one batch from the training loader."""
    images, labels = next(iter(train_loader))
    print(f"Training examples: {len(train_loader.dataset)}")
    print(f"Validation examples: {len(validation_loader.dataset)}")
    print(f"Test examples: {len(test_loader.dataset)}")
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"Image batch dtype: {images.dtype}")
    print(f"Label batch dtype: {labels.dtype}")
    print(f"Pixel value range: [{images.min().item():.1f}, {images.max().item():.1f}]")
    print(f"Class names: {class_names}")


if __name__ == "__main__":
    train_dataset, test_dataset = load_datasets()
    train_loader, validation_loader, test_loader = make_dataloaders(
        train_dataset, test_dataset
    )
    print_data_summary(
        train_loader, validation_loader, test_loader, train_dataset.classes
    )
