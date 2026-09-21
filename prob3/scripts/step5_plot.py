"""Step 5: plot training/validation accuracy and loss curves.

Adds the training-accuracy plot requested for this assignment on top of the
book's chapter (accuracy per epoch alongside loss per epoch), saved to a PNG
so the learning curves can be inspected without rerunning training.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from step1_setup import get_device, set_seed
from step2_data import build_dataloaders
from step3_model import build_model
from step4_train import SEED, train_model

OUTPUT_PATH = Path(__file__).resolve().parent / "training_curves.png"


def plot_training_curves(history: dict, save_path: Path = OUTPUT_PATH) -> Path:
    """Plot accuracy and loss curves (train + val) side by side and save to `save_path`."""
    epochs = range(1, len(history["train_acc"]) + 1)

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 5))

    ax_acc.plot(epochs, history["train_acc"], marker="o", label="Train accuracy")
    ax_acc.plot(epochs, history["val_acc"], marker="o", label="Validation accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Training and Validation Accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    ax_loss.plot(epochs, history["train_loss"], marker="o", label="Train loss")
    ax_loss.plot(epochs, history["val_loss"], marker="o", label="Validation loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training and Validation Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    return save_path


if __name__ == "__main__":
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders()
    model = build_model().to(device)

    history = train_model(model, train_loader, val_loader, device)

    saved_path = plot_training_curves(history)
    print(f"Saved training curves to {saved_path}")
