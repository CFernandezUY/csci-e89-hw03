"""Plot Fashion-MNIST training curves to visualize model learning progress.

The accuracy and loss plots make it easy to compare training with validation
performance and identify possible overfitting.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from step1_setup import get_device, set_seed
from step2_data import BATCH_SIZE, load_datasets, make_dataloaders
from step3_model import build_model
from step4_train import train_model


OUTPUT_PATH = Path(__file__).with_name("training_curves.png")


def plot_history(history, output_path=OUTPUT_PATH):
    """Plot accuracy and loss histories, then save and display the figure."""
    epochs = range(1, len(history["train_accuracy"]) + 1)
    figure, (accuracy_axis, loss_axis) = plt.subplots(1, 2, figsize=(12, 4))

    accuracy_axis.plot(epochs, history["train_accuracy"], label="Training")
    accuracy_axis.plot(epochs, history["validation_accuracy"], label="Validation")
    accuracy_axis.set_title("Accuracy")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.legend()

    loss_axis.plot(epochs, history["train_loss"], label="Training")
    loss_axis.plot(epochs, history["validation_loss"], label="Validation")
    loss_axis.set_title("Loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Cross-entropy loss")
    loss_axis.legend()

    figure.tight_layout()
    figure.savefig(output_path)
    plt.show()


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    train_dataset, test_dataset = load_datasets()
    train_loader, validation_loader, _ = make_dataloaders(
        train_dataset, test_dataset, batch_size=BATCH_SIZE
    )
    history = train_model(
        build_model(), train_loader, validation_loader, device, epochs=3
    )
    plot_history(history)
