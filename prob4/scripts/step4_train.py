"""Train the Fashion-MNIST image classifier with a manual PyTorch loop.

This records loss and accuracy after each epoch so learning progress can be
inspected and plotted in the next step.
"""

import torch
from torch import nn

from step1_setup import get_device, set_seed
from step2_data import BATCH_SIZE, load_datasets, make_dataloaders
from step3_model import build_model


EPOCHS = 10
LEARNING_RATE = 0.1


def evaluate(model, data_loader, loss_function, device):
    """Return average loss and accuracy for a validation data loader."""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_function(logits, labels)

            total_loss += loss.item() * labels.size(0)
            correct_predictions += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += labels.size(0)

    return total_loss / total_examples, correct_predictions / total_examples


def train_model(
    model,
    train_loader,
    validation_loader,
    device,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
):
    """Train a model with SGD and return per-epoch loss and accuracy history."""
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_examples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct_predictions += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += labels.size(0)

        train_loss = total_loss / total_examples
        train_accuracy = correct_predictions / total_examples
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, loss_function, device
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)
        print(
            f"Epoch {epoch:2d}/{epochs}: "
            f"train loss={train_loss:.4f}, train accuracy={train_accuracy:.2%}, "
            f"validation loss={validation_loss:.4f}, "
            f"validation accuracy={validation_accuracy:.2%}"
        )

    return history


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    train_dataset, test_dataset = load_datasets()
    train_loader, validation_loader, _ = make_dataloaders(
        train_dataset, test_dataset, batch_size=BATCH_SIZE
    )
    model = build_model()
    history = train_model(model, train_loader, validation_loader, device)

    assert history["validation_accuracy"][-1] > 0.80
