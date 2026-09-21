"""Step 4: manual training loop for the Fashion-MNIST MLP classifier.

Reproduces the training loop from "Building an Image Classifier with
PyTorch" (Chapter 10, Hands-On ML with Scikit-Learn and PyTorch, Geron 2025):
CrossEntropyLoss, plain SGD with lr=0.1, for exactly 10 epochs, tracking
train/validation loss and accuracy per epoch.
"""

import torch
from torch import nn

from step1_setup import get_device, set_seed
from step2_data import build_dataloaders
from step3_model import build_model

SEED = 42
LEARNING_RATE = 0.1
N_EPOCHS = 10


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch; return (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate the model on a loader (no gradients); return (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def train_model(model, train_loader, val_loader, device, epochs=N_EPOCHS, lr=LEARNING_RATE):
    """Train for `epochs` epochs with SGD, printing one line per epoch.

    Returns a history dict with train_loss, train_acc, val_loss, val_acc lists.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:2d}/{epochs} - "
            f"train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - "
            f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
        )

    return history


if __name__ == "__main__":
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders()
    model = build_model().to(device)

    history = train_model(model, train_loader, val_loader, device)

    print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")
