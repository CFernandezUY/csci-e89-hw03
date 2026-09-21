"""Save and reload the Fashion-MNIST classifier without changing its outputs.

Saving model weights with their constructor hyperparameters makes a trained
model reproducible when it is loaded in a later Python session.
"""

from pathlib import Path

import torch

from step1_setup import get_device, set_seed
from step2_data import BATCH_SIZE, load_datasets, make_dataloaders
from step3_model import ImageClassifier, build_model
from step4_train import train_model


MODEL_PATH = Path(__file__).resolve().parents[1] / "my_fashion_mnist_model.pt"


def save_model(model, model_path=MODEL_PATH):
    """Save a model state dictionary and the hyperparameters that build it."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_hyperparameters": model.hyperparameters,
    }
    torch.save(checkpoint, model_path)


def load_model(model_path=MODEL_PATH, device="cpu"):
    """Reload saved weights into an ImageClassifier with matching dimensions."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = ImageClassifier(**checkpoint["model_hyperparameters"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def assert_identical_outputs(original_model, reloaded_model, images):
    """Assert that the original and reloaded models produce identical logits."""
    original_model.eval()
    reloaded_model.eval()
    with torch.no_grad():
        original_outputs = original_model(images)
        reloaded_outputs = reloaded_model(images)
    assert torch.equal(original_outputs, reloaded_outputs)


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    train_dataset, test_dataset = load_datasets()
    train_loader, validation_loader, test_loader = make_dataloaders(
        train_dataset, test_dataset, batch_size=BATCH_SIZE
    )
    model = build_model()
    train_model(model, train_loader, validation_loader, device)

    save_model(model)
    reloaded_model = load_model(device=device)
    test_images, _ = next(iter(test_loader))
    assert_identical_outputs(model, reloaded_model, test_images.to(device))
    print(f"Saved and reloaded model: {MODEL_PATH}")
