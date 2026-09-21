"""Step 7: save/load the Fashion-MNIST MLP checkpoint.

Saves the model's state_dict together with its constructor
hyperparameters, reloads the checkpoint with torch.load(weights_only=True)
(the safe-unpickling mode), and asserts the reloaded model produces
identical outputs to the original.
"""

from pathlib import Path

import torch

from step1_setup import get_device, set_seed
from step3_model import ImageClassifier, build_model

SEED = 42
CHECKPOINT_PATH = Path(__file__).resolve().parent / "model_checkpoint.pt"

HYPERPARAMETERS = {
    "input_size": 784,
    "hidden1_size": 300,
    "hidden2_size": 100,
    "num_classes": 10,
}


def save_checkpoint(model: ImageClassifier, path: Path = CHECKPOINT_PATH, hyperparameters: dict = HYPERPARAMETERS) -> Path:
    """Save the model's state_dict plus its constructor hyperparameters."""
    checkpoint = {
        "state_dict": model.state_dict(),
        "hyperparameters": hyperparameters,
    }
    torch.save(checkpoint, path)
    return path


def load_checkpoint(path: Path = CHECKPOINT_PATH, device: torch.device = torch.device("cpu")):
    """Reload a checkpoint saved by save_checkpoint using the safe weights_only mode."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = build_model().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint["hyperparameters"]


if __name__ == "__main__":
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    original_model = build_model().to(device)
    original_model.eval()

    dummy_input = torch.rand(4, 1, 28, 28, device=device)
    with torch.no_grad():
        original_output = original_model(dummy_input)

    saved_path = save_checkpoint(original_model)
    print(f"Saved checkpoint to {saved_path}")

    reloaded_model, reloaded_hyperparameters = load_checkpoint(saved_path, device)
    reloaded_model.eval()
    print(f"Reloaded hyperparameters: {reloaded_hyperparameters}")
    assert reloaded_hyperparameters == HYPERPARAMETERS, "Hyperparameters changed across save/load"

    with torch.no_grad():
        reloaded_output = reloaded_model(dummy_input)

    assert torch.equal(original_output, reloaded_output), "Reloaded model output differs from original"
    print("Reloaded model output is identical to the original. All assertions passed.")
