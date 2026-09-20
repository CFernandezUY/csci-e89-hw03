"""Step 3: the 784-300-100-10 MLP classifier for Fashion-MNIST.

Reproduces the model definition from "Building an Image Classifier with
PyTorch" (Chapter 10, Hands-On ML with Scikit-Learn and PyTorch, Geron 2025):
a flatten layer followed by two hidden ReLU layers and a linear output
layer returning raw logits (no softmax).
"""

import torch
from torch import nn

EXPECTED_PARAM_COUNT = 266_610


class ImageClassifier(nn.Module):
    """784-300-100-10 MLP that returns raw logits over 10 classes."""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden1 = nn.Linear(784, 300)
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(300, 100)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(100, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.hidden1(x))
        x = self.relu2(self.hidden2(x))
        return self.output(x)


def build_model() -> ImageClassifier:
    """Construct an ImageClassifier instance."""
    return ImageClassifier()


def count_parameters(model: nn.Module) -> int:
    """Total number of parameters (trainable and not) in the model."""
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = build_model()
    print(model)

    batch_size = 32
    dummy_input = torch.rand(batch_size, 1, 28, 28)
    logits = model(dummy_input)
    print(f"Output shape: {tuple(logits.shape)}")
    assert logits.shape == (batch_size, 10), f"Unexpected output shape: {logits.shape}"

    n_params = count_parameters(model)
    print(f"Total parameters: {n_params}")
    assert n_params == EXPECTED_PARAM_COUNT, (
        f"Expected {EXPECTED_PARAM_COUNT} parameters, got {n_params}"
    )

    print("All assertions passed.")
