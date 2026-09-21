"""Define the multilayer perceptron used to classify Fashion-MNIST images.

The network maps flattened 28-by-28 images to ten raw class logits for use
with cross-entropy loss during training.
"""

import torch
from torch import nn


class ImageClassifier(nn.Module):
    """A 784-300-100-10 multilayer perceptron image classifier."""

    def __init__(
        self,
        input_size=784,
        first_hidden_size=300,
        second_hidden_size=100,
        num_classes=10,
    ):
        super().__init__()
        self.hyperparameters = {
            "input_size": input_size,
            "first_hidden_size": first_hidden_size,
            "second_hidden_size": second_hidden_size,
            "num_classes": num_classes,
        }
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, first_hidden_size),
            nn.ReLU(),
            nn.Linear(first_hidden_size, second_hidden_size),
            nn.ReLU(),
            nn.Linear(second_hidden_size, num_classes),
        )

    def forward(self, images):
        """Return unnormalized class scores for a batch of images."""
        return self.layers(images)


def build_model(**hyperparameters):
    """Build a new ImageClassifier instance."""
    return ImageClassifier(**hyperparameters)


def count_parameters(model):
    """Return the number of trainable parameters in a model."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    batch_size = 32
    model = build_model()
    images = torch.rand(batch_size, 1, 28, 28)
    output = model(images)
    parameter_count = count_parameters(model)

    assert output.shape == (batch_size, 10)
    assert parameter_count == 266_610

    print(f"Output shape: {output.shape}")
    print(f"Trainable parameters: {parameter_count:,}")
