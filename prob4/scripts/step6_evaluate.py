"""Evaluate the trained Fashion-MNIST classifier once on the held-out test set.

This reports final test accuracy and interprets a few predictions using class
names and softmax probabilities after training has finished.
"""

import torch

from step1_setup import get_device, set_seed
from step2_data import BATCH_SIZE, load_datasets, make_dataloaders
from step3_model import build_model
from step4_train import train_model


def evaluate_test_set(model, test_loader, class_names, device, examples_to_print=3):
    """Print final test accuracy and predictions for a few test images."""
    model.eval()
    correct_predictions = 0
    total_examples = 0
    examples = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            predicted_labels = logits.argmax(dim=1)

            correct_predictions += (predicted_labels == labels).sum().item()
            total_examples += labels.size(0)

            remaining_examples = examples_to_print - len(examples)
            for image_logits, predicted_label, true_label in zip(
                logits[:remaining_examples],
                predicted_labels[:remaining_examples],
                labels[:remaining_examples],
            ):
                probabilities = torch.softmax(image_logits, dim=0)
                top_probabilities, top_labels = probabilities.topk(3)
                top_three = [
                    f"{class_names[label.item()]}: {probability.item():.2%}"
                    for probability, label in zip(top_probabilities, top_labels)
                ]
                examples.append(
                    (
                        class_names[predicted_label.item()],
                        class_names[true_label.item()],
                        top_three,
                    )
                )

    test_accuracy = correct_predictions / total_examples
    print(f"Test accuracy: {test_accuracy:.2%}")
    for number, (predicted_class, true_class, top_three) in enumerate(examples, start=1):
        print(f"Test image {number}:")
        print(f"  Predicted class: {predicted_class}")
        print(f"  True class: {true_class}")
        print(f"  Top-3 softmax probabilities: {', '.join(top_three)}")

    return test_accuracy


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    train_dataset, test_dataset = load_datasets()
    train_loader, validation_loader, test_loader = make_dataloaders(
        train_dataset, test_dataset, batch_size=BATCH_SIZE
    )
    model = build_model()
    train_model(model, train_loader, validation_loader, device)
    evaluate_test_set(model, test_loader, test_dataset.classes, device)
