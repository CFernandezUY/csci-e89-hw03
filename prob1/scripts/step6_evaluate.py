"""Step 6: final test-set evaluation for the Fashion-MNIST MLP classifier.

Reproduces the final evaluation from "Building an Image Classifier with
PyTorch" (Chapter 10, Hands-On ML with Scikit-Learn and PyTorch, Geron 2025):
the test set is touched exactly once, after training, to report accuracy
and a handful of individual predictions with their softmax probabilities.
"""

import torch

from step1_setup import get_device, set_seed
from step2_data import CLASS_NAMES, build_dataloaders
from step3_model import build_model
from step4_train import SEED, evaluate, train_model

TOP_K = 3
N_SAMPLES = 3


@torch.no_grad()
def test_accuracy(model, test_loader, device) -> float:
    """Compute accuracy on the test set. Intended to be called once, after training."""
    criterion = torch.nn.CrossEntropyLoss()
    _, accuracy = evaluate(model, test_loader, criterion, device)
    return accuracy


@torch.no_grad()
def predict_samples(model, dataset, indices, device, class_names=CLASS_NAMES, top_k=TOP_K):
    """For each index into `dataset`, return the true/predicted class and top-k softmax probs."""
    model.eval()
    results = []

    for idx in indices:
        image, true_label = dataset[idx]
        logits = model(image.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1).squeeze(0)
        top_probs, top_classes = probs.topk(top_k)
        predicted_label = int(top_classes[0].item())

        results.append(
            {
                "index": idx,
                "true_class": class_names[true_label],
                "predicted_class": class_names[predicted_label],
                "top_k": [
                    (class_names[int(c)], float(p))
                    for p, c in zip(top_probs, top_classes)
                ],
            }
        )

    return results


if __name__ == "__main__":
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders()
    model = build_model().to(device)

    train_model(model, train_loader, val_loader, device)

    acc = test_accuracy(model, test_loader, device)
    print(f"\nTest accuracy: {acc:.4f}")

    sample_indices = list(range(N_SAMPLES))
    predictions = predict_samples(model, test_loader.dataset, sample_indices, device)

    print(f"\nSample predictions (top-{TOP_K} softmax probabilities):")
    for pred in predictions:
        print(f"\n  Test image #{pred['index']}")
        print(f"    True class:      {pred['true_class']}")
        print(f"    Predicted class: {pred['predicted_class']}")
        for class_name, prob in pred["top_k"]:
            print(f"      {class_name:<14s} {prob:.4f}")
