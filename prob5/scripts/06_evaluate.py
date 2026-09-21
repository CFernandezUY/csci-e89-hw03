# This is notebook step 6. Run the steps in numeric order in one Python session, or run run_image_classifier.py for a standalone program.

test_loss, test_accuracy = evaluate(test_loader)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

model.eval()
example_images, example_labels = next(iter(test_loader))
with torch.no_grad():
    example_probabilities = model(example_images[:3].to(device)).softmax(dim=1).cpu()
for index, (probabilities, true_label) in enumerate(zip(example_probabilities, example_labels[:3]), start=1):
    top_probabilities, top_indices = probabilities.topk(3)
    top_three = [(test_dataset.classes[class_index], round(probability.item(), 4))
                 for probability, class_index in zip(top_probabilities, top_indices)]
    print(
        f"Test image {index}: predicted={test_dataset.classes[probabilities.argmax().item()]}, "
        f"true={test_dataset.classes[true_label.item()]}, top-3={top_three}"
    )
