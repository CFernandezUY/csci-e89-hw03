# This is notebook step 7. Run the steps in numeric order in one Python session, or run run_image_classifier.py for a standalone program.

constructor_hyperparameters = {
    "input_features": 28 * 28,
    "hidden1": 300,
    "hidden2": 100,
    "num_classes": 10,
}
torch.save(
    {"state_dict": model.state_dict(), "constructor_hyperparameters": constructor_hyperparameters},
    CHECKPOINT_PATH,
)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
reloaded_model = ImageClassifier(**checkpoint["constructor_hyperparameters"]).to(device)
reloaded_model.load_state_dict(checkpoint["state_dict"])
reloaded_model.eval()
with torch.no_grad():
    original_outputs = model(example_images[:3].to(device))
    reloaded_outputs = reloaded_model(example_images[:3].to(device))
assert torch.equal(original_outputs, reloaded_outputs)
print(f"Saved and reloaded identical outputs from {CHECKPOINT_PATH}")
