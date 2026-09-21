# This is notebook step 3. Run the steps in numeric order in one Python session, or run run_image_classifier.py for a standalone program.

class ImageClassifier(nn.Module):
    def __init__(self, input_features=28 * 28, hidden1=300, hidden2=100, num_classes=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):
        return self.network(x)

model = ImageClassifier().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

with torch.no_grad():
    logits = model(images.to(device))
assert logits.shape == (images.shape[0], 10)
parameter_count = sum(parameter.numel() for parameter in model.parameters())
assert parameter_count == 266_610
print(f"Output shape: {tuple(logits.shape)}")
print(f"Trainable parameters: {parameter_count:,}")
