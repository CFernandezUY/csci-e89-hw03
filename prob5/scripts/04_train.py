# This is notebook step 4. Run the steps in numeric order in one Python session, or run run_image_classifier.py for a standalone program.

def evaluate(data_loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for batch_images, batch_labels in data_loader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            batch_logits = model(batch_images)
            total_loss += loss_fn(batch_logits, batch_labels).item() * batch_labels.size(0)
            total_correct += (batch_logits.argmax(dim=1) == batch_labels).sum().item()
            total_examples += batch_labels.size(0)
    return total_loss / total_examples, total_correct / total_examples

history = {"train_loss": [], "train_accuracy": [], "validation_loss": [], "validation_accuracy": []}
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss_total = 0.0
    train_correct = 0
    train_examples = 0
    for batch_images, batch_labels in train_loader:
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        batch_logits = model(batch_images)
        loss = loss_fn(batch_logits, batch_labels)
        loss.backward()
        optimizer.step()

        train_loss_total += loss.item() * batch_labels.size(0)
        train_correct += (batch_logits.argmax(dim=1) == batch_labels).sum().item()
        train_examples += batch_labels.size(0)

    train_loss = train_loss_total / train_examples
    train_accuracy = train_correct / train_examples
    validation_loss, validation_accuracy = evaluate(validation_loader)
    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)
    history["validation_loss"].append(validation_loss)
    history["validation_accuracy"].append(validation_accuracy)
    print(
        f"Epoch {epoch:02d}/{EPOCHS}: train loss={train_loss:.4f}, train accuracy={train_accuracy:.4f}, "
        f"validation loss={validation_loss:.4f}, validation accuracy={validation_accuracy:.4f}"
    )
