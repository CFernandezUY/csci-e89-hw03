# This is notebook step 2. Run the steps in numeric order in one Python session, or run run_image_classifier.py for a standalone program.

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

train_full = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=transform)
train_dataset, validation_dataset = random_split(
    train_full,
    [55_000, 5_000],
    generator=torch.Generator().manual_seed(SEED),
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

images, labels = next(iter(train_loader))
print(f"Split sizes: train={len(train_dataset)}, validation={len(validation_dataset)}, test={len(test_dataset)}")
print(f"Batch image shape/dtype: {images.shape}, {images.dtype}")
print(f"Batch label shape/dtype: {labels.shape}, {labels.dtype}")
print(f"Pixel range: [{images.min().item():.1f}, {images.max().item():.1f}]")
print(f"Class names: {train_full.classes}")
