# ============================================================
# Fine-tune MobileNetV3-Large on Custom Dataset (PyTorch)
# macOS / CPU SAFE VERSION
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time

def main():
    # ====== DEVICE CONFIGURATION ======
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🟢 Using device: {device}")

    # ====== PATH SETTINGS ======
    train_dir = "/Users/enescaglar/Documents/Lecture Notes/ENS491/tiny-imagenet-200/train"
    val_dir   = "/Users/enescaglar/Documents/Lecture Notes/ENS491/tiny-imagenet-200/val"

    # ====== HYPERPARAMETERS ======
    num_classes = 200
    batch_size = 32
    num_epochs = 5
    learning_rate = 0.001

    # ====== TRANSFORMS ======
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # ====== DATASETS & LOADERS ======
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset   = datasets.ImageFolder(root=val_dir, transform=transform)

    # On macOS, set num_workers=0 to avoid multiprocessing error
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"✔ Loaded dataset: {len(train_dataset)} training images, {len(val_dataset)} validation images")
    print(f"✔ Number of classes: {num_classes}")

    # ====== MODEL: MobileNetV3-Large ======
    mobilenet_v3_large = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)

    # Freeze base layers (optional)
    for param in mobilenet_v3_large.features.parameters():
        param.requires_grad = False

    # Replace final classification head
    mobilenet_v3_large.classifier[3] = nn.Linear(in_features=1280, out_features=num_classes)
    mobilenet_v3_large = mobilenet_v3_large.to(device)

    # ====== LOSS & OPTIMIZER ======
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mobilenet_v3_large.parameters(), lr=learning_rate)

    # ====== TRAINING LOOP ======
    print("\n🚀 Starting training...")
    for epoch in range(num_epochs):
        mobilenet_v3_large.train()
        running_loss = 0.0
        t0 = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = mobilenet_v3_large(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        elapsed = time.time() - t0
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f} - Time: {elapsed:.1f}s")

    # ====== EVALUATION ======
    mobilenet_v3_large.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = mobilenet_v3_large(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\n✅ Validation Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
