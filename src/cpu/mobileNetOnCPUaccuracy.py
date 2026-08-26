import os
import shutil
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import (
    mobilenet_v2,
    mobilenet_v3_small,
    mobilenet_v3_large,
    MobileNet_V2_Weights,
    MobileNet_V3_Small_Weights,
    MobileNet_V3_Large_Weights
)

def prepare_val_folder(val_dir):
    val_annotations_path = os.path.join(val_dir, 'val_annotations.txt')
    val_images_dir = os.path.join(val_dir, 'images')

    if not os.path.exists(val_images_dir):
        print("Validation images already reorganized.")
        return

    with open(val_annotations_path, 'r') as f:
        data = f.readlines()

    for line in data:
        parts = line.strip().split('\t')
        file_name = parts[0]
        class_name = parts[1]
        class_folder = os.path.join(val_dir, class_name)

        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

        source = os.path.join(val_images_dir, file_name)
        destination = os.path.join(class_folder, file_name)
        shutil.move(source, destination)

    shutil.rmtree(val_images_dir)
    print("Validation set reorganized successfully!")

def get_data_loaders(data_root, batch_size=64):
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')

    transform = transforms.Compose([
        transforms.Resize((160, 160)),  # 🔥 make images smaller
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(root=train_dir, transform=transform)
    val_dataset = ImageFolder(root=val_dir, transform=transform)

    # Optional: use only a subset for quick training
    train_subset = torch.utils.data.Subset(train_dataset, range(0, 10000))  # 🔥 first 10k images

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def train_one_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(data_loader)

def main():
    data_root = '<path-to-tiny-imagenet-200>'
    device = torch.device('cpu')

    print(f"Device in use: {device}")

    prepare_val_folder(os.path.join(data_root, 'val'))
    train_loader, val_loader = get_data_loaders(data_root)

    # Load and modify models
    model_v2 = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model_v2.classifier[1] = nn.Linear(model_v2.last_channel, 200)

    model_v3_small = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model_v3_small.classifier[3] = nn.Linear(model_v3_small.classifier[3].in_features, 200)

    model_v3_large = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    model_v3_large.classifier[3] = nn.Linear(model_v3_large.classifier[3].in_features, 200)

    models = {
        "MobileNetV2": model_v2,
        "MobileNetV3 Small": model_v3_small,
        "MobileNetV3 Large": model_v3_large
    }

    results = {}

    for name, model in models.items():
        print(f"\nFine-tuning only classifier of {name} (2 epochs)...")
        model = model.to(device)

        # ✅ Freeze everything except the classifier
        for param in model.parameters():
            param.requires_grad = False

        if name == "MobileNetV2":
            for param in model.classifier[1].parameters():
                param.requires_grad = True
        else:
            for param in model.classifier[3].parameters():
                param.requires_grad = True

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        start_time = time.time()

        for epoch in range(2):  # only 2 epochs
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            print(f"Epoch [{epoch+1}/2], Loss: {loss:.4f}")

        acc = evaluate(model, val_loader, device)
        elapsed_time = time.time() - start_time

        print(f"{name} Accuracy after fine-tuning: {acc:.2f}%")
        print(f"{name} Fine-tuning + Evaluation Time: {elapsed_time/60:.2f} minutes")

        results[name] = {"accuracy": acc, "time_minutes": elapsed_time / 60}

    print("\n=== Final Fine-tuning Results (2 Epochs, Classifier Only) ===")
    for name, res in results.items():
        print(f"{name}: Accuracy = {res['accuracy']:.2f}%, Fine-tuning Time = {res['time_minutes']:.2f} minutes")

if __name__ == '__main__':
    main()
