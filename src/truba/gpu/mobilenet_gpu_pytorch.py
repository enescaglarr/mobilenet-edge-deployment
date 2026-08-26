import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import pandas as pd
from tqdm import tqdm
import os

# === 1. Paths & Settings ===
dataset_dir = "/arf/scratch/oceylan/mobileNetGPU/dataset"
BATCH_SIZES = [16, 32, 64, 128, 256, 512, 1024]   # keep ≤1024 to avoid OOM
IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n🧠 Device in use: {device}")
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)} ({gpu.total_memory/1024**3:.1f} GB VRAM)")
else:
    print("⚠️ GPU not detected, running on CPU.")

# === 2. Load pretrained MobileNetV2 ===
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.eval().to(device)
print("✅ Loaded pretrained MobileNetV2 (ImageNet weights).")

# === 3. Dataset loader ===
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
dataset = ImageFolder(root=dataset_dir, transform=transform)

# === 4. Evaluation ===
results = []

@torch.no_grad()
def evaluate(batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    total_images, correct_top1, correct_top5 = 0, 0, 0
    start = time.time()

    try:
        for imgs, labels in tqdm(dataloader, desc=f"Batch {batch_size}", unit="batch"):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)

            # --- Accuracy ---
            top5 = probs.topk(5, dim=1)
            correct = top5.indices.eq(labels.view(-1, 1))
            correct_top1 += correct[:, :1].sum().item()
            correct_top5 += correct.sum().item()
            total_images += imgs.size(0)

        elapsed = time.time() - start
        throughput = total_images / elapsed
        avg_batch = elapsed / len(dataloader)
        avg_image = elapsed / total_images
        top1_acc = (correct_top1 / total_images) * 100
        top5_acc = (correct_top5 / total_images) * 100

        results.append({
            "Batch Size": batch_size,
            "Total Images": total_images,
            "Total Time (s)": round(elapsed, 2),
            "Avg Batch Time (s)": round(avg_batch, 4),
            "Avg Image Time (ms)": round(avg_image * 1000, 2),
            "Throughput (img/s)": round(throughput, 1),
            "Top-1 Acc (%)": round(top1_acc, 2),
            "Top-5 Acc (%)": round(top5_acc, 2)
        })

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"⚠️ Skipping batch {batch_size} due to OOM.")
            torch.cuda.empty_cache()
        else:
            raise e

# === 5. Run benchmark ===
for bs in BATCH_SIZES:
    evaluate(bs)

# === 6. Save & show ===
df = pd.DataFrame(results)
print("\n📊 === PyTorch GPU Benchmark Results ===")
print(df.to_string(index=False))
save_path = "mobilenet_gpu_benchmark_pytorch.csv"
df.to_csv(save_path, index=False)
print(f"\n💾 Results saved as {save_path}")
print("\n✅ All evaluations completed successfully!")
