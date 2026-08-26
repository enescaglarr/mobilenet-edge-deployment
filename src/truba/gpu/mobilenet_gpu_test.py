import os
os.environ["KERAS_HOME"] = "/arf/scratch/oceylan/.keras"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
import time
from tqdm import tqdm
import numpy as np
import pandas as pd


# === 1. Dataset path ===
dataset_dir = "/arf/scratch/oceylan/mobileNetGPU/dataset"  # <-- buraya dataset dizininin yolunu yaz
IMG_SIZE = 224
BATCH_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

# === 2. GPU Check ===
print("\n🔍 Checking available devices:")
print(tf.config.list_physical_devices())

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU detected: {gpus[0].name}")
else:
    print("⚠️ No GPU detected. Running on CPU instead.")

# === 3. Load pretrained MobileNet ===
model = MobileNet(weights="imagenet")
print("✅ Loaded pretrained MobileNet (ImageNet weights).")

# === 4. Dataset loader ===
def load_dataset(batch_size):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        label_mode="int",
        shuffle=False
    )

    def preprocess(img, label):
        img = preprocess_input(img)
        return img, label

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# === 5. Evaluate for each batch size ===
results = []

for batch_size in BATCH_SIZES:
    print(f"\n🚀 Evaluating with batch size = {batch_size}...")
    ds = load_dataset(batch_size)

    top1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name="top1")
    top5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")

    start_time = time.time()
    total_images = 0
    num_batches = 0

    for imgs, labels in tqdm(ds, desc=f"Batch {batch_size}", unit="batch"):
        preds = model.predict(imgs, verbose=0)
        y_true = tf.one_hot(labels, depth=1000)
        top1.update_state(y_true, preds)
        top5.update_state(y_true, preds)
        total_images += imgs.shape[0]
        num_batches += 1

    elapsed = time.time() - start_time
    throughput = total_images / elapsed
    avg_batch_time = elapsed / num_batches
    avg_image_time = elapsed / total_images

    results.append({
        "Batch Size": batch_size,
        "Total Images": total_images,
        "Total Time (s)": round(elapsed, 2),
        "Avg Batch Time (s)": round(avg_batch_time, 4),
        "Avg Image Time (ms)": round(avg_image_time * 1000, 2),
        "Throughput (img/sec)": round(throughput, 1),
        "Top-1 Acc (%)": round(top1.result().numpy() * 100, 2),
        "Top-5 Acc (%)": round(top5.result().numpy() * 100, 2)
    })

# === 6. Save & show results ===
df = pd.DataFrame(results)
print("\n📊 === Batch Size GPU Benchmark Results ===")
print(df.to_string(index=False))

df.to_csv("mobilenet_gpu_benchmark.csv", index=False)
print("\n💾 Results saved as mobilenet_gpu_benchmark.csv")

print("\n✅ All evaluations completed successfully!")
