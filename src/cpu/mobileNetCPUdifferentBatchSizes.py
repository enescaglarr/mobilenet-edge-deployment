import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
import time
from tqdm import tqdm
import numpy as np
import pandas as pd

# === 1. Dataset path ===
dataset_dir = "/Users/enescaglar/Downloads/dataset"  # ← change to your actual folder
IMG_SIZE = 224
BATCH_SIZES = [16, 32, 64, 128, 256, 512]

# === 2. Load pretrained MobileNet once ===
model = MobileNet(weights="imagenet")
print("✅ Loaded pretrained MobileNet (ImageNet weights).")

# === 3. Function to load dataset for a given batch size ===
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

# === 4. Run evaluation for each batch size ===
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

# === 5. Show results as a table ===
df = pd.DataFrame(results)
print("\n📊 === Batch Size Benchmark Results ===")
print(df.to_string(index=False))

# === 6. Optional: save results to CSV ===
df.to_csv("mobilenet_batch_benchmark.csv", index=False)
print("\n💾 Results saved as mobilenet_batch_benchmark.csv")
