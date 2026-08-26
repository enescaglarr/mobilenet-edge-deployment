import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
import time
from tqdm import tqdm

# === 1. Path to your dataset ===
dataset_dir = "/Users/enescaglar/Downloads/dataset"  # ← change this to your actual folder

# === 2. Load dataset using tf.data ===
IMG_SIZE = 224
BATCH_SIZE = 64

ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",     # integer class IDs (0–999)
    shuffle=False
)

# === 3. Preprocess for MobileNet ===
def preprocess(img, label):
    img = preprocess_input(img)  # scales pixels to [-1,1]
    return img, label

ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

# === 4. Load pretrained MobileNet ===
model = MobileNet(weights="imagenet")
print("✅ Loaded pretrained MobileNet with ImageNet weights.")

# === 5. Define metrics ===
top1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name="top1")
top5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")

# === 6. Evaluate ===
start_time = time.time()
total_images = 0

for imgs, labels in tqdm(ds, desc="Evaluating", unit="batch"):
    preds = model.predict(imgs, verbose=0)
    y_true = tf.one_hot(labels, depth=1000)
    top1.update_state(y_true, preds)
    top5.update_state(y_true, preds)
    total_images += imgs.shape[0]

elapsed = time.time() - start_time
speed = total_images / elapsed

# === 8. Detailed performance breakdown ===
import numpy as np

# Ortalama batch başına geçen süre (saniye)
avg_batch_time = elapsed / len(list(ds))
avg_image_time = elapsed / total_images

print("\n📊 Detailed CPU Performance Report:")
print("----------------------------------------")
print(f"🧠 Device: CPU")
print(f"🕒 Total evaluation time: {elapsed:.2f} sec")
print(f"📦 Total batches: {len(list(ds))}")
print(f"🖼️ Total images: {total_images}")
print(f"⏱️ Avg time per batch: {avg_batch_time:.4f} sec")
print(f"⚡ Avg time per image: {avg_image_time*1000:.2f} ms")
print(f"🚀 Throughput: {speed:.1f} images/sec")
print(f"🎯 Top-1 Accuracy: {top1.result().numpy()*100:.2f}%")
print(f"🎯 Top-5 Accuracy: {top5.result().numpy()*100:.2f}%")
print("----------------------------------------")

