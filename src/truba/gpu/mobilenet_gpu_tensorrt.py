import os
os.environ["KERAS_HOME"] = "/arf/scratch/<truba-user>/.keras"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import preprocess_input
import onnx
import onnxruntime as ort

# === 1) Paths & constants ===
dataset_dir = "/arf/scratch/<truba-user>/mobileNetGPU/dataset"
IMG_SIZE = 224
BATCH_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
onnx_path = "mobilenet_v1.onnx"
results_csv = "mobilenet_trt_benchmark.csv"

# === 2) Export Keras MobileNet to ONNX once (NHWC, FP32) ===
if not os.path.exists(onnx_path):
    from tensorflow.keras.applications import MobileNet
    import tf2onnx

    print("🔄 Exporting MobileNet (ImageNet) to ONNX ...")
    model = MobileNet(weights="imagenet")
    # Explicit dynamic batch input (NHWC)
    spec = (tf.TensorSpec((None, IMG_SIZE, IMG_SIZE, 3), tf.float32, name="input"),)
    _model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=13, output_path=onnx_path
    )
    print(f"✅ Saved {onnx_path}")

# (Optional) sanity check ONNX
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

# === 3) Create ONNX Runtime session with TensorRT EP (fallback to CUDA) ===
sess_options = ort.SessionOptions()
# Slightly larger graph optimization & workspace hints
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

preferred_providers = [
    ("TensorrtExecutionProvider", {
        "trt_fp16_enable": True,          # enable FP16 kernels when possible
        "trt_engine_cache_enable": True,  # cache engines
        "trt_engine_cache_path": "./trt_cache",
        "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,  # 2GB
    }),
    "CUDAExecutionProvider",  # fallback if TRT EP not available
]

available_eps = ort.get_available_providers()
print("🔌 Available Execution Providers:", available_eps)

providers_to_use = []
for ep in preferred_providers:
    name = ep if isinstance(ep, str) else ep[0]
    if name in available_eps:
        providers_to_use.append(ep)

if not providers_to_use:
    raise RuntimeError(
        "No GPU execution provider found. Install onnxruntime-gpu and ensure CUDA/TensorRT libs are visible."
    )

session = ort.InferenceSession(onnx_path, sess_options, providers=providers_to_use)
print("✅ Using providers:", session.get_providers())

# Detect input/output names
inp = session.get_inputs()[0]
out = session.get_outputs()[0]
input_name = inp.name
output_name = out.name
print(f"🔎 I/O — input: {input_name}, shape: {inp.shape}, output: {output_name}, shape: {out.shape}")

# === 4) Dataset loader (same as TF version) ===
def load_dataset(batch_size):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        label_mode="int",
        shuffle=False
    )

    def _pre(img, label):
        # Keras MobileNet preprocess_input expects float in [-1,1], NHWC
        img = preprocess_input(img)
        return img, label

    ds = ds.map(_pre, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# === 5) Benchmark loop ===
results = []

for batch_size in BATCH_SIZES:
    print(f"\n🚀 Evaluating with batch size = {batch_size}...")
    ds = load_dataset(batch_size)

    top1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name="top1")
    top5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")

    total_images = 0
    num_batches = 0

    start_time = time.time()

    try:
        for imgs, labels in tqdm(ds, desc=f"Batch {batch_size}", unit="batch"):
            # imgs: TF tensor [N, 224, 224, 3], already preprocessed
            batch_np = imgs.numpy().astype(np.float32)
            # ORT expects dict {input_name: np.ndarray}
            preds = session.run([output_name], {input_name: batch_np})[0]  # shape [N,1000]

            y_true = tf.one_hot(labels, depth=1000)
            top1.update_state(y_true, preds)
            top5.update_state(y_true, preds)

            n = batch_np.shape[0]
            total_images += n
            num_batches += 1

    except RuntimeError as e:
        # Handle OOM or provider fallback issues gracefully
        print(f"⚠️  Stopped early at batch size {batch_size} due to: {e}")
        if num_batches == 0:
            # Skip entry if nothing processed
            continue

    elapsed = time.time() - start_time
    throughput = total_images / elapsed if elapsed > 0 else 0.0
    avg_batch_time = elapsed / num_batches if num_batches > 0 else 0.0
    avg_image_time = elapsed / total_images if total_images > 0 else 0.0

    results.append({
        "Batch Size": batch_size,
        "Total Images": total_images,
        "Total Time (s)": round(elapsed, 2),
        "Avg Batch Time (s)": round(avg_batch_time, 4),
        "Avg Image Time (ms)": round(avg_image_time * 1000, 2),
        "Throughput (img/sec)": round(throughput, 1),
        "Top-1 Acc (%)": round(float(top1.result().numpy()) * 100, 2),
        "Top-5 Acc (%)": round(float(top5.result().numpy()) * 100, 2),
        "EP Used": ",".join(session.get_providers()),
    })

# === 6) Save & show results ===
df = pd.DataFrame(results)
print("\n📊 === TensorRT-backed (ORT) GPU Benchmark Results ===")
print(df.to_string(index=False))

df.to_csv(results_csv, index=False)
print(f"\n💾 Results saved as {results_csv}")
print("\n✅ All evaluations completed successfully!")
