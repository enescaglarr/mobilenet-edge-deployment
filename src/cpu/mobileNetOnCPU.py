import torch
import torchvision.models as models
import timm
import time
import matplotlib.pyplot as plt

device = torch.device("cpu")
runs = 10
batch_size = 50
thread_counts = [3,4]

# Load models
mobilenet_v1 = timm.create_model('mobilenetv1_100', pretrained=True).to(device).eval()
mobilenet_v2 = models.mobilenet_v2(pretrained=True).to(device).eval()
mobilenet_v3 = models.mobilenet_v3_large(pretrained=True).to(device).eval()

models_dict = {
    "MobileNet V1": mobilenet_v1,
    "MobileNet V2": mobilenet_v2,
    "MobileNet V3": mobilenet_v3
}

# Store results: results[model_name] = [tp_1thread, tp_4threads, tp_8threads]
results = {model_name: [] for model_name in models_dict.keys()}

def benchmark_model(model, batch_size, runs):
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    times = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.time()
            _ = model(dummy_input)
            end = time.time()
            times.append(end - start)
    avg_time = sum(times) / len(times)
    throughput = batch_size / avg_time
    return throughput

# Run benchmark for each model and thread count
for threads in thread_counts:
    print(f"\n🔧 THREADS: {threads}")
    torch.set_num_threads(threads)
    for model_name, model in models_dict.items():
        tp = benchmark_model(model, batch_size, runs)
        results[model_name].append(tp)
        print(f"  {model_name}: {tp:.2f} images/sec")


plt.figure(figsize=(10, 6))

for model_name, throughputs in results.items():
    plt.plot(thread_counts, throughputs, marker='o', label=model_name)

plt.title("MobileNet Inference Throughput vs Thread Count (CPU)")
plt.xlabel("Number of CPU Threads")
plt.ylabel("Throughput (images/sec)")
plt.xticks(thread_counts)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()