"""End-to-end MobileNetV2 benchmark: latency/throughput per backend and
(optionally) top-1/top-5 accuracy on an ImageNet-style folder.

    python bench_model.py --batch 1 8 32 --data ./dataset_subset --out results_model.csv

Backends:
  fp32_eager   torchvision model, FP32, cuDNN
  fp16_eager   torchvision model, FP16, cuDNN (BN not folded)
  fused_cudnn  BN folded, FP16, cuDNN for dw + pw          <- fair baseline
  custom_naive BN folded, FP16, naive kernels for dw + pw
  custom_tiled BN folded, FP16, tiled kernels for dw + pw
  mixed        tiled depthwise + cuDNN pointwise (best-of-both)
"""
import argparse
import csv
import os
import torch
import torchvision
import torchvision.transforms as T

from custom_ops import convert_mobilenet_v2, set_impl, load_ext

BACKENDS = {
    "fused_cudnn": ("cudnn", "cudnn"),
    "custom_naive": ("naive", "naive"),
    "custom_tiled": ("tiled", "tiled"),
    "mixed": ("tiled", "cudnn"),
}


@torch.no_grad()
def time_model(model, x, iters=50, warmup=10):
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        model(x)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


@torch.no_grad()
def accuracy(model, loader, device, dtype):
    top1 = top5 = n = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, dtype), labels.to(device)
        out = model(imgs).float()
        pred = out.topk(5, dim=1).indices
        top1 += (pred[:, 0] == labels).sum().item()
        top5 += (pred == labels[:, None]).any(dim=1).sum().item()
        n += labels.numel()
    return 100 * top1 / n, 100 * top5 / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--data", default=None, help="ImageFolder root (class folders sorted = ImageNet index)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", default="results_model.csv")
    ap.add_argument("--profile", action="store_true", help="print top CUDA kernels for custom_tiled @ batch 1")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))
    load_ext()

    base = torchvision.models.mobilenet_v2(
        weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1).eval().to(device)
    base16 = torchvision.models.mobilenet_v2(
        weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1).eval().to(device).half()
    custom = convert_mobilenet_v2(base).to(device)

    # numerical agreement vs FP32 eager
    x = torch.randn(4, 3, 224, 224, device=device)
    with torch.no_grad():
        ref = base(x)
        for name, (dw, pw) in BACKENDS.items():
            set_impl(custom, dw, pw)
            y = custom(x.half()).float()
            print(f"{name:13s} max|Δlogit| vs fp32 = {(y - ref).abs().max().item():.4f}  "
                  f"argmax agree = {(y.argmax(1) == ref.argmax(1)).float().mean().item() * 100:.0f}%")

    loader = None
    if args.data:
        tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        ds = torchvision.datasets.ImageFolder(args.data, transform=tf)
        loader = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=2, pin_memory=True)
        print(f"dataset: {len(ds)} images, {len(ds.classes)} classes")

    rows = []
    configs = [("fp32_eager", base, torch.float32), ("fp16_eager", base16, torch.float16)]
    configs += [(name, custom, torch.float16) for name in BACKENDS]
    for name, model, dtype in configs:
        if name in BACKENDS:
            set_impl(custom, *BACKENDS[name])
        acc1 = acc5 = float("nan")
        if loader is not None:
            acc1, acc5 = accuracy(model, loader, device, dtype)
        for bs in args.batch:
            x = torch.randn(bs, 3, 224, 224, device=device, dtype=dtype)
            ms = time_model(model, x, args.iters)
            print(f"{name:13s} bs={bs:3d}  {ms:8.3f} ms/batch  {ms / bs:7.3f} ms/img  "
                  f"{bs * 1000 / ms:9.1f} img/s   top1 {acc1:.2f}  top5 {acc5:.2f}")
            rows.append(dict(backend=name, batch=bs, ms_per_batch=ms, ms_per_img=ms / bs,
                             img_per_s=bs * 1000 / ms, top1=acc1, top5=acc5))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved", args.out)

    if args.profile:
        from torch.profiler import profile, ProfilerActivity
        set_impl(custom, "tiled", "tiled")
        x = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16)
        with torch.no_grad():
            for _ in range(5):
                custom(x)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(10):
                    custom(x)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


if __name__ == "__main__":
    main()
