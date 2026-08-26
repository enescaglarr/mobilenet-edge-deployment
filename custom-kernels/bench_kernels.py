"""Per-layer micro-benchmark + correctness check.

For every distinct depthwise/pointwise layer shape in MobileNetV2 (224x224)
measure naive / tiled / cuDNN kernels and compare against an FP32 reference.

    python bench_kernels.py --batch 1 8 32 --out results_kernels.csv
"""
import argparse
import csv
import torch
import torch.nn.functional as F
import torchvision

from custom_ops import (DepthwiseConv3x3, PointwiseConv1x1, convert_mobilenet_v2,
                        custom_layers, load_ext)


@torch.no_grad()
def collect_shapes(model, device):
    """Run one dry pass and record input H,W of every custom layer."""
    shapes, hooks = [], []
    for m in custom_layers(model):
        hooks.append(m.register_forward_hook(
            lambda mod, inp, out: shapes.append((mod, tuple(inp[0].shape[2:])))))
    model(torch.randn(1, 3, 224, 224, device=device).half())
    for h in hooks:
        h.remove()
    seen, uniq = set(), []
    for m, hw in shapes:
        key = (type(m).__name__, getattr(m, "C", None), getattr(m, "Ci", None),
               getattr(m, "Co", None), getattr(m, "stride", 1), hw, m.relu6)
        if key not in seen:
            seen.add(key)
            uniq.append((m, hw))
    return uniq


@torch.no_grad()
def time_fn(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


@torch.no_grad()
def reference(layer, x):
    """FP32 reference of the fused op (conv + bias + optional ReLU6)."""
    xf = x.float()
    if isinstance(layer, DepthwiseConv3x3):
        y = F.conv2d(xf, layer.w4.float(), None if layer.b32 is None else layer.b32,
                     stride=layer.stride, padding=1, groups=layer.C)
    else:
        y = F.conv2d(xf, layer.w4.float(), None if layer.b32 is None else layer.b32)
    return F.relu6(y) if layer.relu6 else y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", default="results_kernels.csv")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))
    load_ext(verbose=True)

    base = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model = convert_mobilenet_v2(base).to(device)
    layers = collect_shapes(model, device)
    print(f"{len(layers)} distinct layer shapes")

    rows = []
    for bs in args.batch:
        for layer, (H, W) in layers:
            if isinstance(layer, DepthwiseConv3x3):
                name, cin = "dw3x3", layer.C
                desc = f"dw3x3 C={layer.C} s={layer.stride} {H}x{W}"
                flops = 2 * bs * layer.C * ((H + 2 - 3) // layer.stride + 1) * ((W + 2 - 3) // layer.stride + 1) * 9
            else:
                name, cin = "pw1x1", layer.Ci
                desc = f"pw1x1 {layer.Ci}->{layer.Co} {H}x{W}"
                flops = 2 * bs * layer.Ci * layer.Co * H * W
            x = (torch.randn(bs, cin, H, W, device=device) * 2).half().contiguous()
            ref = reference(layer, x)
            res = {}
            for impl in ("cudnn", "naive", "tiled"):
                layer.impl = impl
                y = layer(x)
                err = (y.float() - ref).abs().max().item()
                ms = time_fn(lambda: layer(x), args.iters)
                res[impl] = (ms, err)
            c, n, t = res["cudnn"][0], res["naive"][0], res["tiled"][0]
            print(f"bs={bs:3d} {desc:32s} cudnn {c:7.4f} ms | naive {n:7.4f} ms | tiled {t:7.4f} ms "
                  f"| tiled/naive {n / t:5.2f}x  tiled/cudnn {c / t:5.2f}x "
                  f"| maxerr naive {res['naive'][1]:.3e} tiled {res['tiled'][1]:.3e} "
                  f"| tiled {flops / t / 1e9:.0f} GFLOP/s")
            rows.append(dict(batch=bs, op=name, desc=desc, H=H, W=W,
                             cudnn_ms=c, naive_ms=n, tiled_ms=t,
                             speedup_vs_naive=n / t, speedup_vs_cudnn=c / t,
                             maxerr_naive=res["naive"][1], maxerr_tiled=res["tiled"][1],
                             tiled_gflops=flops / t / 1e9))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved", args.out)

    # summary per op / batch
    for bs in args.batch:
        for op in ("dw3x3", "pw1x1"):
            sel = [r for r in rows if r["batch"] == bs and r["op"] == op]
            tot = {k: sum(r[k] for r in sel) for k in ("cudnn_ms", "naive_ms", "tiled_ms")}
            print(f"SUM bs={bs} {op}: cudnn {tot['cudnn_ms']:.3f} ms  naive {tot['naive_ms']:.3f} ms  "
                  f"tiled {tot['tiled_ms']:.3f} ms  (tiled vs naive {tot['naive_ms'] / tot['tiled_ms']:.2f}x, "
                  f"vs cudnn {tot['cudnn_ms'] / tot['tiled_ms']:.2f}x)")


if __name__ == "__main__":
    main()
