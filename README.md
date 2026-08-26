# MobileNet Edge Deployment

Profiling and accelerating **MobileNetV2** inference for edge/IoT deployment — from CPU
thread-scaling and multi-GPU batch sweeps on Turkey's national HPC (TRUBA), through
**TensorRT FP32/FP16** engines profiled with Nsight Systems, to hand-written **FP16 CUDA
kernels** for the two operators that dominate MobileNetV2 (3×3 depthwise and 1×1 pointwise
convolution).

Graduation project, Sabancı University (2025). All numbers below are measurements from this
repo's scripts; raw logs and CSVs are in [`results/`](results/).

## Headline results

| What | Where | Result |
|---|---|---|
| TensorRT FP16 vs FP32, batch 1 | Colab A100 | 0.475 ms → **0.297 ms** (2,104 → 3,372 FPS), 1.6× |
| TensorRT FP16 vs FP32, batch 64 | Colab A100 | 0.370 → 0.294 ms per batch (172,975 → 217,835 FPS) |
| Runtimes at batch 1 (same notebook) | Colab | PyTorch eager **on CPU** 20.5 ms → ONNX Runtime **on CPU** 2.5 ms → TensorRT FP16 on A100 0.39 ms (~52×, CPU-vs-GPU) |
| Custom depthwise 3×3 kernel vs `F.conv2d` FP16 (layer-level) | Colab A100 | **1.6× / 2.0× / 2.1×** at C=16·112², C=32·56², C=64·28² |
| Custom pointwise 1×1 kernel vs `F.conv2d` FP16 (layer-level) | Colab A100 | **2.1× / 2.1× / 2.6×** on the same shapes; drops to **0.7×** at C=128 (naive kernel loses to cuDNN) |
| PyTorch MobileNetV2, 50k ImageNet-val | TRUBA V100 | plateaus at ~840–900 img/s from batch 32; Top-1/Top-5 69.25 / 88.81 |
| TF-Keras MobileNetV1, 50k ImageNet-val | TRUBA V100 | 218 img/s (batch 16) → 2,234 img/s (batch 4096); Top-1/Top-5 69.06 / 88.52; same curve with 1, 2 or 4 GPUs requested |
| CPU thread scaling, MobileNetV1/V2/V3 | laptop | V1 +87% from 1→3 threads, saturates at 4; V3 only +50–60% |

Two things the numbers do **not** say, stated explicitly:

- The ~52× figure compares CPU-side PyTorch/ONNX Runtime against GPU TensorRT. The
  like-for-like GPU comparison is the FP16-vs-FP32 row.
- The custom kernels were benchmarked as **isolated layers**. They were not integrated into a
  full-model forward pass, so no end-to-end or accuracy figure is claimed for them. That
  integration (plus tiled/register-blocked versions of both kernels) lives in
  [`custom-kernels/`](custom-kernels/) and is the current work in progress.

## Repository layout

```
src/
  colab/          MobileNetCUDA.ipynb  — ONNX→TensorRT (FP32/FP16), cuda-python runtime,
                                         batch sweep, PyTorch/ORT/TRT comparison, nsys,
                                         custom dw/pw FP16 CUDA kernels via cpp_extension
                  LLMoptimization.ipynb — same PyTorch→ORT→TensorRT pipeline on BERT-base
  truba/gpu/      TF-Keras MobileNetV1 + PyTorch MobileNetV2 batch sweeps on 50k ImageNet-val,
                  SLURM scripts for 1/2/4×V100 (akya) and P100 (barbun, never scheduled)
  truba/tensorrt/ TensorRT engine build on TRUBA (failed: CUDA init error 35 — see results/)
  cpu/            MobileNetV1/V2/V3 CPU thread-scaling, Tiny-ImageNet fine-tuning experiments
custom-kernels/   v2: naive + tiled FP16 kernels, BN-folded MobileNetV2 integration,
                  per-layer and end-to-end benchmarks, Colab notebook  (see its README)
results/truba/    SLURM .out logs and benchmark CSVs (the source of every TRUBA number above)
docs/reports/     Proposal, progress reports, final report
docs/presentations/  Final presentation (pdf + pptx)
docs/weekly-slides/  Weekly progress decks (S1 = first semester, S2 = second)
models/           ONNX exports (git-ignored; regenerate with the scripts below)
dataset/          ImageNet-val subset, 1000 classes × 50 images (git-ignored)
```

## Reproducing

**TensorRT + custom kernels (Colab, A100/T4):** open `src/colab/MobileNetCUDA.ipynb`,
upload `models/MobileNet-v2.onnx` (or export one with `torchvision.models.mobilenet_v2` +
`torch.onnx.export`), run top to bottom. Needs `tensorrt`, `cuda-python`, `onnxruntime`.

**Custom kernels v2 (Colab, T4):** see [`custom-kernels/README.md`](custom-kernels/README.md) —
`bench_kernels.py` (per-layer naive / tiled / cuDNN) and `bench_model.py` (end-to-end + accuracy).

**TRUBA sweeps:** `sbatch src/truba/gpu/akya-slurm-{min,average,max}[-pytorch].slurm`. Scripts
assume `/arf/scratch/<user>/mobileNetGPU/dataset` and the `apps/truba-ai/gpu-2024.0` module;
edit the paths at the top of each `.py`.

**Dataset:** `dataset/` is an ImageNet-1k validation subset organised as `ImageFolder`
(`00000/` … `00999/`, folder index = ImageNet class index). It is not in the repo;
`custom-kernels/make_subset.py` builds a 2-images-per-class version for quick accuracy checks.

## Hardware & software

- TRUBA Akya-CUDA: Tesla V100-SXM2 16 GB (1/2/4 per job), SLURM, `apps/truba-ai/gpu-2024.0`
  (TensorFlow + PyTorch). TensorRT engine builds failed there with CUDA error 35
  (insufficient driver), so all TensorRT work moved to Colab.
- Google Colab: A100-SXM4-40 GB (CUDA driver 550.54), plus L4 / T4 for the GPU-selection
  comparison. TensorRT 10.x, `cuda-python`, Nsight Systems 2025.6.
- Custom kernels: `torch.utils.cpp_extension.load`, `-O3 --use_fast_math -lineinfo`, FP16
  in/out with FP32 accumulation; validated against `F.conv2d` (depthwise bit-exact,
  pointwise max abs error 0.016).

## Authors

Enes Çağlar — MobileNet track (this repo). The parallel EfficientNet track was carried out by
a teammate and is not included here.
