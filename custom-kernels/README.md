# MobileNetV2 Custom CUDA Kernels (v2)

Continuation of the custom-kernel work from the graduation project. Targets **free Google Colab
(T4)** instead of TRUBA; inference only, no training — pretrained torchvision MobileNetV2.

## Contents

| File | What it is |
|---|---|
| `kernels/mnv2_kernels.cu` | 4 kernels: depthwise 3×3 **naive** / **tiled**, pointwise 1×1 **naive** / **tiled** (FP16 in/out, FP32 accumulate, fused bias + ReLU6) |
| `custom_ops.py` | JIT build, `DepthwiseConv3x3` / `PointwiseConv1x1` modules, BatchNorm folding + layer replacement in torchvision MobileNetV2, `set_impl()` to switch backend at runtime |
| `bench_kernels.py` | Per-layer-shape latency for naive / tiled / cuDNN, max error vs an FP32 reference, GFLOP/s |
| `bench_model.py` | End-to-end latency/throughput for 6 backends + top-1/top-5 on an ImageNet subset + `torch.profiler` kernel breakdown |
| `make_subset.py` | Builds a small N-images-per-class subset of `../dataset` for uploading to Colab |
| `MobileNetCustomKernels_Colab.ipynb` | Runs everything end to end on Colab |

## What changed from naive to tiled

**Depthwise 3×3 (memory-bound)**
- Naive: one thread per output, 9 global loads; neighbouring threads re-read the same pixels.
- Tiled: each block computes a 32×8 output tile; the input tile plus halo and the 9 weights are
  staged in shared memory once. Global reads drop ~9×. Stride 1/2 is a compile-time template.

**Pointwise 1×1 (compute-bound — it is a GEMM)**
- Naive: one thread per output, strided global reads along `Ci`, poor cache use, no data reuse.
- Tiled: classic shared-memory GEMM (64×64 output block, K-step 16) with **register blocking**
  (4×4 outputs per thread) and `float4` shared-memory reads. Arithmetic intensity goes from
  1 FMA/load to 16 FMA/load.

**Fused epilogue in both**: bias + ReLU6 inside the kernel, removing PyTorch's separate `relu6`
launch and one full tensor read/write.

## Running on Colab

1. On the Mac: `python make_subset.py --per-class 2 && zip -r dataset_subset.zip dataset_subset`
2. Zip this folder without the subset: `zip -r custom-kernels.zip custom-kernels -x '*dataset_subset*'`
3. Colab → T4 runtime → open the notebook, upload both zips, run the cells in order.
4. Download `results_kernels.csv` and `results_model.csv`.

Total time: ~1–2 min compile, ~3–5 min benchmarks.

## Expected outcome (honest estimate)

- Depthwise tiled: clear gain over naive, **close to or better than cuDNN** (cuDNN's depthwise
  kernels are comparatively weak).
- Pointwise tiled: 10×+ over naive, but not expected to catch cuDNN/cuBLAS, which use **tensor
  cores**. The `mixed` backend (tiled depthwise + cuDNN pointwise) is therefore likely the fastest
  end-to-end configuration.
- Accuracy: FP16 should cost ≤0.2 pp top-1 vs FP32; naive and tiled should match each other
  exactly (same FP32 accumulation order per output).

## Next steps

- `wmma` (tensor-core) version of the pointwise kernel — the only route to cuDNN parity.
- Fuse depthwise + pointwise (one inverted-residual block per kernel).
- Wrap the kernels as a TensorRT `IPluginV2` and plug them into the project's TRT FP16 pipeline.
