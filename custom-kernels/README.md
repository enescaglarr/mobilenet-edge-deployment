# MobileNetV2 Custom CUDA Kernels (v2)

ENS492'deki "custom kernel" çalışmasının devamı. TRUBA yerine **ücretsiz Google Colab (T4)** hedeflenir; eğitim yok, sadece pretrained MobileNetV2 inference.

## Ne var

| Dosya | İçerik |
|---|---|
| `kernels/mnv2_kernels.cu` | 4 kernel: depthwise 3×3 **naive** / **tiled**, pointwise 1×1 **naive** / **tiled** (FP16 giriş-çıkış, FP32 accumulate, fused bias + ReLU6) |
| `custom_ops.py` | JIT derleme, `DepthwiseConv3x3` / `PointwiseConv1x1` modülleri, BN-folding + torchvision MobileNetV2 katman değişimi, `set_impl()` ile backend seçimi |
| `bench_kernels.py` | Her katman şekli için naive / tiled / cuDNN latency, FP32 referansa göre max hata, GFLOP/s |
| `bench_model.py` | Uçtan uca latency/throughput (6 backend) + ImageNet alt kümesinde top-1/top-5 + `torch.profiler` |
| `make_subset.py` | `../dataset`'ten sınıf başına N görüntülük küçük alt küme (Colab'a yüklemek için) |
| `MobileNetCustomKernels_Colab.ipynb` | Colab'da baştan sona çalıştıran notebook |

## Naive → tiled'da ne değişti

**Depthwise 3×3 (memory-bound)**
- Naive: her thread bir çıktı, 9 global load; komşu thread'ler aynı pikselleri tekrar okur.
- Tiled: blok başına 32×8 çıktı, giriş tile'ı + halo shared memory'ye bir kez alınır; 9 ağırlık da shared'da. Global okuma ~9× azalır. Stride 1/2 template ile derleme zamanında sabitlenir.

**Pointwise 1×1 (compute-bound, aslında GEMM)**
- Naive: her thread bir çıktı, `Ci` boyunca stride'lı global okuma → düşük cache verimi, sıfır veri paylaşımı.
- Tiled: klasik shared-memory GEMM (64×64 çıktı bloğu, K=16 adım) + **register blocking** (thread başına 4×4 çıktı) + `float4` shared okuma. Aritmetik yoğunluk 1 FMA/load'dan 16 FMA/load'a çıkar.

**Her ikisinde de fused epilogue**: bias + ReLU6 aynı kernel'de → PyTorch'taki ayrı `relu6` kernel launch'ı ve bir tam tensor okuma/yazma ortadan kalkar.

## Çalıştırma (Colab)

1. Mac'te: `python make_subset.py --per-class 2 && zip -r dataset_subset.zip dataset_subset` (hazır: `dataset_subset.zip`)
2. `custom-kernels/` klasörünü zip'le (`dataset_subset*` hariç): `zip -r custom-kernels.zip custom-kernels -x '*dataset_subset*'`
3. Colab → T4 → notebook'u aç, iki zip'i yükle, hücreleri sırayla çalıştır.
4. `results_kernels.csv` ve `results_model.csv` indir.

Toplam süre: derleme ~1-2 dk, benchmark'lar ~3-5 dk.

## Beklenen sonuç (dürüst tahmin)

- Depthwise tiled: naive'e göre belirgin, cuDNN'e göre **yakın veya daha iyi** (cuDNN'in depthwise kernel'leri zayıftır).
- Pointwise tiled: naive'e göre 10×+, ama cuDNN/cuBLAS **tensor core** kullandığı için ona yetişmesi beklenmez. Bu yüzden `mixed` backend (tiled dw + cuDNN pw) muhtemelen en hızlı uçtan uca sonuç.
- Doğruluk: FP16 nedeniyle fp32'ye göre top-1 farkı ≤0.2 puan olmalı; naive ve tiled birbirine eşit olmalı (aynı FP32 accumulate).

## Sonraki adımlar

- Pointwise için `wmma` (tensor core) versiyonu — cuDNN'e yaklaşmanın tek yolu.
- Depthwise + pointwise **fusion** (inverted residual bloğunu tek kernel'de).
- Kernel'leri TensorRT `IPluginV2` olarak sarıp ENS492'deki TRT FP16 pipeline'ına takmak.
