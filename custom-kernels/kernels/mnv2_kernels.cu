// MobileNetV2 custom FP16 CUDA kernels (PyTorch extension)
//
// Two operator families that make up ~95% of MobileNetV2 compute:
//   1. depthwise 3x3 conv (stride 1 or 2, padding 1)   -> memory-bound
//   2. pointwise 1x1 conv (== batched GEMM)             -> compute-bound
//
// For each we ship a NAIVE version (one thread per output, global-memory loads,
// same design as the ENS492 W13 kernels) and a TILED version
// (shared-memory tiling, coalesced loads, fused bias + ReLU6, FP32 accumulate).
// Layout is NCHW, input/weights/output are __half, bias is float.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>

#define CHECK_INPUT(x) \
  TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor"); \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

__device__ __forceinline__ float act(float v, bool relu6) {
  return relu6 ? fminf(fmaxf(v, 0.f), 6.f) : v;
}

// ============================================================================
// 1. DEPTHWISE 3x3
// ============================================================================

// ---- naive: one thread per output element, 9 global loads each -------------
template <int STRIDE>
__global__ void dw3x3_naive_kernel(const __half* __restrict__ x,
                                   const __half* __restrict__ w,   // [C, 9]
                                   const float*  __restrict__ b,   // [C] or null
                                   __half* __restrict__ y,
                                   int N, int C, int H, int W, int OH, int OW,
                                   bool relu6) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)N * C * OH * OW;
  if (idx >= total) return;
  int ow = idx % OW; long t = idx / OW;
  int oh = t % OH;   t /= OH;
  int c  = t % C;    int n = t / C;

  const __half* xp = x + ((size_t)n * C + c) * H * W;
  const __half* wp = w + c * 9;
  float acc = b ? b[c] : 0.f;
  int ih0 = oh * STRIDE - 1, iw0 = ow * STRIDE - 1;
#pragma unroll
  for (int kh = 0; kh < 3; ++kh) {
    int ih = ih0 + kh; if (ih < 0 || ih >= H) continue;
#pragma unroll
    for (int kw = 0; kw < 3; ++kw) {
      int iw = iw0 + kw; if (iw < 0 || iw >= W) continue;
      acc += __half2float(xp[ih * W + iw]) * __half2float(wp[kh * 3 + kw]);
    }
  }
  y[idx] = __float2half(act(acc, relu6));
}

// ---- tiled: each block computes a TH x TW output tile of ONE (n,c) plane ---
// Input tile (+halo) and the 9 weights are staged in shared memory, so every
// input element is read from global memory once instead of up to 9 times.
constexpr int DW_TW = 32;
constexpr int DW_TH = 8;

template <int STRIDE>
__global__ void dw3x3_tiled_kernel(const __half* __restrict__ x,
                                   const __half* __restrict__ w,
                                   const float*  __restrict__ b,
                                   __half* __restrict__ y,
                                   int N, int C, int H, int W, int OH, int OW,
                                   bool relu6) {
  constexpr int IN_TW = DW_TW * STRIDE + 2;
  constexpr int IN_TH = DW_TH * STRIDE + 2;
  __shared__ float s_in[IN_TH][IN_TW];
  __shared__ float s_w[9];

  const int tx = threadIdx.x, ty = threadIdx.y;
  const int tid = ty * DW_TW + tx;
  const int ow0 = blockIdx.x * DW_TW;
  const int oh0 = blockIdx.y * DW_TH;
  const int iw0 = ow0 * STRIDE - 1;
  const int ih0 = oh0 * STRIDE - 1;

  // gridDim.z may be clamped to 65535 -> loop over planes
  for (int nc = blockIdx.z; nc < N * C; nc += gridDim.z) {
    const int c = nc % C;
    const __half* xp = x + (size_t)nc * H * W;

    if (tid < 9) s_w[tid] = __half2float(w[c * 9 + tid]);
    for (int i = tid; i < IN_TH * IN_TW; i += DW_TW * DW_TH) {
      int r = i / IN_TW, q = i % IN_TW;
      int ih = ih0 + r, iw = iw0 + q;
      s_in[r][q] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                       ? __half2float(xp[ih * W + iw]) : 0.f;
    }
    __syncthreads();

    const int oh = oh0 + ty, ow = ow0 + tx;
    if (oh < OH && ow < OW) {
      float acc = b ? b[c] : 0.f;
#pragma unroll
      for (int kh = 0; kh < 3; ++kh)
#pragma unroll
        for (int kw = 0; kw < 3; ++kw)
          acc += s_in[ty * STRIDE + kh][tx * STRIDE + kw] * s_w[kh * 3 + kw];
      y[((size_t)nc * OH + oh) * OW + ow] = __float2half(act(acc, relu6));
    }
    __syncthreads();  // before next plane overwrites s_in
  }
}

torch::Tensor dw3x3_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b,
                            int stride, bool relu6, bool tiled) {
  CHECK_INPUT(x); CHECK_INPUT(w);
  TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be half");
  TORCH_CHECK(w.scalar_type() == torch::kHalf, "w must be half");
  TORCH_CHECK(stride == 1 || stride == 2, "stride must be 1 or 2");
  const int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
  TORCH_CHECK(w.numel() == (long)C * 9, "w must be [C,9]");
  const int OH = (H + 2 - 3) / stride + 1;
  const int OW = (W + 2 - 3) / stride + 1;
  auto y = torch::empty({N, C, OH, OW}, x.options());

  const float* bp = nullptr;
  if (b.defined() && b.numel() > 0) {
    CHECK_INPUT(b); TORCH_CHECK(b.scalar_type() == torch::kFloat, "b must be float32");
    bp = b.data_ptr<float>();
  }
  auto xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
  auto wp = reinterpret_cast<const __half*>(w.data_ptr<at::Half>());
  auto yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());
  auto stream = at::cuda::getCurrentCUDAStream();

  if (tiled) {
    dim3 block(DW_TW, DW_TH);
    dim3 grid((OW + DW_TW - 1) / DW_TW, (OH + DW_TH - 1) / DW_TH,
              (unsigned)std::min((long)N * C, 65535L));
    if (stride == 1)
      dw3x3_tiled_kernel<1><<<grid, block, 0, stream>>>(xp, wp, bp, yp, N, C, H, W, OH, OW, relu6);
    else
      dw3x3_tiled_kernel<2><<<grid, block, 0, stream>>>(xp, wp, bp, yp, N, C, H, W, OH, OW, relu6);
  } else {
    long total = (long)N * C * OH * OW;
    int threads = 256;
    long blocks = (total + threads - 1) / threads;
    if (stride == 1)
      dw3x3_naive_kernel<1><<<blocks, threads, 0, stream>>>(xp, wp, bp, yp, N, C, H, W, OH, OW, relu6);
    else
      dw3x3_naive_kernel<2><<<blocks, threads, 0, stream>>>(xp, wp, bp, yp, N, C, H, W, OH, OW, relu6);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// ============================================================================
// 2. POINTWISE 1x1  ==  Y[n, co, p] = sum_ci W[co, ci] * X[n, ci, p]  (+b, ReLU6)
// ============================================================================

// ---- naive: one thread per output, strided global reads over ci -----------
__global__ void pw1x1_naive_kernel(const __half* __restrict__ x,   // [N, Ci, P]
                                   const __half* __restrict__ w,   // [Co, Ci]
                                   const float*  __restrict__ b,   // [Co] or null
                                   __half* __restrict__ y,         // [N, Co, P]
                                   int N, int Ci, int Co, int P, bool relu6) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)N * Co * P;
  if (idx >= total) return;
  int p  = idx % P; long t = idx / P;
  int co = t % Co;  int n = t / Co;
  const __half* xp = x + (size_t)n * Ci * P + p;
  const __half* wp = w + (size_t)co * Ci;
  float acc = b ? b[co] : 0.f;
  for (int ci = 0; ci < Ci; ++ci)
    acc += __half2float(wp[ci]) * __half2float(xp[(size_t)ci * P]);
  y[idx] = __float2half(act(acc, relu6));
}

// ---- tiled GEMM with register blocking -------------------------------------
// Block tile: BM (co) x BN (p), K-step BK (ci). 16x16 threads, each thread
// owns a TM x TN register tile => 4x4 = 16 outputs per thread.
constexpr int PW_BM = 64, PW_BN = 64, PW_BK = 16, PW_TM = 4, PW_TN = 4;
constexpr int PW_THREADS = (PW_BM / PW_TM) * (PW_BN / PW_TN);  // 256

__global__ void pw1x1_tiled_kernel(const __half* __restrict__ x,
                                   const __half* __restrict__ w,
                                   const float*  __restrict__ b,
                                   __half* __restrict__ y,
                                   int N, int Ci, int Co, int P, bool relu6) {
  __shared__ __align__(16) float sW[PW_BM][PW_BK];
  __shared__ __align__(16) float sX[PW_BK][PW_BN];

  const int tx = threadIdx.x, ty = threadIdx.y;     // 0..15
  const int tid = ty * 16 + tx;
  const int p0  = blockIdx.x * PW_BN;
  const int co0 = blockIdx.y * PW_BM;
  const int n   = blockIdx.z;
  const __half* xn = x + (size_t)n * Ci * P;

  float acc[PW_TM][PW_TN];
#pragma unroll
  for (int i = 0; i < PW_TM; ++i)
#pragma unroll
    for (int j = 0; j < PW_TN; ++j) acc[i][j] = 0.f;

  for (int ci0 = 0; ci0 < Ci; ci0 += PW_BK) {
    // stage W tile [BM x BK]
    for (int i = tid; i < PW_BM * PW_BK; i += PW_THREADS) {
      int r = i / PW_BK, k = i % PW_BK;
      int co = co0 + r, ci = ci0 + k;
      sW[r][k] = (co < Co && ci < Ci) ? __half2float(w[(size_t)co * Ci + ci]) : 0.f;
    }
    // stage X tile [BK x BN]  (q contiguous -> coalesced)
    for (int i = tid; i < PW_BK * PW_BN; i += PW_THREADS) {
      int k = i / PW_BN, q = i % PW_BN;
      int ci = ci0 + k, p = p0 + q;
      sX[k][q] = (ci < Ci && p < P) ? __half2float(xn[(size_t)ci * P + p]) : 0.f;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < PW_BK; ++k) {
      float4 xv = *reinterpret_cast<const float4*>(&sX[k][tx * PW_TN]);
      float xr[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
      for (int m = 0; m < PW_TM; ++m) {
        float a = sW[ty * PW_TM + m][k];
#pragma unroll
        for (int j = 0; j < PW_TN; ++j) acc[m][j] += a * xr[j];
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < PW_TM; ++m) {
    int co = co0 + ty * PW_TM + m;
    if (co >= Co) continue;
    float bias = b ? b[co] : 0.f;
    __half* yrow = y + ((size_t)n * Co + co) * P;
#pragma unroll
    for (int j = 0; j < PW_TN; ++j) {
      int p = p0 + tx * PW_TN + j;
      if (p < P) yrow[p] = __float2half(act(acc[m][j] + bias, relu6));
    }
  }
}

torch::Tensor pw1x1_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b,
                            bool relu6, bool tiled) {
  CHECK_INPUT(x); CHECK_INPUT(w);
  TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be half");
  TORCH_CHECK(w.scalar_type() == torch::kHalf, "w must be half");
  const int N = x.size(0), Ci = x.size(1), H = x.size(2), W = x.size(3);
  const int P = H * W;
  TORCH_CHECK(w.dim() == 2 && w.size(1) == Ci, "w must be [Co, Ci]");
  const int Co = w.size(0);
  auto y = torch::empty({N, Co, H, W}, x.options());

  const float* bp = nullptr;
  if (b.defined() && b.numel() > 0) {
    CHECK_INPUT(b); TORCH_CHECK(b.scalar_type() == torch::kFloat, "b must be float32");
    bp = b.data_ptr<float>();
  }
  auto xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
  auto wp = reinterpret_cast<const __half*>(w.data_ptr<at::Half>());
  auto yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());
  auto stream = at::cuda::getCurrentCUDAStream();

  if (tiled) {
    dim3 block(16, 16);
    dim3 grid((P + PW_BN - 1) / PW_BN, (Co + PW_BM - 1) / PW_BM, N);
    pw1x1_tiled_kernel<<<grid, block, 0, stream>>>(xp, wp, bp, yp, N, Ci, Co, P, relu6);
  } else {
    long total = (long)N * Co * P;
    int threads = 256;
    long blocks = (total + threads - 1) / threads;
    pw1x1_naive_kernel<<<blocks, threads, 0, stream>>>(xp, wp, bp, yp, N, Ci, Co, P, relu6);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dw3x3_forward", &dw3x3_forward,
        "Depthwise 3x3 conv FP16 (x, w[C,9], b[C]|empty, stride, relu6, tiled)");
  m.def("pw1x1_forward", &pw1x1_forward,
        "Pointwise 1x1 conv FP16 (x, w[Co,Ci], b[Co]|empty, relu6, tiled)");
}
