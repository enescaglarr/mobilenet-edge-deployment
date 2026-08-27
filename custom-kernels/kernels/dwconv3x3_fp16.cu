#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void dwconv3x3_fp16_kernel(
    const half* __restrict__ x,
    const half* __restrict__ w,
    half* __restrict__ y,
    int N, int C, int H, int W
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (tid >= total) return;

    int t = tid;
    int w0 = t % W; t /= W;
    int h0 = t % H; t /= H;
    int c  = t % C; t /= C;
    int n  = t;

    const half* wc = w + c * 9;

    float acc = 0.0f;
    #pragma unroll
    for (int kh = -1; kh <= 1; kh++) {
        #pragma unroll
        for (int kw = -1; kw <= 1; kw++) {
            int h = h0 + kh;
            int ww = w0 + kw;
            if ((unsigned)h < (unsigned)H && (unsigned)ww < (unsigned)W) {
                float xv = __half2float(x[((n*C + c)*H + h)*W + ww]);
                float wv = __half2float(wc[(kh+1)*3 + (kw+1)]);
                acc += xv * wv;
            }
        }
    }
    y[((n*C + c)*H + h0)*W + w0] = __float2half(acc);
}

torch::Tensor dwconv3x3_fp16(torch::Tensor x, torch::Tensor w) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w.is_cuda(), "w must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be FP16");
    TORCH_CHECK(w.scalar_type() == at::kHalf, "w must be FP16");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto y = torch::empty_like(x);

    int total = N*C*H*W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    dwconv3x3_fp16_kernel<<<blocks, threads>>>(
        (half*)x.data_ptr<at::Half>(),
        (half*)w.data_ptr<at::Half>(),
        (half*)y.data_ptr<at::Half>(),
        N, C, H, W
    );

    return y;
}
