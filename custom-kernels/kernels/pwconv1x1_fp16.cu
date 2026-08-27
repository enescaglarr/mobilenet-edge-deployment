#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void pwconv1x1_fp16_kernel(
    const half* __restrict__ x,
    const half* __restrict__ w,
    half* __restrict__ y,
    int N, int Cin, int Cout, int H, int W
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * H * W;
    if (tid >= total) return;

    int t = tid;
    int w0 = t % W; t /= W;
    int h0 = t % H; t /= H;
    int co = t % Cout; t /= Cout;
    int n  = t;

    float acc = 0.f;
    const half* w_row = w + co * Cin;

    for (int ci = 0; ci < Cin; ci++) {
        float xv = __half2float(x[((n*Cin + ci)*H + h0)*W + w0]);
        float wv = __half2float(w_row[ci]);
        acc += xv * wv;
    }

    y[((n*Cout + co)*H + h0)*W + w0] = __float2half(acc);
}

torch::Tensor pwconv1x1_fp16(torch::Tensor x, torch::Tensor w) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w.is_cuda(), "w must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be FP16");
    TORCH_CHECK(w.scalar_type() == at::kHalf, "w must be FP16");
    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(w.dim() == 4, "w must be [Cout, Cin, 1, 1]");

    int N = x.size(0);
    int Cin = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int Cout = w.size(0);

    auto y = torch::empty({N, Cout, H, W}, x.options());

    int total = N * Cout * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    pwconv1x1_fp16_kernel<<<blocks, threads>>>(
        (half*)x.data_ptr<at::Half>(),
        (half*)w.data_ptr<at::Half>(),
        (half*)y.data_ptr<at::Half>(),
        N, Cin, Cout, H, W
    );

    return y;
}
