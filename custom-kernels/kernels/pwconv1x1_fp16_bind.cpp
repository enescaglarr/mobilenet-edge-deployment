#include <torch/extension.h>

torch::Tensor pwconv1x1_fp16(torch::Tensor x, torch::Tensor w);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pwconv1x1_fp16", &pwconv1x1_fp16, "Pointwise 1x1 FP16 (CUDA)");
}
