#include <torch/extension.h>
torch::Tensor dwconv3x3_fp16(torch::Tensor x, torch::Tensor w);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dwconv3x3_fp16", &dwconv3x3_fp16, "Depthwise 3x3 FP16 (CUDA)");
}
