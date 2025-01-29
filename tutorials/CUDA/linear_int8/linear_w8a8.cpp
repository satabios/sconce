#include <torch/extension.h>

// CUDA forward declarations
void linear_w8a8_cuda(const torch::Tensor& input, const torch::Tensor& weights, torch::Tensor& output);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor linear_w8a8(torch::Tensor input, torch::Tensor weights) {
    CHECK_INPUT(input);
    CHECK_INPUT(weights);

    auto output = torch::zeros({input.size(0), weights.size(1)}, torch::TensorOptions().dtype(torch::kInt32).device(input.device()));

    linear_w8a8_cuda(input, weights, output);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_w8a8", &linear_w8a8, "Linear layer with W8A8 quantization (CUDA)");
}

