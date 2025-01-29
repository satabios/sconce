#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <torch/extension.h>

__global__ void linear_w8a8_kernel(
    const int8_t* __restrict__ A,  // Input activations
    const int8_t* __restrict__ B,  // Weights
    int32_t* __restrict__ C,       // Output
    const int m, const int n, const int k) {

    // Calculate row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        int32_t sum = 0;
        for (int i = 0; i < k; ++i) {
            sum += static_cast<int32_t>(A[row * k + i]) * static_cast<int32_t>(B[i * n + col]);
        }
        C[row * n + col] = sum;
    }
}

void linear_w8a8_cuda(const torch::Tensor& input, const torch::Tensor& weights, torch::Tensor& output) {
    const auto m = input.size(0);
    const auto k = input.size(1);
    const auto n = weights.size(1);

    const dim3 block_size(16, 16);
    const dim3 num_blocks((n + block_size.x - 1) / block_size.x, (m + block_size.y - 1) / block_size.y);

    AT_DISPATCH_INTEGRAL_TYPES(input.scalar_type(), "linear_w8a8_cuda", ([&] {
        linear_w8a8_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<int8_t>(),
            weights.data_ptr<int8_t>(),
            output.data_ptr<int32_t>(),
            m, n, k);
    }));
}

