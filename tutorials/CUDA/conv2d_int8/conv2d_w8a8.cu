#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void conv2d_kernel_int8(
    const int8_t* __restrict__ input,
    const int8_t* __restrict__ kernel,
    int32_t* __restrict__ output,
    int inputWidth, int inputHeight,
    int kernelWidth, int kernelHeight,
    int stride, int padding,
    int outputWidth, int outputHeight,
    int inChannels, int outChannels,
    int batchSize) {

    extern __shared__ int8_t sharedMemory[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;

    const int x = blockIdx.x * blockDim.x + tx;
    const int y = blockIdx.y * blockDim.y + ty;
    const int out_c = blockIdx.z % outChannels;
    const int b = blockIdx.z / outChannels;

    const int sharedInputWidth = blockDim.x + kernelWidth - 1;
    const int sharedInputHeight = blockDim.y + kernelHeight - 1;

    int8_t* sharedInput = sharedMemory;
    int8_t* sharedKernel = sharedMemory + sharedInputWidth * sharedInputHeight * inChannels;

    // Load input into shared memory
    for (int c = 0; c < inChannels; ++c) {
        for (int i = ty; i < sharedInputHeight; i += blockDim.y) {
            for (int j = tx; j < sharedInputWidth; j += blockDim.x) {
                int inputX = blockIdx.x * blockDim.x + j - padding;
                int inputY = blockIdx.y * blockDim.y + i - padding;

                if (inputX >= 0 && inputX < inputWidth && inputY >= 0 && inputY < inputHeight) {
                    sharedInput[(c * sharedInputHeight + i) * sharedInputWidth + j] = input[((b * inChannels + c) * inputHeight + inputY) * inputWidth + inputX];
                } else {
                    sharedInput[(c * sharedInputHeight + i) * sharedInputWidth + j] = 0;
                }
            }
        }
    }

    // Load kernel into shared memory
    for (int i = ty; i < kernelHeight; i += blockDim.y) {
        for (int j = tx; j < kernelWidth; j += blockDim.x) {
            for (int c = 0; c < inChannels; ++c) {
                sharedKernel[(c * kernelHeight + i) * kernelWidth + j] = kernel[((out_c * inChannels + c) * kernelHeight + i) * kernelWidth + j];
            }
        }
    }

    __syncthreads();

    if (x < outputWidth && y < outputHeight) {
        int32_t sum = 0;

        for (int c = 0; c < inChannels; ++c) {
            for (int i = 0; i < kernelHeight; i++) {
                for (int j = 0; j < kernelWidth; j++) {
                    int inputX = tx + j;
                    int inputY = ty + i;

                    sum += static_cast<int32_t>(sharedInput[(c * sharedInputHeight + inputY) * sharedInputWidth + inputX]) *
                           static_cast<int32_t>(sharedKernel[(c * kernelHeight + i) * kernelWidth + j]);
                }
            }
        }

        output[((b * outChannels + out_c) * outputHeight + y) * outputWidth + x] = sum;
    }
}


void conv2d_int8(torch::Tensor input, torch::Tensor kernel, torch::Tensor output, 
                 int stride, int padding) {
    const int batch_size = input.size(0);
    const int channel_in = input.size(1);
    const int inputWidth = input.size(3);
    const int inputHeight = input.size(2);
    const int channel_out = kernel.size(0);
    const int kernelWidth = kernel.size(3);
    const int kernelHeight = kernel.size(2);

    const int outputWidth = (inputWidth + 2 * padding - kernelWidth) / stride + 1;
    const int outputHeight = (inputHeight + 2 * padding - kernelHeight) / stride + 1;

    output.resize_({batch_size, channel_out, outputHeight, outputWidth});

    const dim3 blockSize(16, 16); 
    const dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x, (outputHeight + blockSize.y - 1) / blockSize.y, batch_size * channel_out);
    size_t sharedMemorySize = (blockSize.x + kernelWidth - 1) * (blockSize.y + kernelHeight - 1) * channel_in * sizeof(int8_t) +
                              kernelWidth * kernelHeight * channel_in * sizeof(int8_t);

    conv2d_kernel_int8<<<gridSize, blockSize, sharedMemorySize>>>(
        input.data_ptr<int8_t>(),
        kernel.data_ptr<int8_t>(),
        output.data_ptr<int32_t>(),
        inputWidth, inputHeight,
        kernelWidth, kernelHeight,
        stride, padding,
        outputWidth, outputHeight,
        channel_in, channel_out,
        batch_size
    );

    cudaDeviceSynchronize();
}

// Python binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_int8", &conv2d_int8, "2D Convolution with int8");
}