#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 1024
#define COARSE_FACTOR 2
#define BLOCK_DIM (N / (2 * COARSE_FACTOR))

// Kernel for shared memory reduction with coarsening
__global__ void shared_reduce_kernel_coalesced_coarsened(float* input, float* partialSums) {
    unsigned int segment = blockIdx.x * blockDim.x * 2 * COARSE_FACTOR;
    unsigned int i = segment + threadIdx.x;

    __shared__ float input_s[BLOCK_DIM];
    float sum = 0.0f;

    for (unsigned int tile = 0; tile < COARSE_FACTOR * 2; ++tile) {
    if (i + tile * blockDim.x < N) { // Ensure index bounds
        sum += input[i + tile * blockDim.x];
    }
}
    input_s[threadIdx.x] = sum;

    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            input_s[threadIdx.x] += input_s[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partialSums[blockIdx.x] = input_s[0];
    }
}

// Kernel for shared memory reduction (no coarsening)
__global__ void shared_reduced_kernel_coalesced(float* input, float* partialSums) {
    unsigned int segment = blockIdx.x * blockDim.x * 2;
    unsigned int i = segment + threadIdx.x;

    __shared__ float sharedData[BLOCK_DIM];
    sharedData[threadIdx.x] = input[i] + input[i + blockDim.x];
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            sharedData[threadIdx.x] += sharedData[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partialSums[blockIdx.x] = sharedData[0];
    }
}

// Kernel for reduction without shared memory
__global__ void reduce_kernel_coalesced(float* input, float* partialSums) {
    unsigned int segment = blockIdx.x * blockDim.x * 2;
    unsigned int i = segment + threadIdx.x;

    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partialSums[blockIdx.x] = input[i];
    }
}

// Naive kernel for reduction
__global__ void naive_reduce_kernel(float* input, float* partialSums) {
    unsigned int segment = blockIdx.x * blockDim.x * 2;
    unsigned int i = segment + threadIdx.x * 2;

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partialSums[blockIdx.x] = input[i];
    }
}

int main() {
    // Allocate and initialize host memory
    float* input = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        input[i] = 1.0f; // Initialize with 1.0 for testing
    }

    // Allocate device memory
    float *input_d, *partialsums_d;
    cudaMalloc((void**)&input_d, N * sizeof(float));
    cudaMemcpy(input_d, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate memory for partial sums
    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numBlocks = N / (numThreadsPerBlock * 2);
    float* partialsums = (float*)malloc(numBlocks * sizeof(float));
    cudaMalloc((void**)&partialsums_d, numBlocks * sizeof(float));

    // Lambda to run kernels and time execution
    auto runKernelAndTime = [&](auto kernel, const char* kernelName) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, partialsums_d);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("%s Kernel Time: %.3f ms\n", kernelName, milliseconds);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    };

    // Run kernels and time them
    runKernelAndTime(shared_reduce_kernel_coalesced_coarsened, "shared_reduce_kernel_coalesced_coarsened");
    runKernelAndTime(shared_reduced_kernel_coalesced, "shared_reduced_kernel_coalesced");
    runKernelAndTime(reduce_kernel_coalesced, "reduce_kernel_coalesced");
    runKernelAndTime(naive_reduce_kernel, "naive_reduce_kernel");

    // Copy partial sums back to host
    cudaMemcpy(partialsums, partialsums_d, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Final reduction on host
    float sum = 0.0f;
    for (unsigned int i = 0; i < numBlocks; ++i) {
        sum += partialsums[i];
    }

    // Print the final result
    printf("Final sum: %f\n", sum);

    // Free memory
    cudaFree(input_d);
    cudaFree(partialsums_d);
    free(input);
    free(partialsums);

    return 0;
}
