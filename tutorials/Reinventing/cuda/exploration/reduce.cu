#include <iostream>

#define BLOCK_DIM 4
#define GRID_DIM 2


__global__ void reduce_kernel(float* input, float* partialSums, unsigned int N) {
    unsigned int segment = ( blockIdx.x*blockDim.x+ threadIdx.x )*2;

    // Perform reduction within the block
    for(unsigned int stride = 1; stride <= BLOCK_DIM; stride *= 2) {
        if(threadIdx.x%stride == 0) {
            input[segment] += input[segment + stride]; }
    __syncthreads() ;
    }
    if(threadIdx.x == 0) {
    partialSums[blockIdx.x] = input[segment];
    }
}


int main() {
    const unsigned int N = 4 * 4;  // Size of input array
    float* input = (float*)malloc(N * sizeof(float));
    float* partialSums = (float*)malloc(GRID_DIM * sizeof(float));

    // Initialize the input array with random values
    for (unsigned int i = 0; i < N; i++) {
        input[i] = i;
    }

    // Allocate device memory
    float* d_input;
    float* d_partialSums;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_partialSums, GRID_DIM * sizeof(float));

    // Copy input data to the device
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the reduction kernel
    reduce_kernel<<<GRID_DIM, BLOCK_DIM>>>(d_input, d_partialSums, N);

    // Copy the partial sums back to the host
    cudaMemcpy(partialSums, d_partialSums, GRID_DIM * sizeof(float), cudaMemcpyDeviceToHost);

    // Perform a final reduction on the partial sums on the host
    float finalSum = 0.0f;
    for (unsigned int i = 0; i < GRID_DIM; i++) {
        finalSum += partialSums[i];
    }

    // Print the result
    printf("Final sum: %f\n", finalSum);

    // Free memory
    free(input);
    free(partialSums);
    cudaFree(d_input);
    cudaFree(d_partialSums);

    return 0;
}
