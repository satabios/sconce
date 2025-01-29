#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "cpu_kernel.h" 
#include "gpu_kernel.cuh"
#include "test.h"

int main() {

    float* kernel = (float*)malloc(sizeof(float) * (K * K));
    float* input_data = (float*)malloc(sizeof(float) * (N * N));
    
    int out_shape = (N-K+ 2*(padding))/stride + 1;
    float* output_naive = (float *)malloc(sizeof(float) * (out_shape* out_shape));
    float* output_const = (float *)malloc(sizeof(float) * (out_shape* out_shape));
    float* output_tiled = (float *)malloc(sizeof(float) * (out_shape* out_shape));
    float* output_cpu = (float*)malloc(sizeof(float) * (out_shape * out_shape)); // Allocate memory for CPU output

    for (int idx = 0; idx < N * N; idx++) input_data[idx] = idx;
    for (int idx = 0; idx < K * K; idx++) {
        if (idx % 4 == 0) kernel[idx] = 1;
        else kernel[idx] = 0;
    }

    float *d_input_data, *kernel_data, *d_output_naive, *d_output_const, *d_output_tiled;

    cudaMalloc((void**)&d_input_data, sizeof(float) * (N * N));
    cudaMalloc((void**)&d_output_naive, sizeof(float) * (out_shape * out_shape)); 
    cudaMalloc((void**)&d_output_const, sizeof(float) * (out_shape * out_shape)); 
    cudaMalloc((void**)&d_output_tiled, sizeof(float) * (out_shape * out_shape)); 
    cudaMalloc((void**)&kernel_data, sizeof(float) * (K * K)); 
    cudaMemcpyToSymbol(d_kernel, kernel, sizeof(float) * (K * K), 0, cudaMemcpyHostToDevice); 
    
    cudaMemcpy(d_input_data, input_data, sizeof(float) * (N * N), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_data, kernel, sizeof(float) * (K * K), cudaMemcpyHostToDevice);

    cpu_convolution(input_data, kernel, output_cpu, out_shape);
    naive_conv(d_input_data, kernel_data, d_output_naive, out_shape);
    const_mem_conv(d_input_data, d_output_const, out_shape);
    const_mem_tiled_conv(d_input_data, d_output_tiled, out_shape);

    cudaMemcpy(output_naive, d_output_naive, sizeof(float) * (out_shape * out_shape), cudaMemcpyDeviceToHost); 
    cudaMemcpy(output_const, d_output_const, sizeof(float) * (out_shape * out_shape), cudaMemcpyDeviceToHost); 
    cudaMemcpy(output_tiled, d_output_tiled, sizeof(float) * (out_shape * out_shape), cudaMemcpyDeviceToHost); 

    

    testResult(output_naive, output_cpu, out_shape, "Naive Kernel");
    testResult(output_const, output_cpu, out_shape, "Constant Memory Kernel");
    testResult(output_tiled, output_cpu, out_shape, "Constant Memory Tiled Kernel");


    cudaFree(d_input_data);
    cudaFree(d_output_naive);
    cudaFree(d_output_tiled);

    free(input_data);
    free(kernel);
    free(output_naive);
    free(output_const);
    free(output_tiled);
    free(output_cpu); // Free CPU output memory

    return 0;
}