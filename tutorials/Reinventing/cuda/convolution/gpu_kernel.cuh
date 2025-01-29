#include <cuda_runtime.h>
#include "device_launch_parameters.h"


__constant__ float d_kernel[K * K];

__global__ void constMemTiledConvolution(float* input_data, float* output_data, int out_shape) {
    // Shared memory for the tile
    __shared__ float tile[K + 2 * padding][K + 2 * padding];

    int OutRow = blockIdx.x * blockDim.x + threadIdx.x;
    int OutCol = blockIdx.y * blockDim.y + threadIdx.y;

    int InRow = OutRow * stride - padding;
    int InCol = OutCol * stride - padding;

    if (InRow >= 0 && InRow < N && InCol >= 0 && InCol < N) {
        tile[threadIdx.x + padding][threadIdx.y + padding] = input_data[InRow * N + InCol];
    } else {
        tile[threadIdx.x + padding][threadIdx.y + padding] = 0.0f; 
    }
    __syncthreads();

    if (OutRow >= 0 && OutRow < out_shape && OutCol >= 0 && OutCol < out_shape) {
        float sum = 0.0f;

        for (int KRow = 0; KRow < K; KRow++) {
            for (int KCol = 0; KCol < K; KCol++) {
                int SharedRow = threadIdx.x + KRow;
                int SharedCol = threadIdx.y + KCol;

                sum += d_kernel[KRow * K + KCol] * tile[SharedRow][SharedCol];
            }
        }

        output_data[OutRow * out_shape + OutCol] = sum;
    }
}

void const_mem_tiled_conv(float *input_data, float *output_data, int out_shape){
    dim3 ThreadsPerBlock(K,K);
    dim3 BlocksPerGrid((out_shape+K-1)/K, (out_shape+K-1)/K); 

        cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    constMemTiledConvolution<<<BlocksPerGrid, ThreadsPerBlock>>>(input_data, output_data, out_shape);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << " \t\t\t\t Const Memory Tiled Execution Time: " << milliseconds << "ms" << std::endl;    
}



__global__ void constMemConvolution(float * input_data, float * output_data, int out_shape){
    // Calculate the row and column index of the output matrix
    int OutRow = blockIdx.x * blockDim.x + threadIdx.x;
    int OutCol = blockIdx.y * blockDim.y + threadIdx.y;

    if (OutRow < out_shape && OutCol < out_shape) {
        float sum = 0.0f;
        for (int KRow = 0; KRow < K; KRow++) {
            for (int KCol = 0; KCol < K; KCol++) {
                int InRow = OutRow * stride + KRow - padding;
                int InCol = OutCol * stride + KCol - padding;

                // Check if the input indices are within bounds
                if (InRow >= 0 && InRow < N && InCol >= 0 && InCol < N) {
                    sum += d_kernel[KRow * K + KCol] * input_data[InRow * N + InCol];
                }
            }
        }

        output_data[OutRow * out_shape + OutCol] = sum;
    }
}


void const_mem_conv(float *input_data, float *output_data, int out_shape){
    dim3 ThreadsPerBlock(K,K);
    dim3 BlocksPerGrid((out_shape+K-1)/K, (out_shape+K-1)/K); 

        cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    constMemConvolution<<<BlocksPerGrid, ThreadsPerBlock>>>(input_data, output_data, out_shape);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << " \t\t\t\t Const Memory Kernel Execution Time: " << milliseconds << "ms" << std::endl;    
}

__global__ void naiveConvolution(float * input_data, float *kernel_data, float * output_data, int out_shape){
    int OutRow = blockDim.x * blockIdx.x + threadIdx.x;
    int OutCol = blockDim.y * blockIdx.y + threadIdx.y;

    if(OutCol>=0 && OutRow< out_shape && OutCol>=0 && OutCol<out_shape){
        float sum = 0.0f;
        for(int KRow=0; KRow<K; KRow++){
            for(int KCol=0; KCol<K; KCol++){
                int InRow = OutRow * stride + KRow - padding; 
                int InCol = OutCol * stride + KCol - padding;
                if(InCol>=0 && InCol<N && InRow>=0 && InRow<N){
                    sum+= kernel_data[KRow*K+ KCol] * input_data[InRow*N+ InCol];
                }

            }
        }
        output_data[OutRow*out_shape+ OutCol]=sum;
    }

}

void naive_conv(float *input_data, float * kernel_data, float *output_data, int out_shape){
    dim3 ThreadsPerBlock(K,K);
    dim3 BlocksPerGrid((out_shape+K-1)/K, (out_shape+K-1)/K); 

        cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    naiveConvolution<<<BlocksPerGrid, ThreadsPerBlock>>>(input_data, kernel_data, output_data, out_shape);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << " \t\t\t\t Naive Kernel Execution Time: " << milliseconds << "ms" << std::endl;    
}

