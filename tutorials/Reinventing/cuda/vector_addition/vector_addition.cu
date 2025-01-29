#include<stdio.h>
#include "cuda_common.cuh"

#define BLOCK_SIZE 32
#define N 128

__global__ void vector_addition(int *a, int *b, int *c){
    
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<N){
        // remove to see the effect of synchronization
        // printf("ThreadIdx: %d, BlockIdx: %d blockDim: %d\n", threadIdx.x, blockIdx.x, blockDim.x);
        c[i] = a[i] + b[i];
    }
}



int main(){
    

    int *a, *b, *c; // host variables
    int *d_a, *d_b, *d_c; // device variables
    // Memory allocation on host
    a = (int*)malloc(N*sizeof(int)); 
    b = (int*)malloc(N*sizeof(int));
    c = (int*)malloc(N*sizeof(int));
    // Memory allocation on device
    CUDA_CHECK(cudaMalloc(&d_a, N*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_b, N*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_c, N*sizeof(int)));

    for(int i=0; i<N; i++){
        a[i] = i;
        b[i] = i;
    }

    CUDA_CHECK(cudaMemcpy(d_a, a, N*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, N*sizeof(int), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((N+threadsPerBlock.x-1)/threadsPerBlock.x); // (16+16-1)/16 = 2 ; General Formula: (n+blockDim-1)/blockDim for number that are not multiple of blockDim
    
    cudaEvent_t start_kernel, stop_kernel;
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));

    CUDA_CHECK(cudaEventRecord(start_kernel));
    vector_addition<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);
    CUDA_CHECK(cudaEventRecord(stop_kernel));

    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaEventSynchronize(stop_kernel));

    float milliseconds_kernel = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds_kernel, start_kernel, stop_kernel));
    printf("Vector Add - elapsed time: %f ms\n", milliseconds_kernel);

    cudaMemcpy(c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);

    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}