#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_common.cuh"

#define NUM_BINS 256

__global__ void histogram_kernel(const unsigned char* input, unsigned int* histogram, int size) { 
  __shared__ unsigned int local_histogram[NUM_BINS];

  for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
    local_histogram[i] = 0;
  }
  __syncthreads();

  int global_start = blockIdx.x * blockDim.x  + threadIdx.x;
  int stride = blockDim.x * gridDim.x ;

  for (int i = global_start; i < size; i += stride) {
      unsigned char value = input[i ];
      atomicAdd(&local_histogram[value], 1);
  
  }
  __syncthreads();

  for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
    atomicAdd(&histogram[i], local_histogram[i]);
  }
}



int main() {
  int width = 512;
  int height = 512;
  int size = width * height;

  unsigned char* image = (unsigned char*)malloc(size * sizeof(unsigned char));
  unsigned int* bins = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));

  for (int i = 0; i < size; i++) {
    image[i] = rand() % 256;
  }

  unsigned char* d_image;
  unsigned int* d_bins;
  CUDA_CHECK(cudaMalloc((void**)&d_image, size * sizeof(unsigned char)));
  CUDA_CHECK(cudaMalloc((void**)&d_bins, NUM_BINS * sizeof(unsigned int)));

  CUDA_CHECK(cudaMemcpy(d_image, image, size * sizeof(unsigned char), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemset(d_bins, 0, NUM_BINS * sizeof(unsigned int))); 
  
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));

  int threadsPerBlock = 1024; // Higher thread count for coarsening
  int blocksPerGrid = (size + threadsPerBlock - 1) / (threadsPerBlock);
  histogram_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_bins, size);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

  printf("Kernel execution time: %.3f ms\n", milliseconds);

  CUDA_CHECK(cudaMemcpy(bins, d_bins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  free(image);
  free(bins);
  CUDA_CHECK(cudaFree(d_image));
  CUDA_CHECK(cudaFree(d_bins));

  return 0;
}
