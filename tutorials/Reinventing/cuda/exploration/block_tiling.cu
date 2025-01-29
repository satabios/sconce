#include <stdio.h>
#include <cuda_runtime.h>

#define IN_TILE_DIM 8
#define OUT_TILE_DIM 6

__global__ void stencil_kernel(float* in, float* out, unsigned int N) {
  unsigned int i = blockIdx.y * OUT_TILE_DIM + threadIdx.y; // Row index
  unsigned int j = blockIdx.x * OUT_TILE_DIM + threadIdx.x; // Column index

  __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM];

  if (i < N && j < N) {
    printf("Block (%d, %d):\n  Row: %d, Column: %d\n", blockIdx.x, blockIdx.y, i, j);
    // printf(,);
    in_s[threadIdx.y][threadIdx.x] = in[i * N + j];
  }

  __syncthreads();

  // Print the values from one thread per block
//   if (threadIdx.x == 0 && threadIdx.y == 0) {
//     printf("Block (%d, %d):\n", blockIdx.x, blockIdx.y);
//     for (int y = 0; y < IN_TILE_DIM; y++) {
//       for (int x = 0; x < IN_TILE_DIM; x++) {
//         printf("%.2f ", in_s[y][x]);
//       }
//       printf("\n");
//     }
//     printf("\n");
//   }
}

int main() {
  const int N = 16;
  float* in_h;
  float* in_d;
  float* out_d;

  // Allocate memory on the host
  in_h = (float*)malloc(N * N * sizeof(float));

  // Initialize the input matrix on the host
  for (int i = 0; i < N * N; i++) {
    in_h[i] = i;
  }

  // Allocate memory on the device
  cudaMalloc((void**)&in_d, N * N * sizeof(float));
  cudaMalloc((void**)&out_d, N * N * sizeof(float));

  // Copy the input matrix from host to device
  cudaMemcpy(in_d, in_h, N * N * sizeof(float), cudaMemcpyHostToDevice);

  // Define grid and block dimensions
  dim3 blockDim(IN_TILE_DIM, IN_TILE_DIM);
  dim3 gridDim(1, 1);

  // Launch the kernel
  stencil_kernel<<<gridDim, blockDim>>>(in_d, out_d, N);

  // Wait for the kernel to complete
  cudaDeviceSynchronize();

  // Free memory on the device
  cudaFree(in_d);
  cudaFree(out_d);

  // Free memory on the host
  free(in_h);

  return 0;
}