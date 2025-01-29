#include <iostream>
#include <cuda_runtime.h>
#include "cpu_kernel.h"
#include "test.h"
#include "cuda_common.cuh"

#define COARSE_FACTOR 4     
#define TILE_SIZE 128   
#define ROWS_A 1024
#define COLS_A 1024
#define ROWS_B 1024
#define COLS_B 1024
#define ELEMENT_WISE false


#define sizeA (sizeof(float) * (ROWS_A * COLS_A))
#define sizeB (sizeof(float) * (ROWS_B * COLS_B))
#define sizeC (sizeof(float) * (ROWS_A * COLS_B))


__global__ void coarsened_tiled_mm(float* A, float* B, float* C) {
    __shared__ float A_s[TILE_SIZE][TILE_SIZE];
    __shared__ float B_s[TILE_SIZE][TILE_SIZE];

    // Accessing A row-wise and B column-wise
    // Idea of Coarsening: Load a Coarse Factor amount of Tiles of B in Shared Memory,
    // while loading "One Tile" of A in Shared Memory
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int colStart = blockIdx.x * blockDim.x * COARSE_FACTOR + threadIdx.x;
    // colStart is the starting column index of the tile in B [0, COARSE_FACTOR, 2*COARSE_FACTOR, ...]

    float sum[COARSE_FACTOR];

    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        sum[c] = 0.0f;
    }

    for(unsigned tile = 0; tile < (COLS_A+ TILE_SIZE-1)/TILE_SIZE; ++tile){ //Assuming COLS_A is larger than ROWS_A
        // For every Tile of A, Load a Coarsed Factor of B
        A_s[threadIdx.y][threadIdx.x] = A[row * COLS_A + tile * TILE_SIZE + threadIdx.x];

        for(unsigned int c = 0; c < COARSE_FACTOR; ++c){
            unsigned int col = colStart + c * TILE_SIZE;
            B_s[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * COLS_B + col];
            __syncthreads();

            for(unsigned int i = 0; i< TILE_SIZE; ++i){
                sum[c] += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
            }
            __syncthreads();
        }
    }
    for(unsigned int c = 0; c < COARSE_FACTOR; ++c){
        unsigned int col = colStart + c * TILE_SIZE;
        C[row * COLS_A + col] = sum[c];
        }
}


void coarsened_tiled_mm_gpu(float* A, float* B, float* C) {
    float *d_A, *d_B, *d_C;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMalloc((void**)&d_A, ROWS_A * COLS_A * sizeof(float));
    cudaMalloc((void**)&d_B, ROWS_B * COLS_B * sizeof(float));
    cudaMalloc((void**)&d_C, ROWS_A * COLS_B * sizeof(float));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "\t\t\t\t GPU Memory Allocation Time: " << milliseconds << "ms" << std::endl;

    cudaEventRecord(start);
    cudaMemcpy(d_A, A, ROWS_A * COLS_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, ROWS_B * COLS_B * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "\t\t\t\t Host to Device Transfer Time: " << milliseconds << "ms" << std::endl;

    dim3 numThreadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((COLS_B + TILE_SIZE - 1) / TILE_SIZE / COARSE_FACTOR,
                   (ROWS_A + TILE_SIZE - 1) / TILE_SIZE);

    cudaEventRecord(start);
    coarsened_tiled_mm<<<numBlocks, numThreadsPerBlock>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize(); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "\t\t\t\t Tiled Kernel Execution Time: " << milliseconds << "ms" << std::endl;

    cudaEventRecord(start);
    cudaMemcpy(C, d_C, ROWS_A * COLS_B * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "\t\t\t\t Device to Host Transfer Time: " << milliseconds << "ms" << std::endl;

    cudaEventRecord(start);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "\t\t\t\t Device Deletion Time: " << milliseconds << "ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


int main() {

    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);
    float *h_C = (float *)malloc(sizeC);
    float *h_C_ref = (float *)malloc(sizeC);

    for (int idx = 0; idx < ROWS_A * COLS_A; idx++) h_A[idx] = idx % 100;
    for (int idx = 0; idx < ROWS_B * COLS_B; idx++) h_B[idx] = idx % 100;

    coarsened_tiled_mm_gpu(h_A, h_B, h_C);

    testResult(h_C, h_C_ref, ROWS_A, COLS_B, ELEMENT_WISE, "Coarsened Tiled Kernel");

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);


    return 0;
}

