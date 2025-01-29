#include <iostream>
#include <cuda_runtime.h>
#include "cpu_kernel.h"
#include "test.h"

#define TILE_SIZE 128
#define ROWS_A 1024
#define COLS_A 1024
#define ROWS_B 1024
#define COLS_B 1024
#define ELEMENT_WISE false

#define sizeA (sizeof(float) * (ROWS_A * COLS_A))
#define sizeB (sizeof(float) * (ROWS_B * COLS_B))
#define sizeC (sizeof(float) * (ROWS_A * COLS_B))


// Kernel for tiled matrix multiplication
__global__ void tiledMatrixMultiply(float *A, float *B, float *C) {
    // Calculate global row and column indices
    unsigned int Gcol = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int Grow = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    // Shared memory for tiles
    __shared__ float SharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float SharedB[TILE_SIZE][TILE_SIZE];

    // Loop over all tiles required
    for (unsigned int tileIdx = 0; tileIdx < (COLS_A + TILE_SIZE - 1) / TILE_SIZE; tileIdx++) {
        // Load tiles into shared memory
        unsigned int arow = Grow;
        unsigned int acol = tileIdx * TILE_SIZE + threadIdx.y;
        unsigned int brow = tileIdx * TILE_SIZE + threadIdx.x;
        unsigned int bcol = Gcol;

        SharedA[threadIdx.x][threadIdx.y] = (arow < ROWS_A && acol < COLS_A) ? A[arow * COLS_A + acol] : 0.0f;
        SharedB[threadIdx.x][threadIdx.y] = (brow < ROWS_B && bcol < COLS_B) ? B[brow * COLS_B + bcol] : 0.0f;

        __syncthreads();

        // Perform partial computation for the tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += SharedA[threadIdx.x][k] * SharedB[k][threadIdx.y];
        }
        __syncthreads();
    }

    // Write final result to global memory
    if (Grow < ROWS_A && Gcol < COLS_B) {
        C[Grow * COLS_B + Gcol] = sum;
    }
}

void tiled_kernel(float *h_A, float *h_B, float *h_C) {

    float *d_A, *d_B, *d_C; 

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "\t\t\t\t GPU Memory Allocation Time: " << milliseconds << "ms" << std::endl;

    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "\t\t\t\t Host to Device Transfer Time: " << milliseconds << "ms" << std::endl;

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((ROWS_A + TILE_SIZE - 1) / TILE_SIZE, (COLS_B + TILE_SIZE - 1) / TILE_SIZE);

    cudaEventRecord(start);
    tiledMatrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    milliseconds = 0; 
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "\t\t\t\t Tiled Kernel Execution Time: " << milliseconds << "ms" << std::endl;


    cudaEventRecord(start);
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
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

    MatrixMulCPU(h_A, h_B, h_C_ref, ROWS_A, COLS_A, COLS_B);

    tiled_kernel(h_A, h_B, h_C);

    testResult(h_C, h_C_ref, ROWS_A, COLS_B, ELEMENT_WISE, "Tiled Kernel");

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    return 0;
}