#include <iostream>
#include <cuda_runtime.h>
#include "cpu_kernel.h"
#include "test.h"

#define BLOCK_SIZE 128
#define ROWS_A 1024
#define COLS_A 1024
#define ROWS_B 1024
#define COLS_B 1024
#define ELEMENT_WISE false


#define sizeA (sizeof(float) * (ROWS_A * COLS_A))
#define sizeB (sizeof(float) * (ROWS_B * COLS_B))
#define sizeC (sizeof(float) * (ROWS_A * COLS_B))

__global__ void naiveMatrixMultiply(float *A, float *B, float *C) {
    unsigned int Gcol = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int Grow = blockDim.x * blockIdx.x + threadIdx.x;

    if (Grow < ROWS_A && Gcol < COLS_B) {
        float sum = 0.0f;
        for (int k = 0; k < COLS_A; k++) {
            sum += A[Grow * COLS_A + k] * B[k * COLS_B + Gcol];
        }
        C[Grow * COLS_B + Gcol] = sum;
    }
}

void naive_kernel(float *h_A, float *h_B, float *h_C) {

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

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid(ceil(ROWS_A / BLOCK_SIZE), ceil(COLS_B / BLOCK_SIZE));

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    naiveMatrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << " \t\t\t\t Naive Kernel Execution Time: " << milliseconds << "ms" << std::endl;

    cudaEventRecord(start);
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "\t\t\t\t Device to Host Transfer Time: " << milliseconds << "ms" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(){

    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);
    float *h_C = (float *)malloc(sizeC);
    float *h_C_ref = (float *)malloc(sizeC);

    for (int idx = 0; idx < ROWS_A * COLS_A; idx++) h_A[idx] = idx % 100;
    for (int idx = 0; idx < ROWS_B * COLS_B; idx++) h_B[idx] = idx % 100;

    MatrixMulCPU(h_A, h_B, h_C_ref, ROWS_A, COLS_A, COLS_B);

    naive_kernel(h_A, h_B, h_C);

    testResult(h_C, h_C_ref, ROWS_A, COLS_B, ELEMENT_WISE, "Naive Kernel");


    free(h_A);
    free(h_B);
    free(h_C);

    return 0;

}