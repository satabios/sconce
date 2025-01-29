#include <iostream>
#include <cuda_runtime.h>

#define N 4 // Matrix size

// Host function to allocate memory on device
void allocateDeviceMemory(float **h_A, float **d_A, float **h_B, float **d_B, float **h_C, float **d_C) {
    *h_A = (float*)malloc(N * N * sizeof(float));
    *h_B = (float*)malloc(N * N * sizeof(float));
    *h_C = (float*)malloc(N * N * sizeof(float));

    cudaMalloc((void**)&(*d_A), N * N * sizeof(float));
    cudaMalloc((void**)&(*d_B), N * N * sizeof(float));
    cudaMalloc((void**)&(*d_C), N * N * sizeof(float));
}

// Host function to initialize matrices
void initMatrix(float *A, float *B, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i * n + j] = rand() / static_cast<float>(RAND_MAX);
            B[i * n + j] = rand() / static_cast<float>(RAND_MAX);
        }
    }
}

// Host function to copy data from host to device
void copyToDevice(float *h_A, float *d_A, int size) {
    cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice);
}

// Device kernel for matrix addition
__global__ void matAddKernel(float *A, float *B, float *C, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < n) {
        C[i * n + j] = A[i * n + j] + B[i * n + j];
    }
}

// Host function to copy data from device to host
void copyToHost(float *d_C, float *h_C, int size) {
    cudaMemcpy(h_C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);
}

// Host function to free allocated memory
void freeMemory(float *h_A, float *h_B, float *h_C, float *d_A, float *d_B, float *d_C) {
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Host function to print matrix
void printMatrix(float *A, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << A[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    allocateDeviceMemory(&h_A, &d_A, &h_B, &d_B, &h_C, &d_C);

    initMatrix(h_A, h_B, N);
    copyToDevice(h_A, d_A, N * N);
    copyToDevice(h_B, d_B, N * N);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matAddKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    copyToHost(d_C, h_C, N * N);

    std::cout << "Matrix A:" << std::endl;
    printMatrix(h_A, N);
    std::cout << "Matrix B:" << std::endl;
    printMatrix(h_B, N);
    std::cout << "Result Matrix C (A + B):" << std::endl;
    printMatrix(h_C, N);

    freeMemory(h_A, h_B, h_C, d_A, d_B, d_C);

    return 0;
}