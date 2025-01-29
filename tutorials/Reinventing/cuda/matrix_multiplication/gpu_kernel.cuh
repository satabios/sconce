#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void tiledCoarseMatrixMultiply(float *A, float *B, float *C) {
    // Global Indexes
    unsigned int Gcol = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int Grow = blockDim.x * blockIdx.x * COARSE_FACTOR + threadIdx.x;

    float sum = 0.0f;
    __shared__ float SharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float SharedB[TILE_SIZE][TILE_SIZE];

    // Load the Shared Data
    for (unsigned int tileIdx = 0; tileIdx < COLS_A / TILE_SIZE; tileIdx++) { 
        unsigned int arow = Grow; 
        unsigned int acol = tileIdx * TILE_SIZE + threadIdx.y;

        unsigned int bcol = Gcol; 
        unsigned int brow = tileIdx * TILE_SIZE + threadIdx.x;


        if (brow < ROWS_B && bcol < COLS_B) {
                SharedB[threadIdx.x][threadIdx.y] = B[brow * COLS_B + bcol];
            } else {
                SharedB[threadIdx.x][threadIdx.y] = 0.0f;
            }
        // For One Tile of B Load all the Tiles of A
        for( unsigned coarse_idx = 0; coarse_idx < COARSE_FACTOR; coarse_idx++) {
            if (arow < ROWS_A && acol < COLS_A) {
                SharedA[threadIdx.x][threadIdx.y] = A[arow * COLS_A + acol];
            } else {
                SharedA[threadIdx.x][threadIdx.y] = 0.0f;
            }

            __syncthreads();

            for (int k = 0; k < TILE_SIZE; k++) {
                sum += SharedA[threadIdx.x][k] * SharedB[k][threadIdx.y];
            }
            __syncthreads(); 
        }
    }

    if (Grow < ROWS_A && Gcol < COLS_B) {
        C[Grow * COLS_B + Gcol] = sum;
    }
}

void tiled_coarse_kernel(float *d_A, float *d_B, float *d_C) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(ceil(ROWS_A / TILE_SIZE), ceil(COLS_B / TILE_SIZE));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    tiledCoarseMatrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaDeviceSynchronize();
    std::cout << " \t\t\t\t Tiled Coarse Kernel Execution Time: " << milliseconds << "ms" << std::endl;
}

__global__ void tiledMatrixMultiply(float *A, float *B, float *C) {
    // Global Indexes
    unsigned int Gcol = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int Grow = blockDim.x * blockIdx.x + threadIdx.x;

    float sum = 0.0f;
    __shared__ float SharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float SharedB[TILE_SIZE][TILE_SIZE];

    // Load the Shared Data
    for (unsigned int tileIdx = 0; tileIdx < COLS_A / TILE_SIZE; tileIdx++) { 
        unsigned int arow = Grow; 
        unsigned int acol = tileIdx * TILE_SIZE + threadIdx.y;

        unsigned int bcol = Gcol; 
        unsigned int brow = tileIdx * TILE_SIZE + threadIdx.x;

        if (arow < ROWS_A && acol < COLS_A) {
            SharedA[threadIdx.x][threadIdx.y] = A[arow * COLS_A + acol];
        } else {
            SharedA[threadIdx.x][threadIdx.y] = 0.0f;
        }

        if (brow < ROWS_B && bcol < COLS_B) {
            SharedB[threadIdx.x][threadIdx.y] = B[brow * COLS_B + bcol];
        } else {
            SharedB[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += SharedA[threadIdx.x][k] * SharedB[k][threadIdx.y];
        }
        __syncthreads(); 
    }

    if (Grow < ROWS_A && Gcol < COLS_B) {
        C[Grow * COLS_B + Gcol] = sum;
    }
}

void tiled_kernel(float *d_A, float *d_B, float *d_C) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(ceil(ROWS_A / TILE_SIZE), ceil(COLS_B / TILE_SIZE));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    tiledMatrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << " \t\t\t\t Tiled Kernel Execution Time: " << milliseconds << "ms" << std::endl;    
}

