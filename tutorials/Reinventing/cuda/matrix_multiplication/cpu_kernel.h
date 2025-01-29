#include <iostream>
#include "device_launch_parameters.h"


void MatrixMulCPU(float *A, float *B, float *C, int ROWS_A, int COLS_A, int COLS_B){
    for(int i = 0; i < ROWS_A; i++){
        for(int j = 0; j < COLS_B; j++){
            float sum = 0;
            for(int k = 0; k < COLS_A; k++){
                sum += A[i * COLS_A + k] * B[k * COLS_B + j];
            }
            C[i * COLS_B + j] = sum;
        }
    }
}