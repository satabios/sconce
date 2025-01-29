#include "device_launch_parameters.h"

#ifndef CPU_KERNEL_H
#define CPU_KERNEL_H

void cpu_convolution(float* input_data, float * kernel_data, float* output_data, int out_shape) {
    for (int OutRow = 0; OutRow < out_shape; OutRow++) {
        for (int OutCol = 0; OutCol < out_shape; OutCol++) {
            float sum = 0.0f;
            for (int KRow = 0; KRow < K; KRow++) {
                for (int KCol = 0; KCol < K; KCol++) {
                    int InRow = OutRow * stride + KRow - padding;
                    int InCol = OutCol * stride + KCol - padding;
                    if (InCol >= 0 && InCol < N && InRow >= 0 && InRow < N) {
                        sum += kernel_data[KRow * K + KCol] * input_data[InRow * N + InCol];
                    }
                }
            }
            output_data[OutRow * out_shape + OutCol] = sum;
        }
    }
}

#endif // CPU_KERNEL_H