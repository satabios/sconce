#include <iostream>

void initialize_data(float *data, int dim1, int dim2, int dim3, int dim4) {
    for(int i = 0; i < dim1; i++) {
        for(int j = 0; j < dim2; j++) {
            for(int k = 0; k < dim3; k++) {
                for(int l = 0; l < dim4; l++)
                    data[i * dim2 * dim3 * dim4 + j * dim3 * dim4 + k * dim4 + l] = float(i + j + k + l);
            }
        }
    }
}

void display_data(float *data, int dim1, int dim2, int dim3, int dim4) {
    for(int i = 0; i < dim1; i++) {
        for(int j = 0; j < dim2; j++) {
            for(int k = 0; k < dim3; k++) {
                for(int l = 0; l < dim4; l++)
                    std::cout << data[i * dim2 * dim3 * dim4 + j * dim3 * dim4 + k * dim4 + l] << " ";
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}


void conv2d(float *feature_map, float *kernel, float *feature_out_data,
            int batch_size, int feature_map_height, int feature_map_width, int feature_map_channel,
            int kernel_size, int kernel_channel_out,
            int feature_out_height, int feature_out_width,
            int padding, int stride) {

    // Precompute constants for array indexing
    int feature_map_hw = feature_map_height * feature_map_width;
    int kernel_ks_ks = kernel_size * kernel_size;
    int feature_out_hw = feature_out_height * feature_out_width;

    #pragma omp parallel for collapse(2) // Parallelize outer loops for multi-core CPUs
    for (int c_out = 0; c_out < kernel_channel_out; c_out++) {
        for (int bidx = 0; bidx < batch_size; bidx++) {
            for (int height = 0; height < feature_out_height; height++) {
                for (int width = 0; width < feature_out_width; width++) {
                    int h_start = height * stride - padding;
                    int w_start = width * stride - padding;

                    float sum = 0.0f;

                    for (int c_in = 0; c_in < feature_map_channel; c_in++) {
                        int kernel_offset = c_out * (feature_map_channel * kernel_ks_ks) + c_in * kernel_ks_ks;
                        int feature_map_offset = bidx * (feature_map_channel * feature_map_hw) + c_in * feature_map_hw;

                        // Unrolling the kernel loops
                        for (int k_h = 0; k_h < kernel_size; k_h++) {
                            for (int k_w = 0; k_w < kernel_size; k_w += 2) { // Unroll by factor of 2
                                int h_in = h_start + k_h;
                                int w_in1 = w_start + k_w;
                                int w_in2 = w_in1 + 1;
                                int w_in3 = w_in2 + 1;

                                if (h_in >= 0 && h_in < feature_map_height) {
                                    if (w_in1 >= 0 && w_in1 < feature_map_width) {
                                        sum += feature_map[feature_map_offset + h_in * feature_map_width + w_in1] *
                                               kernel[kernel_offset + k_h * kernel_size + k_w];
                                    }
                                    if (w_in2 < feature_map_width && k_w + 1 < kernel_size) {
                                        sum += feature_map[feature_map_offset + h_in * feature_map_width + w_in2] *
                                               kernel[kernel_offset + k_h * kernel_size + (k_w + 1)];
                                    }
                                    if (w_in3 < feature_map_width && k_w + 1 < kernel_size) {
                                        sum += feature_map[feature_map_offset + h_in * feature_map_width + w_in3] *
                                               kernel[kernel_offset + k_h * kernel_size + (k_w + 1)];
                                    }
                                }
                            }
                        }
                    }

                    // Store the result
                    feature_out_data[bidx * (kernel_channel_out * feature_out_hw) +
                                     c_out * feature_out_hw +
                                     height * feature_out_width +
                                     width] = sum;
                }
            }
        }
    }
}


void conv2d_naive(float *feature_map, float *kernel, float *feature_out_data,
            int batch_size, int feature_map_height, int feature_map_width, int feature_map_channel,
            int kernel_size, int kernel_channel_out,
            int feature_out_height, int feature_out_width,
            int padding, int stride) {


    for(int c_out = 0; c_out < kernel_channel_out; c_out++) {
        for(int bidx = 0; bidx < batch_size; bidx++) {
            for(int height = 0; height < feature_out_height; height++) {
                for(int width = 0; width < feature_out_width; width++) {
                    int h_start = height * stride - padding;
                    int w_start = width * stride - padding;

                    float sum = 0.0f;

                    for(int c_in = 0; c_in < feature_map_channel; c_in++) {
                        for(int k_h = 0; k_h < kernel_size; k_h++) {
                            for(int k_w = 0; k_w < kernel_size; k_w++) {
                                int h_in = h_start + k_h;
                                int w_in = w_start + k_w;

                                if(h_in >= 0 && h_in < feature_map_height && w_in >= 0 && w_in < feature_map_width) {
                                    int feature_map_idx = bidx * (feature_map_channel * feature_map_height * feature_map_width) + 
                                                          c_in * (feature_map_height * feature_map_width) + 
                                                          h_in * feature_map_width + 
                                                          w_in;

                                    int kernel_idx = c_out * (feature_map_channel * kernel_size * kernel_size) + 
                                                     c_in * (kernel_size * kernel_size) + 
                                                     k_h * kernel_size + 
                                                     k_w;

                                    sum += feature_map[feature_map_idx] * kernel[kernel_idx];
                                }
                            }
                        }
                    }

                    feature_out_data[bidx * kernel_channel_out * feature_out_height * feature_out_width +
                                     c_out * feature_out_height * feature_out_width +
                                     height * feature_out_width +
                                     width] = sum;
                }
            }
        }
    }
}

int main() {
    int kernel_width = 3, kernel_height = 3, kernel_in_channels = 1, kernel_out_channels = 2;
    int feature_in_channels = 1, feature_height = 5, feature_width = 5;
    int batch_size = 1;
    int padding = 0, stride = 1;  // Account for Padding
    
    //TODO: Add suppor for dilations and groups

    // Define kernel and feature map
    float kernel[kernel_out_channels][kernel_in_channels][kernel_height][kernel_width];
    float feature_map[batch_size][feature_in_channels][feature_height][feature_width];

    initialize_data(reinterpret_cast<float*>(&kernel[0][0][0][0]), kernel_out_channels, kernel_in_channels, kernel_height, kernel_width);
    initialize_data(reinterpret_cast<float*>(&feature_map[0][0][0][0]), batch_size, feature_in_channels, feature_height, feature_width);

    int feature_out_height = (feature_height - kernel_height + 2 * padding) / stride + 1;
    int feature_out_width = (feature_width - kernel_width + 2 * padding) / stride + 1;
    float feature_out_data[batch_size * kernel_out_channels * feature_out_height * feature_out_width];

    // Run convolution
    conv2d(reinterpret_cast<float*>(&feature_map[0][0][0]), reinterpret_cast<float*>(&kernel[0][0][0][0]), feature_out_data,
                                                                                     batch_size, feature_height, feature_width, feature_in_channels,
                                                                                     kernel_height, kernel_out_channels,
                                                                                     feature_out_height, feature_out_width,
                                                                                     padding, stride);

    std::cout << "Kernel Map:\n";
    display_data(reinterpret_cast<float*>(&kernel[0][0][0]), kernel_out_channels, kernel_in_channels, kernel_height, kernel_width);

    // Display input and output
    std::cout << "Feature Map:\n";
    display_data(reinterpret_cast<float*>(&feature_map[0][0][0]), batch_size, feature_in_channels, feature_height, feature_width);

    std::cout << "\nOutput Feature Map:\n";
    display_data(feature_out_data, batch_size, kernel_out_channels, feature_out_height, feature_out_width);

    return 0;
}