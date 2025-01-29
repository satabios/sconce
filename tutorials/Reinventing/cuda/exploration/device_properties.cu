#include <cuda_runtime.h>
#include <iostream>

int main() {
    // Create a cudaDeviceProp structure to hold the device properties
    cudaDeviceProp devProp;

    // Get device properties for device 0
    cudaError_t err = cudaGetDeviceProperties(&devProp, 0);
    if (err != cudaSuccess) {
        // Handle the error
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Successfully retrieved device properties, print relevant information
    std::cout << "Device Name: " << devProp.name << std::endl;
    std::cout << "Total Global Memory: " << devProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
    std::cout << "Total Shared Memory per Block: " << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "Max Threads per Block: " << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "Warp Size: " << devProp.warpSize << std::endl;
    std::cout << "Number of Multiprocessors (SM): " << devProp.multiProcessorCount << std::endl;
    std::cout << "Max Threads per Multiprocessor: " << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Number of Registers/Block : " << devProp.regsPerBlock << std::endl;

    // Approximation: Max blocks per multiprocessor can be estimated
    std::cout << "Estimated Max Blocks per Multiprocessor: "
              << devProp.maxThreadsPerMultiProcessor / devProp.maxThreadsPerBlock << std::endl;

    std::cout<<"Max Threads Along X:"<<devProp.maxThreadsDim[0]<<std::endl;
    std::cout<<"Max Threads Along Y:"<<devProp.maxThreadsDim[1]<<std::endl;
    std::cout<<"Max Threads Along Z:"<<devProp.maxThreadsDim[2]<<std::endl;

    std::cout<<"Max Blocks Along X:"<<devProp.maxGridSize[0]<<std::endl;
    std::cout<<"Max Blocks Along Y:"<<devProp.maxGridSize[1]<<std::endl;
    std::cout<<"Max Blocks Along Z:"<<devProp.maxGridSize[2]<<std::endl;

    return 0;
}


// ----------Output----------
// Device Name: NVIDIA GeForce RTX 4090
// Total Global Memory: 23.6429 GB
// Total Shared Memory per Block: 48 KB
// Max Threads per Block: 1024
// Warp Size: 32
// Number of Multiprocessors (SM): 128
// Max Threads per Multiprocessor: 1536
// Number of Registers/Block : 65536
// Estimated Max Blocks per Multiprocessor: 1
// Max Threads Along X:1024
// Max Threads Along Y:1024
// Max Threads Along Z:64
// Max Blocks Along X:2147483647
// Max Blocks Along Y:65535
// Max Blocks Along Z:65535