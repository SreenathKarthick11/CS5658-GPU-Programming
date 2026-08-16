#include <iostream>
#include <cuda_runtime.h>

int getCoresPerSM(int major, int minor) {
    switch (major) {
        case 2: // Fermi
            return (minor == 1) ? 48 : 32;

        case 3: // Kepler
            return 192;

        case 5: // Maxwell
            return 128;

        case 6: // Pascal
            if (minor == 1 || minor == 2)
                return 128;
            if (minor == 0)
                return 64;
            return 128;

        case 7: // Volta / Turing
            return 64;

        case 8: // Ampere / Ada
            if (minor == 0)
                return 64;
            if (minor == 6 || minor == 9)
                return 128;
            return 64;

        case 9: // Hopper / Blackwell
            return 128;

        default:
            return 128;
    }
}

int main() {
    int deviceCount = 0;

    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s).\n\n";

    for (int i = 0; i < deviceCount; ++i) {

        cudaDeviceProp prop;

        error = cudaGetDeviceProperties(&prop, i);

        if (error != cudaSuccess) {
            std::cerr << "Error getting properties for device " << i << ": " << cudaGetErrorString(error) << std::endl;
            continue;
        }

        int coresPerSM = getCoresPerSM(prop.major, prop.minor);


        std::cout << "Device " << i << ": "  << prop.name << "\n";
        std::cout << "========================================\n";

        // Basic information
        std::cout << "Compute Capability:             " << prop.major << "." << prop.minor << "\n";
        std::cout << "Total Global Memory:            " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
        std::cout << "Streaming Multiprocessors:      " << prop.multiProcessorCount << "\n";
        std::cout << "Cores Per SM:                   " << coresPerSM << "\n";
        std::cout << "Estimated Total CUDA Cores:     " << prop.multiProcessorCount * coresPerSM << "\n";

        // Thread information
        std::cout << "\n--- Thread Configuration ---\n";
        std::cout << "Warp Size:                      " << prop.warpSize << "\n";
        std::cout << "Max Threads Per Block:          " << prop.maxThreadsPerBlock << "\n";
        std::cout << "Max Threads Per SM:             " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "Max Threads Dimension:          "<< prop.maxThreadsDim[0] << " x "<< prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << "\n";
        std::cout << "Max Grid Dimension:             "<< prop.maxGridSize[0] << " x "<< prop.maxGridSize[1] << " x "<< prop.maxGridSize[2] << "\n";

        // Memory information
        std::cout << "\n--- Memory ---\n";
        std::cout << "Shared Memory Per Block:        " << prop.sharedMemPerBlock / 1024 << " KB\n";
        std::cout << "Shared Memory Per SM:           " << prop.sharedMemPerMultiprocessor / 1024 << " KB\n";
        std::cout << "Constant Memory:                " << prop.totalConstMem / 1024 << " KB\n";
        std::cout << "L2 Cache Size:                  " << prop.l2CacheSize / 1024 << " KB\n";
        std::cout << "Memory Bus Width:               " << prop.memoryBusWidth << " bits\n";

        // Clock information
        std::cout << "\n--- Clock Information ---\n";
        std::cout << "GPU Clock Rate:                 " << prop.clockRate / 1000 << " MHz\n";
        std::cout << "Memory Clock Rate:              " << prop.memoryClockRate / 1000 << " MHz\n";

        // Hardware resources
        std::cout << "\n--- Hardware Resources ---\n";
        std::cout << "Registers Per Block:            " << prop.regsPerBlock << "\n";
        std::cout << "Registers Per SM:               " << prop.regsPerMultiprocessor << "\n";
        std::cout << "Concurrent Kernels:             " << (prop.concurrentKernels ? "Yes" : "No") << "\n";
        std::cout << std::endl;
    }

    return 0;
}


// ---- OUTPUT FROM GOOGLE COLAB ----

// Found 1 CUDA device(s).

// Device 0: Tesla T4
// ========================================
// Compute Capability:             7.5
// Total Global Memory:            14912 MB
// Streaming Multiprocessors:      40
// Cores Per SM:                   64
// Estimated Total CUDA Cores:     2560

// --- Thread Configuration ---
// Warp Size:                      32
// Max Threads Per Block:          1024
// Max Threads Per SM:             1024
// Max Threads Dimension:          1024 x 1024 x 64
// Max Grid Dimension:             2147483647 x 65535 x 65535

// --- Memory ---
// Shared Memory Per Block:        48 KB
// Shared Memory Per SM:           64 KB
// Constant Memory:                64 KB
// L2 Cache Size:                  4096 KB
// Memory Bus Width:               256 bits

// --- Clock Information ---
// GPU Clock Rate:                 1590 MHz
// Memory Clock Rate:              5001 MHz

// --- Hardware Resources ---
// Registers Per Block:            65536
// Registers Per SM:               65536
// Concurrent Kernels:             Yes

