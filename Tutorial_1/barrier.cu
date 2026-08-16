#include <iostream>
#include <cuda_runtime.h>

__device__ int barrier_count = 0;

__device__ void global_barrier(int number_of_blocks)
{
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(&barrier_count, 1);
    }

    while (atomicAdd(&barrier_count, 0) < number_of_blocks) {
      //spin
    }

    __syncthreads();
}

__global__ void barrier_test_kernel(int number_of_blocks)
{
    // Print before the barrier.
    if (threadIdx.x == 0) {
        printf("Block %d: BEFORE barrier\n", blockIdx.x);
    }

    // Global barrier.
    global_barrier(number_of_blocks);

    // Print after the barrier.
    if (threadIdx.x == 0) {
        printf("Block %d: AFTER barrier\n", blockIdx.x);
    }
}

void check_cuda_error(const char *message)
{
    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: "
                  << message << ": "
                  << cudaGetErrorString(error)
                  << std::endl;
        exit(1);
    }
}

int main()
{
    const int blocks = 4;
    const int threads_per_block = 256;

    std::cout << "Launching kernel...\n\n";

    barrier_test_kernel<<<blocks, threads_per_block>>>(
        blocks
    );

    check_cuda_error("kernel launch");

    cudaDeviceSynchronize();

    check_cuda_error("kernel execution");

    std::cout << "\nKernel completed.\n";

    return 0;
}

// ---- OUTPUT FROM GOOGLE COLLAB ----

// Launching kernel...
// Block 1: BEFORE barrier
// Block 0: BEFORE barrier
// Block 3: BEFORE barrier
// Block 2: BEFORE barrier
// Block 1: AFTER barrier
// Block 3: AFTER barrier
// Block 2: AFTER barrier
// Block 0: AFTER barrier
// Kernel completed.
