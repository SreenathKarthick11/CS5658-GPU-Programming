#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define N 1024

// CUDA Kernel for Vector Addition
__global__ void vecAddKernel(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    size_t size = N * sizeof(float);


    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);


    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }


    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);


    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);


    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Launching Kernel: " << blocksPerGrid << " blocks, "
              << threadsPerBlock << " threads/block." << std::endl;

    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();


    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);


    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_C[i] - 3.0f) > 1e-5) {
            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "SUCCESS: Vector addition correct! Example output C[0] = " << h_C[0] << std::endl;
    } else {
        std::cout << "FAILURE: Incorrect results generated." << std::endl;
    }

  
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}