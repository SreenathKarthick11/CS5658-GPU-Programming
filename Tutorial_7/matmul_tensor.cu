#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;
using namespace std;

#define SIZE 64
#define TILE 16

__global__ void matmulTensorCore( const __half* A, const __half* B, float* C) {

    int tileRow = blockIdx.y;
    int tileCol = blockIdx.x;

    wmma::fragment< wmma::matrix_a, TILE, TILE, TILE, __half, wmma::row_major > a_frag;
    wmma::fragment< wmma::matrix_b, TILE, TILE, TILE, __half, wmma::col_major > b_frag;
    wmma::fragment< wmma::accumulator, TILE, TILE, TILE, float > c_frag;

    wmma::fill_fragment(c_frag, 0.0f);


    for (int k = 0; k < SIZE / TILE; k++) {

        const __half* tileA = A + tileRow * TILE * SIZE + k * TILE;
        const __half* tileB = B + k * TILE * SIZE + tileCol * TILE;

        wmma::load_matrix_sync( a_frag, tileA, SIZE);
        wmma::load_matrix_sync( b_frag, tileB, SIZE);

        wmma::mma_sync(c_frag,a_frag,b_frag,c_frag);
    }

    float* tileC = C + tileRow * TILE * SIZE + tileCol * TILE;

    wmma::store_matrix_sync( tileC, c_frag, SIZE, wmma::mem_row_major);
}


int main() {

    const int elements = SIZE * SIZE;

    size_t halfSize = elements * sizeof(__half);
    size_t floatSize = elements * sizeof(float);

    __half* h_A = new __half[elements];
    __half* h_B = new __half[elements];
    float* h_C = new float[elements];

    for (int i = 0; i < elements; i++) {
        h_A[i] = __float2half(1.0f);
        h_B[i] = __float2half(2.0f);
    }

    __half* d_A;
    __half* d_B;
    float* d_C;

    cudaMalloc(&d_A, halfSize);
    cudaMalloc(&d_B, halfSize);
    cudaMalloc(&d_C, floatSize);

    cudaMemcpy(d_A,h_A,halfSize,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,halfSize,cudaMemcpyHostToDevice);


    dim3 blockDim(32);
    dim3 gridDim(SIZE / TILE,SIZE / TILE);

    cout << "Matrix size: " << SIZE << " x " << SIZE << endl;
    cout << "Tensor Core tile size: " << TILE << " x " << TILE << endl;
    cout << "Output tiles: " << gridDim.x << " x " << gridDim.y << " = " << gridDim.x * gridDim.y << endl;

    matmulTensorCore<<<gridDim, blockDim>>>(d_A,d_B,d_C);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        cerr << "Kernel launch failed: " << cudaGetErrorString(error) << endl;
        return 1;
    }

    cudaDeviceSynchronize();

    cudaMemcpy( h_C, d_C, floatSize, cudaMemcpyDeviceToHost);

    float expected = 128.0f;
    bool correct = true;

    for (int i = 0; i < elements; i++) {
        if (fabs(h_C[i] - expected) > 0.01f) {
            correct = false;
            break;
        }
    }

    if (correct) {
        cout << "SUCCESS!" << endl;
        cout << "C[0][0] = " << h_C[0] << endl;
        cout << "Expected = " << expected << endl;
    } else {
        cout << "FAILURE!" << endl;
        cout << "C[0][0] = " << h_C[0] << endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}