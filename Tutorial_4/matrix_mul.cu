#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 500
#define TILE_SIZE 32


__global__ void matrixMul(int *A, int *B, int *C) {

    __shared__ int As[TILE_SIZE][TILE_SIZE];
    __shared__ int Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    int sum = 0;

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < numTiles; tile++) {

        // Load A tile
        int aCol = tile * TILE_SIZE + tx;

        if (row < N && aCol < N)
            As[ty][tx] = A[row * N + aCol];
        else
            As[ty][tx] = 0;

        // Load B tile
        int bRow = tile * TILE_SIZE + ty;

        if (bRow < N && col < N)
            Bs[ty][tx] = B[bRow * N + col];
        else
            Bs[ty][tx] = 0;

        __syncthreads();

        // Multiply tiles
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (row < N && col < N) C[row * N + col] = sum;
}


int main()
{
    size_t size = N * N * sizeof(int);

    int *A = (int *)malloc(size);
    int *B = (int *)malloc(size);
    int *C = (int *)malloc(size);
    int *C_cpu = (int *)malloc(size);

    // Generate random integers
    srand(42);

    for (int i = 0; i < N * N; i++)
    {
        A[i] = rand() % 11;
        B[i] = rand() % 11;
    }


    // CPU matrix multiplication
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            C_cpu[i * N + j] = 0;

            for (int k = 0; k < N; k++)
                C_cpu[i * N + j] +=
                    A[i * N + k] * B[k * N + j];
        }
    }

    // Device matrices
    int *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy A and B to GPU
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // CUDA configuration
    dim3 block(TILE_SIZE, TILE_SIZE);

    dim3 grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (N + TILE_SIZE - 1) / TILE_SIZE
    );

    printf("Matrix size : %d x %d\n", N, N);
    printf("Tile size   : %d x %d\n", TILE_SIZE, TILE_SIZE);
    printf("Block size  : %d threads\n", TILE_SIZE * TILE_SIZE);
    printf("Grid size   : %d x %d\n", grid.x, grid.y);


    // GPU multiplication
    matrixMul<<<grid, block>>>(d_A, d_B, d_C);

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify
    bool correct = true;

    for (int i = 0; i < N * N; i++)
    {
        if (C[i] != C_cpu[i])
        {
            correct = false;
            printf("Mismatch at index %d\n", i);
            printf("GPU = %d, CPU = %d\n", C[i], C_cpu[i]);
            break;
        }
    }

    if (correct)
        printf("\nResults match!\n");
    else
        printf("\nResults do not match!\n");

    // Print first 5x5 elements
    printf("\nFirst 5x5 elements of GPU result:\n");

    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
            printf("%6d ", C[i * N + j]);

        printf("\n");
    }

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);
    free(C_cpu);

    return 0;
}