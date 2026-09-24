# Notes of GPU Programming

```text
Date : 17 Sep 2026
```

---

## GPU GeForce- 2080Ti
- 68 SM
- 64 SP per SM

> [!IMPORTANT] Cool Reference
> This comprehensive notes for GPU : [Model GPU Glossary](https://modal.com/gpu-glossary/readme)

Example : kernel code for matrix multiplication
```c
__global__ void mm(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

---

## Memory Banks

**GPU memory banks** are equally-sized, independent modules of shared memory (or register files) designed to allow parallel data access.  By organizing memory into these banks, a GPU can service multiple memory requests simultaneously, significantly increasing throughput compared to a single, monolithic memory block.

Key characteristics of GPU memory banks:

- __Parallel Access__ : If threads in a warp access distinct banks, all requests are serviced in a single clock cycle.
- __Bank Conflicts__ : If multiple threads in the same warp request different data from the same bank, the hardware must serialize these accesses, reducing performance by a factor equal to the number of conflicting threads.
- __Address Mapping__ : In modern NVIDIA GPUs (Compute Capability 3.x and newer), there are 32 banks, each 4 bytes wide.  Addresses map to banks using the formula `bank_id = (byte_address / 4) % 32`, meaning consecutive 4-byte words map to consecutive banks.
- __Broadcasting__ : If all threads in a warp access the same address within a bank, the hardware retrieves the data once and broadcasts it to all threads, avoiding serialization penalties.

>[!Warning] Understand Bank Conflict
> Refer : [Link](https://modal.com/gpu-glossary/perf/bank-conflict)

---

## Profiling

**GPU profiling** is the process of measuring and analyzing the performance characteristics of GPU applications to identify bottlenecks, optimize resource utilization, and improve system efficiency.

It involves tracking where the GPU spends time and resources, such as during rendering, shading, or memory access, to pinpoint inefficiencies like poorly optimized shaders, excessive draw calls, or underutilized memory bandwidth.

`cudaEvent_t` datatype is used.

```c++
\\ Example Usage of cudaEvent for profiling kernel launch.
cudaEvent_t start,stop;
cudaEventCreate (&start);
cudaEventCreate (&stop);

cudaEventRecord(&start);

kernel launch <<<>>>
cudaDeviceSycronize();

cudaEventRecord(&stop);
float kernel_time;
cudaEventElapsedTime(&kernel_time,start,stop);
```

By using profiling, it had come to our attention that copy taking the most time in Device Host.

There is the existance of `pinned memory`. which prevents the pages of that memory address to be swapped out.

To pin a memory we use `cudaMallocHost()` on that address.

The `cudaMemcpy` takes two steps to transfer the memory.
- Copy from host memory to host pinned memory
- Copy form host pinned memory to device memory

So if we use directly allocate to the pinned memory the time taken for copy reduces , hence improving performance.

But allocating everything to pinned memory can hamper performace , and may lead to memory thrashing in RAM.


> When communitcation is greater that computation time in then ,it better to use CPU.

---
