# Tutorial 4

Implement matrix multiplication on the GPU using CUDA shared-memory tiling.

**Solution:** [matrix_mul.cu](matrix_mul.cu)

> [!Note]
>
> A **tile size of 32 × 32** is used, giving **1024 threads per block**.
>
> Since the matrix size is 500 × 500, which is not divisible by 16, **boundary checks** are required while loading elements into shared memory.
>
> `__syncthreads()` is used to synchronize all threads in a block after loading the tiles into shared memory.
>
> The GPU result is **verified against the CPU matrix multiplication result** to ensure correctness.
