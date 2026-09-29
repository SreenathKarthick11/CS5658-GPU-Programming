# Notes of GPU Programming

```text
Date : 29 Sep 2026
```
---

## Implementing of Matrix Multiplication with Tensor Core

* **Standard CUDA Cores / CPUs**: Execute these MAC (single Multiply-Accumulate )operations **scalar-by-scalar** or along 1D vectors, computing one or a few FMA operations like $C[0,0] = C[0,0] + A[0,1] \cdot B[1,0]$ per instruction pipeline.
* **Tensor Cores**: Execute **entire matrix-level MACs** in hardware per clock cycle. Instead of doing just $C[0,0] = C[0,0] + A[0,1] \cdot B[1,0]$ individually, a Tensor Core executes a hardware instruction that computes a full matrix equation:

$$D = A \times B + C$$

 >[!NOTE]
 A standard NVIDIA Tensor Core computes a $16 \times 16$ matrix multiplied by a $16 \times 16$ matrix and adds a $16 \times 16$ accumulator matrix in a single hardware cycle.


---

## The Warp Cuda functions

`wmma` functions are warp-synchronous primitives, meaning all 32 threads in a CUDA warp must call them together.

| CUDA Function | Primary Purpose | Operation / Data Flow | Warp Behavior |
| --- | --- | --- | --- |
| `load_matrix_sync` | Loads a matrix tile from memory into warp registers. | Shared / Global Memory $\rightarrow$ Fragment Registers | Synchronizes 32 threads in a warp to cooperatively read memory into registers. |
| `store_matrix_sync` | Writes a matrix tile from warp registers back to memory. | Fragment Registers $\rightarrow$ Shared / Global Memory | Synchronizes 32 threads in a warp to cooperatively write registers out to memory. |
| `fill_fragment` | Initializes a matrix fragment with a constant scalar value. | Constant Value $\rightarrow$ Fragment Registers | All 32 threads set their held register values (commonly used to set accumulator $C$ to `0`). |
| `mma_sync` | Executes the hardware matrix multiplication on Tensor Cores. | Registers: $D = A \times B + C$ | Synchronizes the warp to execute a hardware-level matrix multiply-accumulate step. |


---

> QUESTION: How many tensor cores are present per SMs