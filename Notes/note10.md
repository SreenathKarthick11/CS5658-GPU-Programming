# AMD : GUEST Lecture

```text
Date : 5 Nov 2026
```
---

## AMD GPU Programming : From HIP to Hardware

**HIP** is the programming language used to program AMD GPUs.

HIP stands for Heterogeneous compute interface for portability

The programing the paradim is very similar to `cuda`.


## Terminalogy

The terminalogy is differenc btw the cuda and hip.


| Concept / Category | NVIDIA CUDA | AMD HIP |
| --- | --- | --- |
| **Primary Ecosystem** | NVIDIA proprietary | AMD ROCm (Open-source) |
| **Main Header File** | `<cuda_runtime.h>` | `<hip/hip_runtime.h>` |
| **API Prefix** | `cuda` (e.g., `cudaMalloc`) | `hip` (e.g., `hipMalloc`) |
| **Basic Execution Unit** | **Thread** | **Thread** |
| **Thread Group** | **Block** (Thread Block) | **WorkGroup** (Block) |
| **Grid of Thread Groups** | **Grid** | **Grid** |
| **Hardware Execution Unit / SIMD Width** | **Warp** (typically 32 threads) | **Wavefront** (typically 32 or 64 threads depending on architecture) |
| **Shared Memory** | **Shared Memory** | **LDS** (Local Data Share) |
| **Global Memory Allocation** | `cudaMalloc` | `hipMalloc` |
| **Memory Copy** | `cudaMemcpy` | `hipMemcpy` |
| **Kernel Launch Syntax** | `kernel<<<grid, block, sh_mem, stream>>>(...)` | `hipLaunchKernelGGL(kernel, dim3(grid), dim3(block), sh_mem, stream, ...)` |
| **Source-to-Source Portability Tool** | N/A | `hipify-clang` / `hipify-perl` |

## Hardware

> refer slides

Here are the filled-in notes for both sections, tailored to low-level GPU execution models (CUDA warps and AMD wavefronts):

---

## Divergent Flow Handling

>[!QUESTION] **How masks help in handle divergent workflows ?**

Because GPUs execute instructions using a **SIMT** (Single Instruction, Multiple Threads) or SIMD model where threads within a warp (NVIDIA, 32 threads) or wavefront (AMD, 32/64 threads) execute the same instruction in lockstep, conditional branching (`if-else` statements) causes **control flow divergence**.

* **The Execution Mask (Active Mask):** Hardware maintains a bitmask (one bit per thread in the warp/wavefront) called the **execution mask** or **predication mask**. This mask determines which threads are actively executing the current instruction and which are masked out (idle).
* **Serialization Mechanism:**
1. When threads diverge (e.g., half the warp takes the `if` branch and half takes the `else` branch), the hardware sets the mask bits for the `else` threads to `0` and executes the `if` path.
2. Once the `if` path completes, the hardware flips the mask—disabling the `if` threads (`0`) and enabling the `else` threads (`1`)—and executes the `else` path.
3. **Performance Impact:** The two execution paths are **serialized**, effectively cutting performance for that section proportional to the split unless all threads take the same path.


* **Re-convergence:** After both code paths finish, the mask resets, and the threads re-converge to execute subsequent instructions together.

---

## Occupancy : How many wavefront Fits


**Occupancy** is defined as the ratio of active warps (NVIDIA) or wavefronts (AMD) currently residing on a Streaming Multiprocessor (SM / Compute Unit) relative to the maximum number of active warps/wavefronts the hardware can support.

### What Determines How Many Wavefronts / Warps Fit on an SM/CU?

The number of active wavefronts/warps that can be simultaneously scheduled on a hardware unit is bounded by three primary resource constraints:

1. **Register File Usage per Thread:**
* Each SM/CU has a fixed pool of physical registers (e.g., 65,536 32-bit registers per SM in many architectures).
* *Constraint:* $\text{Total Registers Used} = \text{Threads per Block} \times \text{Registers per Thread}$. If a kernel uses a high number of registers per thread, fewer blocks can fit into the SM, lowering occupancy.


2. **Shared Memory (LDS - Local Data Share) Capacity:**
* Each SM/CU has a limited amount of fast on-chip shared memory per block (e.g., 48KB to 164KB+).
* *Constraint:* If a thread block requests a large amount of shared memory, the SM can host fewer concurrent blocks, restricting total active wavefronts.


3. **Thread Block and Hardware Limits:**
* Hardware enforces hard caps on the maximum number of concurrent threads, warps/wavefronts, and thread blocks per SM (e.g., max 1536–2048 threads per SM).
* Block allocation happens at the block granularity—if a block size does not evenly divide the maximum hardware capacity, remaining slots may go unused.



### Why Occupancy Matters

Higher occupancy helps **hide memory latency**. When a wavefront stalls waiting for data from global memory (DRAM), the hardware warp scheduler instantly switches to another ready wavefront with zero overhead, keeping compute units busy.


> refer the slides for full notes.