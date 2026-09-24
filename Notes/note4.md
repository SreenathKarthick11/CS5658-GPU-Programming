# Notes of GPU Programming

```text
Date : 24 Sep 2026
```
---

## Guest Lecture

- We discussed about **Roof Line Analysis**.

---

> [!NOTE]
Bhavani Cluster has 32 GPUs, and compute capability of 8.0

> [!QUESTION]
>```cuda
>void main(){
>    K<<<x,y>>>(); // lauch kernel
>    // let say we don't have `cudaDeviceSyncronize`.
>}
>```
> What is the expected output ?

As the host doesn't waits for device , it just finishs executing,hence most of the time we don't get any output.

---

## Functions

### Warp Shuffle Functions

>[!NOTE]
>LaneId = threadId.x % warp_size

| Function | What it does | Typical use |
|----------|-------------|-------------|
| `__shfl_sync(mask, var, srcLane)` | Every thread gets `var` from lane `srcLane` | **Broadcast** one lane's value to all |
| `__shfl_up_sync(mask, var, delta)` | Thread `i` gets `var` from lane `i - delta` (unchanged if `i < delta`) | **Prefix sum / scan** |
| `__shfl_down_sync(mask, var, delta)` | Thread `i` gets `var` from lane `i + delta` (unchanged if out of range) | **Reduction** (tree) |
| `__shfl_xor_sync(mask, var, laneMask)` | Thread `i` gets `var` from lane `i XOR laneMask` | **Butterfly all-reduce** |

> [!QUESTION] What is the forth argument to warp shuffle function ?
 The 4th argument is width (default 32).  It subdivides the warp into independent sub-groups of that size, and the shuffle only exchanges data within each sub-group.

### Other Bit Operations

| Function | What it does | Example |
|----------|-------------|---------|
| `__popc(unsigned x)` | Count set bits (popcount) | `__popc(0b1011)` =  `3` |
| `__ffs(int x)` | Position of **least-significant** set bit (1-indexed; 0 if `x==0`) | `__ffs(0b1010)` = `2` |

---

## Half Data type in Cuda

- **Format:** 1 sign + 5 exponent + 10 mantissa (IEEE 754 binary16)
- **Size:** 2 bytes | **Range:** ~6.1e-5 to ~65504
- **Header:** `#include <cuda_fp16.h>`
- **Native arithmetic:** CC ≥ 5.3 (Pascal+)

### Types
| Type | Notes |
|------|-------|
| `__half` | 16-bit float (struct over `unsigned short`) |
| `__half2` | 2× `__half` packed -> 2-way SIMD, ~2× throughput |

### Conversions
```cpp
__half  h  = __float2half(f);
float   f  = __half2float(h);
__half2 h2 = __floats2half2_rn(a, b);
float2  f2 = __half22float2(h2);
```


