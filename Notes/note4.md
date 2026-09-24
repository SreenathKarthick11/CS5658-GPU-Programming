# Notes of GPU Programming

```text
Date : 24 Sep 2026
```
---

## Tutorial

- We discussed about **Roof Line Profiling**.

---

> [!NOTE] Bhavani Cluster has 32 GPUs, and compute capability of 8.0

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

>[!NOTE] LaneId = threadId.x % warp_size

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



