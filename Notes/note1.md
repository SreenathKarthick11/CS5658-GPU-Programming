# Notes of GPU Programming

```text
Date : 20 Aug 2026
```


## Some points noted during class

The following is refers the `doubleptr.cu` code.
- understand the address space of device and host.
- when can we allocate them, when can we copy them.
- read the comments

> **TRY**:
> use a struct and allocate it in host.


## Some questions and points

> **VERIFY:** These points

- `shared` Out of 64KB of L1 cache, only 48KB can be used for shared memory.
- `const` key word is used to allocate in device main memory.



> **REFER:**
> The following code base can be refered for sample programs.
>[Cuda Sample](https://github.com/zchee/cuda-sample)


----

## Some notes

> [!NOTE] **TO LEARN**
>- warp divergence
>- warp voteing
>
>understanding `__shared__` memory (like what happens when it declared globally)


Today lets look at loops.


We want to parallelize the following loop
```c++
for(int i=1;i<N;i++){
    arr[i]=arr[i-1]+1;
}
```

> [!Warning]
> We can't do this normally as we will be generating n thread, but there presence a dependencies.


can we parallelised the by modifiy the loop.

## Difference btw Intra and Inter loop dependency

**Intra-loop dependencies** (specifically loop-carried dependencies) occur when an iteration of a loop depends on the result of a previous iteration within the same loop.  GPUs typically cannot parallelize these loops because concurrent threads would attempt to read and write shared variables simultaneously, leading to undefined behavior or race conditions.  To optimize such code, developers must remove dependencies by using atomic operations (e.g., atomicAdd) or rewriting logic to avoid reading results from prior iterations.

**Inter-loop dependencies** exist when the execution of one loop depends on the output of a preceding loop.  In GPU programming, this requires sequential execution of the dependent kernels, as the second loop cannot begin until the first has completed. This serialization prevents the GPU from overlapping the execution of these stages, potentially creating performance bottlenecks if the loops are large.
