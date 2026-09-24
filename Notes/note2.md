# Notes of GPU Programming

```text
Date : 15 Sep 2026
```

## QUESTIONS

- What are abstract for all algorithm when programing in GPU ?

- Can you data transfer with only using `cudaMemcpy` ? like without using  `cudamemcpyAsync`.

- When is `cudaMemcpyFromSymbol` and to `cudaMemcpyToSymbol` ?

- Is it possible to have a kernel with total no blocks > no of threads per block, is it to possible to have a barrier here?

- How to improve cache locality ?

- Explain tileing ?

- What is the best {i,j,k} order for Matmul definition of loop , in terms of cache ?

- What are Fixpoint computation ?

- What does `syncthreads` do ?

- What are the `thread fence` , and how is it used?

- How to reduce bank conflicts in Shared memory usage ?

- What is PTX ?

- Types of scanning ?

- Implementation of barrier for the entire kernel.

