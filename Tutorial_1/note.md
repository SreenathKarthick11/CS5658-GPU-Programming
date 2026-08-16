# Tutorial 1

Basic CUDA programming: querying GPU device properties and implementing thread synchronization using barriers.

## Questions

### 1) Write a program that print properties of a CUDA device (use google colab for geeting GPU). Edit program `cuda.c` uploaded in Moodle and print more properties.

**Solution:** [`properties.cu`](properties.cu)


### 2) Write a program that implements barrier for all the threads of a CUDA kernel using `__synchthreads` and  atomic operations.

**Solution:** [`barrier.cu`](barrier.cu)
