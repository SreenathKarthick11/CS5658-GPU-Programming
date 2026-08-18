# Tutorial 1

Basic CUDA programming: querying GPU device properties and implementing thread synchronization using barriers.

## Questions

1) Write a program that print properties of a CUDA device (use google colab for geeting GPU). Edit program `cuda.c` uploaded in Moodle and print more properties.

**Solution:** [properties.cu](properties.cu)


2) Write a program that implements barrier for all the threads of a CUDA kernel using `__synchthreads` and  atomic operations.

**Solution:** [barrier.cu](barrier.cu)

3) What happens when you put no of block greater than no of SMs in `barrier.cu` ?

**Solution:** The program will not work as expected because the barrier implementation relies on all threads reaching the barrier before proceeding. If there are more blocks than SMs, some blocks may not be able to execute concurrently, leading to deadlock or incorrect synchronization behavior.


