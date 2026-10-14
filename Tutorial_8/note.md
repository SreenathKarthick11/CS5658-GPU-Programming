# Tutorial 8

Profiling CUDA Applications on the Bhavani Cluster using `nsys` (NVIDIA Nsight Systems).

## Task

Profile matrix multiplication of two $1024 \times 1024$ integer matrices on GPU standard CUDA cores (without using Tensor Cores) and inspect performance bottlenecks.

**Files:**

* [matmul.cu](matmul.cu) : Cuda code
* [submit_matmul.sh](submit_matmul.sh) : Slurm batch script

---


## Run

Copy the files to the Bhavani cluster and run the profiling manually:

```bash
scp matmul.cu submit_matmul.sh s112301042-unnikrishnan@192.168.1.133:~/test

```

SSH into the cluster node, compile, and submit the Slurm job:

```bash

nvcc matmul.cu -o matmul_gpu

sbatch submit_matmul.sh

```
---


## Profiling Results (`nsys` Output)

To view the terminal output for the sbatch : [Click Here](profile.txt)

To view the interactive GUI trace on a local machine, retrieve the report using `scp`:

```bash
scp s112301042-unnikrishnan@192.168.1.133:~/test/matmul_profile.nsys-rep .

scp s112301042-unnikrishnan@192.168.1.133:~/test/matmul_profile.sqlite .

```

Open the `.nsys-rep` file in **NVIDIA Nsight Systems GUI** locally to inspect timeline traces and GPU kernel metrics.