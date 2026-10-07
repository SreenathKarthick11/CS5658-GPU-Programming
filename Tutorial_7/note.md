# Tutorial 7

Learning to Use the Bhavani Cluster.

## Task 1

Implement **Vector Addition** using CUDA and schedule the job on the Bhavani Cluster.

**Files:**

* `vec_add.cu` : CUDA implementation
* `submit_vecadd.sh` : Slurm batch script
* `run_job.sh` : local helper script

### Manual Run

Compile the CUDA program on the Bhavani login node and submit it using Slurm.

```bash
module unload gnu12
module load cuda/11.3
nvcc vec_add.cu -o vec_add

sbatch submit_vecadd.sh
```

### Using `run_job.sh`

`run_job.sh` copies the CUDA and Slurm files to the Bhavani cluster, compiles the CUDA program on the login node, submits the Slurm job, waits for it to finish, and prints the output.

```bash
./run_job.sh vec_add.cu submit_vecadd.sh
```

Options:

```bash
./run_job.sh -k vec_add.cu submit_vecadd.sh
./run_job.sh -d vec_add.cu submit_vecadd.sh
```

* `-k` — keep the `.out` and `.err` files.
* `-d` — delete the remote project folder after execution.

---

## Task 2

Implement **Matrix Multiplication** of 64x64 with **Tensor Cores** and tile size of 16x16.

**Solution:** [matmul_with_tensor_cores](matmul_tensor.cu)

**Files:**

* `matmul_tensor.cu` : Tensor Core matrix multiplication using WMMA
* `submit_matmul_tensor.sh` : Slurm batch script

### Manual Run

Copy the files to Cluster

```bash
module unload gnu12
module load cuda/11.3
nvcc matmul_tensor.cu -o matmul_tensor

sbatch submit_matmul_tensor.sh
```

### Using `run_job.sh`

Tensor Core compilation requires the CUDA architecture to be specified using the `-a` option.

```bash
./run_job.sh -a sm_70 matmul_tensor.cu submit_matmul_tensor.sh
```



