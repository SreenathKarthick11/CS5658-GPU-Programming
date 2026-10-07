# Tutorial 7 :

Learning to Use the Bhavani Cluster

## Task 1

Make a simple vector addition program and schedule the job in Bhavani Cluster to run the program.

Files: `vec_add.cu` (CUDA code), `submit_vecadd.sh` (Slurm batch file), `run_job.sh` (local helper script)

### Method 1: Manual (on the cluster)

```bash
cd ~/gpu_projects/Tutorial_7

# 1. Compile on the login node
module unload gnu12
module load cuda/11.3
nvcc vec_add.cu -o vec_add

# 2. Submit the job
sbatch submit_vecadd.sh

# 3. Check the result
squeue -u $USER          # job status
cat job.<JOBID>.out      # output
cat job.<JOBID>.err      # errors
```

### Method 2: Using `run_job.sh` (from the local machine)

Run it from the folder with the `.cu` and sbatch files. It copies the files, compiles, submits, waits and prints the output.

```bash
./run_job.sh vec_add.cu submit_vecadd.sh       # logs deleted, folder stays
./run_job.sh -k vec_add.cu submit_vecadd.sh    # keep the logs
./run_job.sh -d vec_add.cu submit_vecadd.sh    # delete the remote folder
```

Flags go before the filenames.

### Issues faced

- **`features.h: No such file`**: compiling on a compute node, which lacks dev headers. Compile on the login node instead.
- **`unsupported GNU version` / `std::pair` errors**: CUDA 11.3 does not work with the default `gnu12`. Fix: `module unload gnu12` so the system GCC 8.5 is used.

## Task 2

Implement a Matrix Multiplication code with tensors , and schedule the job to run in Bhavani cluster


