#!/bin/bash
#SBATCH --job-name=matmul_nsys
#SBATCH --output=matmul_nsys_%j.out
#SBATCH --error=matmul_nsys_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00

# Profile application execution using Nsight Systems
nsys profile \
    --stats=true \
    --trace=cuda,nvtx,osrt \
    -o matmul_profile \
    ./matmul_gpu