#!/bin/bash

#SBATCH -J tensor_matmul
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 00:05:00

./matmul_tensor