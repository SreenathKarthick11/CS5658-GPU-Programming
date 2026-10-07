#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <cuda_file.cu> <sbatch_file.sh>"
    exit 1
fi

CU_FILE="$1"
SBATCH_FILE="$2"
[ -f "$CU_FILE" ] && [ -f "$SBATCH_FILE" ] || { echo "Error: file not found."; exit 1; }

REMOTE="s112301042-unnikrishnan@192.168.1.133"
REMOTE_DIR="~/gpu_projects/$(basename "$PWD")"
SBATCH_NAME=$(basename "$SBATCH_FILE")
CU_NAME=$(basename "$CU_FILE")
BIN_NAME="${CU_NAME%.cu}"

echo "[1/4] Copying files..."
ssh "$REMOTE" "mkdir -p $REMOTE_DIR" && scp "$CU_FILE" "$SBATCH_FILE" "$REMOTE:$REMOTE_DIR/" || exit 1

echo "[2/4] Compiling on login node..."
ssh "$REMOTE" "bash -l -c 'cd $REMOTE_DIR && module unload gnu12 && module load cuda/11.3 && nvcc $CU_NAME -o $BIN_NAME'" || { echo "Compile failed"; exit 1; }

echo "[3/4] Submitting job..."
JOB_ID=$(ssh "$REMOTE" "cd $REMOTE_DIR && sbatch --parsable $SBATCH_NAME")
[ -z "$JOB_ID" ] && { echo "Submit failed"; exit 1; }
echo "Job ID: $JOB_ID"

echo "[4/4] Waiting..."
while [ -n "$(ssh "$REMOTE" "squeue -j $JOB_ID -h" 2>/dev/null)" ]; do sleep 2; done

echo "=== STDOUT ==="
ssh "$REMOTE" "cat $REMOTE_DIR/job.$JOB_ID.out 2>/dev/null"
echo "=== STDERR ==="
ssh "$REMOTE" "cat $REMOTE_DIR/job.$JOB_ID.err 2>/dev/null"