#!/bin/bash

usage() {
    echo "Usage: $0 [-d] [-k] <cuda_file.cu> <sbatch_file.sh>"
    echo "  -d  delete the remote project folder after the run"
    echo "  -k  keep the job .out/.err files (default: deleted after printing)"
    exit 1
}

DELETE_DIR=0
KEEP_LOGS=0
while getopts "dk" opt; do
    case "$opt" in
        d) DELETE_DIR=1 ;;
        k) KEEP_LOGS=1 ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

[ "$#" -ne 2 ] && usage

CU_FILE="$1"
SBATCH_FILE="$2"
[ -f "$CU_FILE" ] && [ -f "$SBATCH_FILE" ] || { echo "Error: file not found."; exit 1; }

REMOTE="s112301042-unnikrishnan@192.168.1.133"
FOLDER_NAME=$(basename "$PWD")
REMOTE_DIR="~/gpu_projects/${FOLDER_NAME}"
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

# Cleanup
if [ "$DELETE_DIR" -eq 1 ] && [ -n "$FOLDER_NAME" ] && [ "$FOLDER_NAME" != "/" ]; then
    echo "Deleting remote folder ${REMOTE_DIR}..."
    ssh "$REMOTE" "rm -rf $REMOTE_DIR"
elif [ "$KEEP_LOGS" -eq 0 ]; then
    echo "Deleting job logs..."
    ssh "$REMOTE" "rm -f $REMOTE_DIR/job.$JOB_ID.out $REMOTE_DIR/job.$JOB_ID.err"
fi