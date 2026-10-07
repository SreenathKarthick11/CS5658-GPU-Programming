
#!/bin/bash

usage() {
    echo "Usage: $0 [-d] [-k] [-a ARCH] <cuda_file.cu> <sbatch_file.sh>"
    echo "  -d        delete the remote project folder after the run"
    echo "  -k        keep the job .out/.err files"
    echo "  -a ARCH   CUDA architecture, e.g. sm_70, sm_75, sm_80"
    exit 1
}

DELETE_DIR=0
KEEP_LOGS=0
ARCH=""

while getopts "dka:" opt; do
    case "$opt" in
        d)
            DELETE_DIR=1
            ;;
        k)
            KEEP_LOGS=1
            ;;
        a)
            ARCH="$OPTARG"
            ;;
        *)
            usage
            ;;
    esac
done

shift $((OPTIND - 1))

[ "$#" -ne 2 ] && usage

CU_FILE="$1"
SBATCH_FILE="$2"

if [ ! -f "$CU_FILE" ] || [ ! -f "$SBATCH_FILE" ]; then
    echo "Error: file not found."
    exit 1
fi

REMOTE="s112301042-unnikrishnan@192.168.1.133"

FOLDER_NAME=$(basename "$PWD")
REMOTE_DIR="~/gpu_projects/${FOLDER_NAME}"

SBATCH_NAME=$(basename "$SBATCH_FILE")
CU_NAME=$(basename "$CU_FILE")
BIN_NAME="${CU_NAME%.cu}"


# --------------------------------------------------
# 1. Copy files
# --------------------------------------------------

echo "[1/4] Copying files..."

ssh "$REMOTE" "mkdir -p $REMOTE_DIR"

if [ $? -ne 0 ]; then
    echo "Failed to create remote directory."
    exit 1
fi

scp "$CU_FILE" "$SBATCH_FILE" "$REMOTE:$REMOTE_DIR/"

if [ $? -ne 0 ]; then
    echo "Failed to copy files."
    exit 1
fi


# --------------------------------------------------
# 2. Compile
# --------------------------------------------------

echo "[2/4] Compiling on login node..."

if [ -n "$ARCH" ]; then

    echo "Using CUDA architecture: $ARCH"

    ssh "$REMOTE" "
        cd $REMOTE_DIR &&
        module unload gnu12 &&
        module load cuda/11.3 &&
        nvcc -arch=$ARCH $CU_NAME -o $BIN_NAME
    "

else

    echo "Using default CUDA architecture"

    ssh "$REMOTE" "
        cd $REMOTE_DIR &&
        module unload gnu12 &&
        module load cuda/11.3 &&
        nvcc $CU_NAME -o $BIN_NAME
    "

fi

if [ $? -ne 0 ]; then
    echo "Compile failed"
    exit 1
fi


# --------------------------------------------------
# 3. Submit job
# --------------------------------------------------

echo "[3/4] Submitting job..."

JOB_ID=$(ssh "$REMOTE" "
    cd $REMOTE_DIR &&
    sbatch --parsable $SBATCH_NAME
")

if [ -z "$JOB_ID" ]; then
    echo "Submit failed"
    exit 1
fi

echo "Job ID: $JOB_ID"


# --------------------------------------------------
# 4. Wait for job
# --------------------------------------------------

echo "[4/4] Waiting..."

while true; do

    JOB_STATUS=$(ssh "$REMOTE" "squeue -j $JOB_ID -h" 2>/dev/null)

    if [ -z "$JOB_STATUS" ]; then
        break
    fi

    sleep 2

done


echo "=== STDOUT ==="

ssh "$REMOTE" "
    cat $REMOTE_DIR/job.$JOB_ID.out 2>/dev/null
"


echo "=== STDERR ==="

ssh "$REMOTE" "
    cat $REMOTE_DIR/job.$JOB_ID.err 2>/dev/null
"


# --------------------------------------------------
# Cleanup
# --------------------------------------------------

if [ "$DELETE_DIR" -eq 1 ]; then

    echo "Deleting remote folder ${REMOTE_DIR}..."

    ssh "$REMOTE" "rm -rf $REMOTE_DIR"

elif [ "$KEEP_LOGS" -eq 0 ]; then

    echo "Deleting job logs..."

    ssh "$REMOTE" "
        rm -f $REMOTE_DIR/job.$JOB_ID.out
        rm -f $REMOTE_DIR/job.$JOB_ID.err
    "

fi

