#!/bin/bash
#SBATCH --job-name=olmo2_1b_sft
#SBATCH --time=120:00:00
#SBATCH --partition=a3
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --cpus-per-task=32

######################
# ENV Config
######################
CONDA_ROOT_DIR=$(conda info --base)
source $CONDA_ROOT_DIR/etc/profile.d/conda.sh
conda activate olmo
PY_BIN=${CONDA_ROOT_DIR}/${SUFFIX}/bin

######################

MIN=1   # minimum seconds
MAX=100  # maximum seconds
SLEEP_TIME=$((MIN + RANDOM % (MAX - MIN + 1)))
sleep $SLEEP_TIME
MYPORT=8775
while ss -tulnp | grep -q ":$MYPORT"; do
    echo "Port $PORT is occupied. Trying next..."
    ((MYPORT++))
done

DATA_PREFIX="./olmo_data"
SAVE_FOLDER="./ckpts/"
NUM_GPU=4
BATCH_SIZE=4

# parse options
while getopts "n:b:d:s:" opt; do
  case $opt in
    n) NUM_GPU=$OPTARG ;;    # -g <num_gpus>
    b) BATCH_SIZE=$OPTARG ;; # -b <batch_size>
    d) DATA_PREFIX=$OPTARG ;; # -d <data_prefix>
    s) SAVE_FOLDER=$OPTARG ;; # -d <data_prefix>
    \?) echo "Usage: $0 [-g num_gpu] [-b batch_size] [-d data_prefix]" >&2
        exit 1 ;;
  esac
done

echo "DATA_PREFIX=$DATA_PREFIX"
echo "NUM_GPU=$NUM_GPU"
echo "BATCH_SIZE=$BATCH_SIZE"


if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    # RoPE (baseline)
    torchrun --nproc_per_node=$NUM_GPU --master_port=$MYPORT scripts/train.py configs/official-0425/OLMo2-1B-stage2-seed42-NONE.yaml --device_train_microbatch_size=${BATCH_SIZE} --save_overwrite=true --data.num_workers=24 --data_prefix=$DATA_PREFIX
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    # RePo (ours)
    torchrun --nproc_per_node=$NUM_GPU --master_port=$MYPORT scripts/train.py configs/official-0425/OLMo2-1B-stage2-seed42-SEXMH-L5.yaml --device_train_microbatch_size=${BATCH_SIZE}  --save_overwrite=true --data.num_workers=24 --data_prefix=$DATA_PREFIX
else
    echo "unknown job id ${SLURM_ARRAY_TASK_ID}"
fi
