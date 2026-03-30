#!/bin/bash
#SBATCH --job-name=olmo2_7b
#SBATCH --time=120:00:00
#SBATCH --partition=a3
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
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
LOAD_FOLDER="./step928646-unsharded" # please download this ckpt in advance
NUM_GPU=4
BATCH_SIZE=2

# parse options
while getopts "n:b:d:s:p:" opt; do
  case $opt in
    n) NUM_GPU=$OPTARG ;;    # -g <num_gpus>
    b) BATCH_SIZE=$OPTARG ;; # -b <batch_size>
    d) DATA_PREFIX=$OPTARG ;; # -d <data_prefix>
    s) SAVE_FOLDER=$OPTARG ;; # -s <save_folder>
    p) LOAD_FOLDER=$OPTARG ;; # -p <load_folder>
    \?) echo "Usage: $0 [-g num_gpu] [-b batch_size] [-d data_prefix]" >&2
        exit 1 ;;
  esac
done

echo "DATA_PREFIX=$DATA_PREFIX"
echo "NUM_GPU=$NUM_GPU"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "LOAD_FOLDER=$LOAD_FOLDER"


if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    # RoPE (baseline)
    CONFIG_NAME=OLMo2-7B-stage2-seed42-NONE
    torchrun --nproc_per_node=$NUM_GPU --master_port=$MYPORT scripts/train.py configs/official-1124/${CONFIG_NAME}.yaml --device_train_microbatch_size=${BATCH_SIZE} --save_overwrite=true --data.num_workers=24 --data_prefix=$DATA_PREFIX --save_folder=$SAVE_FOLDER/$CONFIG_NAME --load_path=$LOAD_FOLDER # --activation_checkpointing="one_in_eight"
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    # RePo (ours)
    CONFIG_NAME=OLMo2-7B-stage2-seed42-SEXMH-L10
    torchrun --nproc_per_node=$NUM_GPU --master_port=$MYPORT scripts/train.py configs/official-1124/${CONFIG_NAME}.yaml --device_train_microbatch_size=${BATCH_SIZE}  --save_overwrite=true --data.num_workers=24 --data_prefix=$DATA_PREFIX --save_folder=$SAVE_FOLDER/$CONFIG_NAME --load_path=$LOAD_FOLDER # --activation_checkpointing="one_in_eight"
else
    echo "unknown job id ${SLURM_ARRAY_TASK_ID}"
fi
