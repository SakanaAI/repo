#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --time=36:00:00
#SBATCH --partition=a3
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --cpus-per-task=32

######################
# ENV Config
######################
CONDA_ROOT_DIR=$(conda info --base)
source $CONDA_ROOT_DIR/etc/profile.d/conda.sh
conda activate olmes
PY_BIN=${CONDA_ROOT_DIR}/${SUFFIX}/bin
######################



CURRENT_DATE=$(date +"%y%m%d")
OUTPUT_DIR=results_hist/${CURRENT_DATE}_results_graph

MODEL_DIR=../OLMo/hf_ckpts
MODEL=OLMo2-1B-stage2-seed42-SEXMH-L5
STEP=step23852-unsharded
TASK="nlgraph:easy::none"
olmes --task $TASK --batch-size 10000 --model $MODEL_DIR/$MODEL/$STEP --model-args "{\"model_path\": \"$MODEL_DIR/$MODEL/$STEP\", \"max_length\": 4096, \"model_type\": \"vllm\"}"  --output-dir $OUTPUT_DIR/${MODEL//\//_}/$TASK --save-raw-requests true --num-workers 1 --gpus 1