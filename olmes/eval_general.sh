#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --time=100:00:00
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
CAT=general
OUTPUT_DIR=results_hist/$CAT


MODEL_PATH=$1
MODEL_NAME=$2
STUDY_MODE=$3
HF_UPLOAD=$4

for TASK in mmlu_pro:cot::none arc_challenge:mc::large arc_easy:rc::large boolq:rc::large coqa::large drop::large hellaswag:rc::large triviaqa::olmes
do
    CUDA_VISIBLE_DEVICES=0 olmes --task $TASK --batch-size 10000 --model $MODEL_PATH --model-args "{\"model_path\": \"$MODEL_PATH\", \"max_length\": 65536, \"model_type\": \"vllm\"}"  --output-dir $OUTPUT_DIR/${MODEL_NAME//\//_}/$TASK --save-raw-requests true --num-workers 1 --gpus 1

    if [ "$STUDY_MODE" = "true" ]; then
        echo "Study mode enabled: Stopping after the first task."
        break
    fi
    if [ "$HF_UPLOAD" = "true" ]; then
        echo "Uploading Scores to HF"
        hf upload ghrua/OLMo2-7B-Logs $OUTPUT_DIR/${MODEL_NAME//\//_}/$TASK/metrics-all.jsonl $OUTPUT_DIR/${MODEL_NAME//\//_}/$TASK/metrics-all.jsonl
    fi
done