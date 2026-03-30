#!/bin/bash


HF_CKPT_DIR=/path/to/hf_ckpts/
MODEL_NAME=OLMo2-7B-stage2-seed42-SEXMH-L10
STEP=step11931
STUDY_MODE=true
HF_UPLOAD=false

bash eval_ruler.sh $HF_CKPT_DIR/$MODEL_NAME/${STEP}-unsharded $MODEL_NAME $STUDY_MODE $HF_UPLOAD
bash eval_longbench.sh $HF_CKPT_DIR/$MODEL_NAME/${STEP}-unsharded $MODEL_NAME $STUDY_MODE $HF_UPLOAD
bash eval_table.sh $HF_CKPT_DIR/$MODEL_NAME/${STEP}-unsharded $MODEL_NAME $HF_UPLOAD
bash eval_graph.sh $HF_CKPT_DIR/$MODEL_NAME/${STEP}-unsharded $MODEL_NAME $STUDY_MODE $HF_UPLOAD
bash eval_general.sh $HF_CKPT_DIR/$MODEL_NAME/${STEP}-unsharded $MODEL_NAME $STUDY_MODE $HF_UPLOAD
