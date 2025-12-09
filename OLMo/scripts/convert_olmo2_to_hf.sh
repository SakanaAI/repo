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

export WANDB_API_KEY="511a2692431f62cfb93c7c1f8bbe78bae0b9c3df"
######################
STEP_NAME=step23852
# for RUN_NAME in OLMo2-1B-stage2-seed42-NOPE OLMo2-1B-stage2-seed42-SEXMH-L0 OLMo2-1B-stage2-seed42-SEXMH-L5 OLMo2-1B-stage2-seed42-NONE 
# for RUN_NAME in OLMo2-1B-stage2-seed42-NOPE-L5 OLMo2-1B-stage2-seed42-SEXMHFEX-L5 OLMo2-1B-stage2-seed42-SEXMH-L3 OLMo2-1B-stage2-seed42-SEXMHPD-L5
for RUN_NAME in OLMo2-1B-stage2-seed42-SEXMH-L7 OLMo2-1B-stage2-seed42-NOPE-n2r1 OLMo2-1B-stage2-seed42-NOPE-r2n1
do
    CKPT_DIR=$RUN_NAME/${STEP_NAME}-unsharded/
    IN_DIR=ckpts/$CKPT_DIR
    SUFFIX=hf_ckpts/$CKPT_DIR

    python scripts/convert_olmo2_to_hf.py --input_dir $IN_DIR --output_dir $SUFFIX #--no_tokenizer
    breka
done

python scripts/change_max_model_lens.py