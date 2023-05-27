#!/bin/bash
# --- this job will be run on any available node
# and simply output the node's hostname to my_job.output
# SBATCH --job-name="ctx20_len512"
# SBATCH --error="ctx20_len512.err"
# SBATCH --output="ctx20_len512.output"
# SBATCH --partition=gpu
# SBATCH --gres=gpu:4
echo "$HOSTNAME"
# export SLURM_NTASKS=1
# export WORLD_SIZE=1
# export MASTER_ADDR="127.0.0.1" # change this and the following line if needed
# export MASTER_PORT="10020" 
# export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=5,6,7 python scripts/training/train_text_generation.py \
--config_path scripts/training/task_configs/multidoc2dial/t5_supervised.yml \
--project_name rl4lm_exps \
--experiment_name t5-mdd-supervised-1k-span-insert-e5 \
--entity_name wyu-du \
--log_to_wandb 
