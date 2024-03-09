#!/bin/bash
# --- this job will be run on any available node
# and simply output the node's hostname to my_job.output
echo "$HOSTNAME"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/centos/cuda/11.7.0/lib64

CUDA_VISIBLE_DEVICES=0,1 python scripts/training/train_text_generation.py \
--config_path scripts/training/task_configs/faithdial/t5_ppo_on_supervised.yml \
--project_name rl4lm_exps \
--experiment_name t5-fd-supervised-ppo-full-combined_lr \
--entity_name wyu-du \
--log_to_wandb 
