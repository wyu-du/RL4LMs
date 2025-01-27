#!/bin/bash
# --- this job will be run on any available node
# and simply output the node's hostname to my_job.output
echo "$HOSTNAME"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/centos/cuda/11.7.0/lib64

CUDA_VISIBLE_DEVICES=2,3 python scripts/training/train_text_generation.py \
--config_path scripts/training/task_configs/multidoc2dial/t5_ppo_on_supervised.yml \
--project_name rl4lm_exps \
--experiment_name t5-mdd-supervised-ppo-full-token_know_f1 \
--entity_name wyu-du \
--log_to_wandb 
