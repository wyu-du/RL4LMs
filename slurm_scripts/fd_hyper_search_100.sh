#!/bin/bash
# --- this job will be run on any available node
# and simply output the node's hostname to my_job.output
echo "$HOSTNAME"
CUDA_VISIBLE_DEVICES=3 python scripts/training/hyper_search_test.py \
--config_path scripts/training/task_configs/faithdial/t5_ppo_100.yml \
--project_name rl4lm_exps \
--experiment_name t5-fd-supervised-ppo-100-hs \
--entity_name wyu-du \
--log_to_wandb 