import os
from argparse import ArgumentParser

import yaml

from rl4lms.envs.text_generation.logging_utils import Tracker
from rl4lms.envs.text_generation.training_utils import (
    OnPolicyTrainer,
    SupervisedTrainer,
)
import json


def main(
    config_path: str,
    project_name: str,
    experiment_name: str,
    base_path_to_store_results: str,
    entity_name: str,
    log_to_wandb: bool,
):

    # load the config file
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)

    # load tracker
    tracker = Tracker(
        base_path_to_store_results,
        config,
        project_name,
        experiment_name,
        entity_name,
        log_to_wandb,
    )

    # instantiate the trainer here
    if "supervised" in config["alg"]["id"]:
        trainer = SupervisedTrainer(
            tokenizer_config=config["tokenizer"],
            datapool_config=config["datapool"],
            alg_config=config["alg"],
            train_eval_config=config["train_evaluation"],
            tracker=tracker,
        )
    else:
        trainer = OnPolicyTrainer(
            tokenizer_config=config["tokenizer"],
            datapool_config=config["datapool"],
            reward_config=config["reward_fn"],
            env_config=config["env"],
            on_policy_alg_config=config["alg"],
            train_eval_config=config["train_evaluation"],
            tracker=tracker,
        )
    iter_start = trainer._trainer_state["current_iter"]
    trainer._evaluate_on_datapools(epoch=iter_start, splits=["val", "test"])

def read_best_dev(args):
    with open(f'{args.project_name}/{args.experiment_name}/val_split_metrics.jsonl', 'r') as f:
        lines = f.read().strip().split('\n')
    total_scores_dict = {}
    for line in lines:
        obj = json.loads(line)
        if obj['epoch'] == 0: continue
        total_score = 0
        for key, val in obj['metrics'].items():
            if key in ['lexical/sacrebleu', 'lexical/rouge_rougeL', 'semantic/bert_score', 
                        'lexical/bert_know_f1', 'lexical/knowledge_f1']:
                if key == 'lexical/sacrebleu':
                    total_score += val * 0.28
                if key == 'lexical/bert_know_f1':
                    total_score += val * 0.72
                # total_score += val
        total_scores_dict[obj['epoch']] = total_score
    max_key = max(total_scores_dict, key=total_scores_dict.get)
    # max_key = 199
    max_value = total_scores_dict[max_key]
    print(f'Best epoch {max_key}: {max_value}')
    
    for line in lines:
        obj = json.loads(line)
        if obj['epoch'] == max_key:
            for key, val in obj['metrics'].items():
                if key in ['lexical/sacrebleu', 'lexical/rouge_rougeL', 'semantic/bert_score', 
                           'lexical/bert_know_f1', 'lexical/knowledge_f1']:
                    print(f'{key}: {val}')

    


if __name__ == "__main__":
    parser = ArgumentParser(description="Fine-tune LM to generate controlled text")
    parser.add_argument("--config_path", type=str, help="path to the config file")
    parser.add_argument(
        "--project_name", type=str, help="WANDB project name", default="rl4lm_exps"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="WANDB experiment name",
        default="rl4lm_experiment",
    )
    parser.add_argument(
        "--entity_name", type=str, help="WANDB entity name", default=None
    )
    parser.add_argument(
        "--base_path_to_store_results",
        type=str,
        help="Base path to store experiment results",
        default=os.getcwd(),
    )
    parser.add_argument(
        "--log_to_wandb", action="store_true", help="Whether to use wandb logging"
    )
    args = parser.parse_args()

    # main(
    #     args.config_path,
    #     args.project_name,
    #     args.experiment_name,
    #     args.base_path_to_store_results,
    #     args.entity_name,
    #     args.log_to_wandb,
    # )

    read_best_dev(args)