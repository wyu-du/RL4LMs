import os
from argparse import ArgumentParser

import yaml

from rl4lms.envs.text_generation.logging_utils import Tracker
from rl4lms.envs.text_generation.training_utils import (
    OnPolicyTrainer,
    SupervisedTrainer,
)

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


def main(
    config_path: str,
    project_name: str,
    experiment_name: str,
    base_path_to_store_results: str,
    entity_name: str,
    log_to_wandb: bool,
    reward_config,
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

    trainer = OnPolicyTrainer(
        tokenizer_config=config["tokenizer"],
        datapool_config=config["datapool"],
        reward_config=reward_config,
        env_config=config["env"],
        on_policy_alg_config=config["alg"],
        train_eval_config=config["train_evaluation"],
        tracker=tracker,
    )
    total_scores = trainer.train_and_eval_hyper_search()
    return total_scores

def objective(trial: optuna.Trial):
    info_cofs = trial.suggest_float("cof_info", 0.0, 1.0, step=0.01)
    # faith_cofs = trial.suggest_float("cof_faith", 0.0, 1.0, step=0.1)
    # coherence_cofs = trial.suggest_float("cof_coherence", 0.0, 1.0, step=0.1)

    reward_config = {"id": 'ours_combined', 
                     "args": {"cof_info": info_cofs, 
                              "cof_faith": 1-info_cofs,
                              "language": 'en'}}
    total_scores = main(
        args.config_path,
        args.project_name,
        args.experiment_name,
        args.base_path_to_store_results,
        args.entity_name,
        args.log_to_wandb,
        reward_config
    )
    return total_scores


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

    sampler = TPESampler(n_startup_trials=5)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=100 // 3)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=100)

    best_params = study.best_params
    best_value = study.best_value
    print("best_value = "+str(best_value))
    print("best_params:")
    print(best_params)