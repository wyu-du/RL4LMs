import os
from argparse import ArgumentParser

import yaml

from rl4lms.envs.text_generation.logging_utils import Tracker
from rl4lms.envs.text_generation.training_utils import (
    OnPolicyTrainer,
    SupervisedTrainer,
    build_datapool
)
from rl4lms.envs.text_generation.utils_supervised import compute_metrics
import json


def main(
    config_path: str,
    project_name: str,
    experiment_name: str,
    base_path_to_store_results: str,
    entity_name: str,
    log_to_wandb: bool,
    epoch: int,
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
    trainer._evaluate_on_datapools(epoch=epoch, splits=["test"])


def compute_metric_on_baselines(config_path: str, model_output_path: str):
    # load the config file
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)

    metrics_config_dict=config["train_evaluation"].get("metrics")
    samples = build_datapool(config["datapool"])['test']
    all_prompt_texts, all_ref_texts, all_meta_infos = [], [], []
    fout = open('outputs/mdd_refs.jsonl', 'w')
    for sample in samples:
        all_prompt_texts.append(sample.prompt_or_input_text)
        all_ref_texts.append(sample.references)
        all_meta_infos.append(sample.meta_data)
        cur = {'prompt_or_input_text': sample.prompt_or_input_text, 
               'references': sample.references[0],
               'knowledge_text': sample.meta_data["knowledge_passage"]}
        print(cur)
        fout.write(json.dumps(cur)+'\n')
    print(len(samples))
    fout.close()

    with open(model_output_path, "r") as f:
        # lines = f.read().strip().split('\n')
        lines = json.load(f)
    all_generated_texts = []
    for line in lines:
        # line = json.loads(line)["generated_response"][0]
        line = line['generated_text']
        all_generated_texts.append(line)
    print(len(all_generated_texts))
    print(all_generated_texts[0])

    corpus_level_metrics, _ = compute_metrics(
        metrics_config_dict, samples, all_prompt_texts, all_generated_texts, 
        all_ref_texts, all_meta_infos, 'test', None)
    print(corpus_level_metrics)


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
    parser.add_argument(
        "--epoch", type=int, help='model checkpoint'
    )
    args = parser.parse_args()

    main(
        args.config_path,
        args.project_name,
        args.experiment_name,
        args.base_path_to_store_results,
        args.entity_name,
        args.log_to_wandb,
        args.epoch,
    )

    # compute_metric_on_baselines(args.config_path, 
    #                             f'{args.project_name}/{args.experiment_name}/epoch_{args.epoch}_test_split_predictions.json')
                                # 'data/baseline_outputs/generated_FaithDial_test_control-tokens_input350_maxLen50_greedy_entailed.jsonl')