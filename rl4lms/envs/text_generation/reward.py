from abc import ABC, abstractclassmethod

import torch
from datasets import load_metric
from rl4lms.envs.text_generation.observation import Observation
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from rl4lms.envs.text_generation.metric import (
    CIDERMetric,
    MeteorMetric,
    BERTScoreMetric,
    BLEUMetric,
    SpiceMetric,
    ParentToTTo,
    RougeLMax,
    TERMetric,
    chrFmetric,
    IntentAccuracyDailyDialog,
    CoherenceMultiDoc2Dial,
    SacreBLEUMetric,
    BERTF1Metric,
)
import numpy as np
from typing import List, Dict, Any
import regex
import string
from collections import Counter
import pickle
from sklearn.linear_model import LogisticRegression


#Normalization from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


class RewardFunction(ABC):
    @abstractclassmethod
    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        """
        Callable for reward functions for text generation

        Args:
            current_observation (Observation): previous observation (s)
            action (int): action performed (a) at s
            next_observation (Observation): observation after the action was performed (s')
            done (bool): whether the episode is finished or not
            meta_info (dict) - other information regarding textual sample
        Returns:
            float: scalar reward
        """
        raise NotImplementedError


class BatchedRewardFunction(ABC):
    """
    Computes rewards for several instances at once
    """

    @abstractclassmethod
    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_infos: List[Dict[str, Any]] = None,
    ) -> List[float]:
        """
        An abstract class for batched reward functions for text generation
        """
        raise NotImplementedError


### Automated reward functions ###########################


class CommonGenPenaltyShapingFunction(RewardFunction):
    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            prompt_text = next_observation.prompt_or_input_text
            prefix = "generate a sentence with: "
            concept_n_grams = prompt_text.split(prefix)[1][:-1]

            if (
                concept_n_grams.lower() in next_observation.context_text.lower()
                or prefix in next_observation.context_text.lower()
                or "generate" in next_observation.context_text.lower()
                or "sentence" in next_observation.context_text.lower()
            ):
                penalty_score = -1
            else:
                penalty_score = 0
            return penalty_score
        return 0


class BatchedCommonGenPenaltyShapingFunction(BatchedRewardFunction):
    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_info: Dict[str, Any] = None,
    ) -> List[float]:
        scores = []
        for done, prompt_text, gen_text in zip(dones, prompt_texts, gen_texts):
            if done:
                prefix = "generate a sentence with: "
                concept_n_grams = prompt_text.split(prefix)[1][:-1]

                if (
                    concept_n_grams.lower() in gen_text.lower()
                    or prefix in gen_text.lower()
                    or "generate" in gen_text.lower()
                    or "sentence" in gen_text.lower()
                ):
                    penalty_score = -1
                else:
                    penalty_score = 0
                scores.append(penalty_score)
        return scores


class MeteorRewardFunction(RewardFunction):
    def __init__(self, shaping_fn: str = None) -> None:
        super().__init__()
        self._metric = MeteorMetric()
        from rl4lms.envs.text_generation.registry import RewardFunctionRegistry

        self._shaping_fn = (
            RewardFunctionRegistry.get(shaping_fn, {})
            if shaping_fn is not None
            else shaping_fn
        )

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:

        # compute meteor at the end of episode
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_dict = self._metric.compute(None, predicted, references)
            score = metric_dict["lexical/meteor"][1]

            if self._shaping_fn is not None:
                aux_score = self._shaping_fn(
                    current_observation, action, next_observation, done, meta_info
                )
                score = score + aux_score
            return score
        return 0


class RougeRewardFunction(RewardFunction):
    def __init__(
        self, rouge_type: str, shaping_fn: str = None, use_single_ref: bool = True
    ) -> None:
        super().__init__()
        self._metric = load_metric("rouge")
        self._rouge_type = rouge_type
        from rl4lms.envs.text_generation.registry import RewardFunctionRegistry

        self._shaping_fn = (
            RewardFunctionRegistry.get(shaping_fn, {})
            if shaping_fn is not None
            else shaping_fn
        )
        self._use_single_ref = use_single_ref

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            # TBD: considers only one reference for now
            if self._use_single_ref:
                references = [next_observation.target_or_reference_texts[0]]
            else:
                references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]

            metric_results = self._metric.compute(
                predictions=predicted, references=references, use_stemmer=True
            )
            reward = metric_results[self._rouge_type].mid.fmeasure
            if self._shaping_fn is not None:
                aux_score = self._shaping_fn(
                    current_observation, action, next_observation, done, meta_info
                )
                reward = reward + aux_score
            return reward
        return 0


class RougeCombined(RewardFunction):
    def __init__(self, shaping_fn: str = None) -> None:
        super().__init__()
        self._metric = load_metric("rouge")
        from rl4lms.envs.text_generation.registry import RewardFunctionRegistry

        self._shaping_fn = (
            RewardFunctionRegistry.get(shaping_fn, {})
            if shaping_fn is not None
            else shaping_fn
        )

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            # TBD: considers only one reference for now
            references = [next_observation.target_or_reference_texts[0]]
            predicted = [next_observation.context_text]

            metric_results = self._metric.compute(
                predictions=predicted, references=references, use_stemmer=True
            )

            rouge_keys = ["rouge1", "rouge2", "rougeL"]
            scores = [
                metric_results[rouge_type].mid.fmeasure for rouge_type in rouge_keys
            ]
            reward = np.mean(scores)
            if self._shaping_fn is not None:
                aux_score = self._shaping_fn(
                    current_observation, action, next_observation, done, meta_info
                )
                reward = reward + aux_score
            return reward
        return 0


class BERTScoreRewardFunction(RewardFunction):
    def __init__(self, language: str = "en") -> None:
        super().__init__()
        self._metric = BERTScoreMetric(language)

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_results = self._metric.compute(None, predicted, references)
            bert_score = metric_results["semantic/bert_score"][1]
            return bert_score
        return 0


class BLEURewardFunction(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self._metric = BLEUMetric()

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_results = self._metric.compute(None, predicted, references)
            bleu_score = metric_results["lexical/bleu"][1]
            return bleu_score
        return 0


class SacreBleu(RewardFunction):
    def __init__(self, **args) -> None:
        super().__init__()
        self._metric = load_metric("sacrebleu")
        self._args = args

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_results = self._metric.compute(
                predictions=predicted, references=references, **self._args
            )
            return metric_results["score"] / 100
        return 0

class LearnedBatchedRewardFunction(BatchedRewardFunction):
    def __init__(
        self, model_name: str, label_ix: int, include_prompt_for_eval: bool = True
    ) -> None:
        super().__init__()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._metric_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._metric_tokenizer.truncation_side = "left"
        self._metric_model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self._device)
        self._label_ix = label_ix
        self._include_prompt_for_eval = include_prompt_for_eval

    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_infos: List[Dict[str, Any]] = None,
    ) -> List[float]:

        # collect generations at done steps
        gens = []
        indices_with_done = []
        rewards = torch.zeros(len(prompt_texts))
        for ix, (prompt, gen, ref, done) in enumerate(
            zip(prompt_texts, gen_texts, ref_texts, dones)
        ):
            if done:
                generated_text = prompt if self._include_prompt_for_eval else ""
                generated_text += gen
                gens.append(generated_text)
                indices_with_done.append(ix)

        # compute rewards at once
        with torch.no_grad():
            encoded = self._metric_tokenizer(
                gens, return_tensors="pt", truncation=True, padding=True
            )
            outputs = self._metric_model(
                input_ids=encoded.input_ids.to(self._device),
                attention_mask=encoded.attention_mask.to(self._device),
            )
            scores = torch.softmax(outputs.logits, dim=1)
            scores = scores[:, self._label_ix].flatten().cpu()
            rewards[indices_with_done] = scores

        return rewards



class SpiderRewardFunction(BatchedRewardFunction):
    def __init__(
        self, spice_coeff: float, cider_coeff: float, shaping_fn: str = None
    ) -> None:
        """
        Spice + Cider
        """
        super().__init__()
        self._spice_metric = SpiceMetric()
        self._cider_metric = CIDERMetric()
        self._spice_coeff = spice_coeff
        self._cider_coeff = cider_coeff
        from rl4lms.envs.text_generation.registry import RewardFunctionRegistry

        self._shaping_fn = (
            RewardFunctionRegistry.get(shaping_fn, {})
            if shaping_fn is not None
            else shaping_fn
        )

    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_info: Dict[str, Any] = None,
    ) -> List[float]:
        prompts = []
        gens = []
        refs = []
        indices_with_done = []
        rewards = torch.zeros(len(prompt_texts))
        for ix, (prompt, gen, ref, done) in enumerate(
            zip(prompt_texts, gen_texts, ref_texts, dones)
        ):
            if done:
                prompts.append(prompt)
                gens.append(gen)
                refs.append(ref)
                indices_with_done.append(ix)

        if len(indices_with_done) > 0:
            spice_scores = self._spice_metric.compute(prompts, gens, refs)[
                "lexical/spice"
            ][0]
            cider_scores = self._cider_metric.compute(prompts, gens, refs)[
                "lexical/cider"
            ][0]
            total_scores = self._spice_coeff * np.array(
                spice_scores
            ) + self._cider_coeff * np.array(cider_scores)

            if self._shaping_fn is not None:
                aux_scores = self._shaping_fn(prompt_texts, gen_texts, ref_texts, dones)
            else:
                aux_scores = [0] * len(indices_with_done)

            for ind, score, aux_score in zip(
                indices_with_done, total_scores, aux_scores
            ):
                rewards[ind] = score + aux_score

        return rewards


#############################################################################

########## Learned Reward Functions##########################################


class LearnedRewardFunction(RewardFunction):
    def __init__(
        self, model_name: str, label_ix: int, include_prompt_for_eval: bool = True
    ) -> None:
        super().__init__()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._metric_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._metric_tokenizer.truncation_side = "left"
        self._metric_model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self._device)
        self._label_ix = label_ix
        self._include_prompt_for_eval = include_prompt_for_eval

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            generated_text = (
                current_observation.prompt_or_input_text
                if self._include_prompt_for_eval
                else ""
            )
            generated_text += next_observation.context_text

            with torch.no_grad():
                encoded = self._metric_tokenizer(
                    generated_text, return_tensors="pt", truncation=True, padding=True
                )
                outputs = self._metric_model(
                    input_ids=encoded.input_ids.to(self._device),
                    attention_mask=encoded.attention_mask.to(self._device),
                )
                scores = torch.softmax(outputs.logits.flatten(), dim=0)
                score = scores[self._label_ix].item()
                return score
        return 0


class BLEURTRewardFunction(RewardFunction):
    def __init__(self, checkpoint: str = None):
        super().__init__()
        self._metric = load_metric("bleurt", checkpoint=checkpoint)

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_results = self._metric.compute(
                predictions=predicted, references=references
            )
            score = metric_results["scores"][0]
            return score
        return 0


class PARENTRewardFunction(RewardFunction):
    """
    PARENT F1 score as the reward
    """

    def __init__(self) -> None:
        super().__init__()
        self._metric = ParentToTTo()

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            generated_texts = [next_observation.context_text]
            meta_infos = [meta_info]
            scores = self._metric.compute(None, generated_texts, None, meta_infos)
            reward = scores["table_to_text/parent_overall_f_score"][0][0]
            return reward
        return 0


class RougeLMaxRewardFunction(RewardFunction):
    def __init__(self, **args) -> None:
        super().__init__()
        self._metric = RougeLMax(**args)

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            meta_infos = [meta_info]
            scores = self._metric.compute(None, predicted, references, meta_infos)
            reward = scores["lexical/rouge_l_max"][0][0]
            return reward
        return 0


class TER(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self._metric = TERMetric()

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:

        # compute score at the end of episode
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_dict = self._metric.compute(None, predicted, references)
            score = metric_dict["lexical/ter"][1]
            score = 1 - score
            return score
        return 0


class chrF(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self._metric = chrFmetric()

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:

        # compute score at the end of episode
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_dict = self._metric.compute(None, predicted, references)
            score = metric_dict["lexical/chrf"][1]
            return score
        return 0


class IntentAccuracy(BatchedRewardFunction):
    def __init__(
        self, shape: bool = True, intent_coeff: float = 1.0, auto_coeff: float = 1.0
    ) -> None:
        super().__init__()
        self._metric = None
        self._shape = shape
        self._intent_coeff = intent_coeff
        self._auto_coeff = auto_coeff
        self._shaping_metric = MeteorMetric()

    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_infos: List[Dict[str, Any]] = None,
    ) -> List[float]:

        if self._metric is None:
            self._metric = IntentAccuracyDailyDialog()

        # compute rewards for finished episodes only
        rewards = np.zeros(len(gen_texts))

        done_prompt_texts = []
        done_gen_texts = []
        done_ref_texts = []
        done_meta_infos = []
        done_ixs = []
        for ix, (prompt, gen, ref, meta_info, done) in enumerate(
            zip(prompt_texts, gen_texts, ref_texts, meta_infos, dones)
        ):
            if done:
                done_prompt_texts.append(prompt)
                done_gen_texts.append(gen)
                done_ref_texts.append(ref)
                done_meta_infos.append(meta_info)
                done_ixs.append(ix)

                if self._shape:
                    score = self._shaping_metric.compute(
                        done_prompt_texts, done_gen_texts, done_ref_texts
                    )
                    rewards[ix] = self._auto_coeff * score["lexical/meteor"][1]

        scores = self._metric.compute(
            done_prompt_texts, done_gen_texts, done_ref_texts, done_meta_infos
        )["intent/accuracy"][0]
        rewards[done_ixs] += self._intent_coeff * np.array(scores)
        return rewards.tolist()
    

class BERTKnowF1RewardFunction(RewardFunction):
    def __init__(self, language: str = "en") -> None:
        super().__init__()
        self._metric = load_metric("bertscore")
        self._language = language
        self._last_gpu = f"cuda:{torch.cuda.device_count() - 1}"

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            truncated_prompt = ' '.join(next_observation.prompt_or_input_text.split('context:')[-1].split()[:100])
            prompts = [truncated_prompt]
            predicted = [next_observation.context_text]
            metric_results = self._metric.compute(predictions=predicted, references=prompts,
                                                  lang=self._language, device=self._last_gpu)
            bert_score = np.mean(metric_results["f1"])
            return bert_score
        return 0


class CoherenceMDDRewardFunction(RewardFunction):
    def __init__(self, batch_size: int = 1) -> None:
        super().__init__()
        self._metric = CoherenceMultiDoc2Dial(batch_size)

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            truncated_prompt = ' '.join(next_observation.prompt_or_input_text.split()[:100])
            prompts = [truncated_prompt]
            predicted = [next_observation.context_text]
            metric_results = self._metric.compute(prompts, predicted, None)
            score = metric_results["lexical/coherence"][1]
            return score
        return 0


# class CombinedRewardFunction(RewardFunction):
#     def __init__(self, 
#                  language: str = "en", 
#                  cof_info: float = 0.5,
#                  cof_faith: float = 0.5) -> None:
#         super().__init__()
#         self._info_metric = load_metric("sacrebleu")
#         self._faithful_metric = load_metric("bertscore")
#         self._language = language
#         self._last_gpu = f"cuda:{torch.cuda.device_count() - 1}"
#         self.cof_info = cof_info
#         self.cof_faith = cof_faith

#     def __call__(
#         self,
#         current_observation: Observation,
#         action: int,
#         next_observation: Observation,
#         done: bool,
#         meta_info: Dict[str, Any] = None,
#     ) -> float:
#         if done:
#             # accuracy score
#             references = [next_observation.target_or_reference_texts]
#             predicted = [next_observation.context_text]
#             metric_results = self._info_metric.compute(predictions=predicted, references=references)
#             info_score = metric_results["score"] / 100

#             # faithfulness score
#             truncated_prompt = ' '.join(next_observation.prompt_or_input_text.split('context:')[-1].split()[:100])
#             prompts = [truncated_prompt]
#             predicted = [next_observation.context_text]
#             metric_results = self._faithful_metric.compute(predictions=predicted, references=prompts,
#                                                            lang=self._language, device=self._last_gpu)
#             bert_score = np.mean(metric_results["f1"])

#             # total combined score
#             score = self.cof_info*info_score + self.cof_faith*bert_score
#             return score
#         return 0
    

class CombinedRewardFunction(RewardFunction):
    def __init__(self, 
                 cof_info: float = 0.5,
                 cof_faith: float = 0.5) -> None:
        super().__init__()
        self._info_metric = load_metric("sacrebleu")
        self.cof_info = cof_info
        self.cof_faith = cof_faith

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            # accuracy score
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_results = self._info_metric.compute(predictions=predicted, references=references)
            info_score = metric_results["score"] / 100
            # ref_f1 = f1_score(predicted[0], references[0][0])
            # print('ref_f1:', ref_f1)

            # faithfulness score
            truncated_prompt = ' '.join(next_observation.prompt_or_input_text.split('context:')[-1].split()[:100])
            prompts = [truncated_prompt]
            predicted = [next_observation.context_text]
            f1 = f1_score(predicted[0], prompts[0])
            # print('know_f1:', f1)

            # total combined score
            score = self.cof_info*info_score + self.cof_faith*f1
            # print('combined:', score)
            return score
        return 0


class CombinedRewardWithSpanFunction(RewardFunction):
    def __init__(self, 
                 language: str = "en", 
                 cof_info: float = 0.5,
                 cof_faith: float = 0.5) -> None:
        super().__init__()
        self._info_metric = SacreBLEUMetric()
        self._faithful_metric = BERTF1Metric(language)
        self.cof_info = cof_info
        self.cof_faith = cof_faith

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            # accuracy score
            predicted = [next_observation.context_text]
            references = [next_observation.target_or_reference_texts]
            metric_results = self._info_metric.compute(prompt_texts=None, generated_texts=predicted, reference_texts=references)
            info_score = metric_results["lexical/sacrebleu"][1]

            # faithfulness score
            metric_results = self._faithful_metric.compute(prompt_texts=None, generated_texts=predicted, 
                                                           reference_texts=None, meta_infos=[meta_info])
            bert_score = metric_results["lexical/bert_know_f1"][1]

            # total combined score
            score = self.cof_info*info_score + self.cof_faith*bert_score
            return score
        return 0


class MDDRewardFunction(RewardFunction):
    def __init__(self, batch_size: int = 1) -> None:
        super().__init__()
        self._tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self._model = AutoModelForSequenceClassification.from_pretrained(
            "/p/fewshot/RL4LMs/ckpts/roberta-mdd-reward-model/checkpoint-13000"
        )
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)
        self._batch_size = batch_size

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            truncated_prompt = ' '.join(next_observation.prompt_or_input_text.split()[:100])
            input_texts = [next_observation.context_text + '[SEP]' + truncated_prompt]
            encoded = self._tokenizer(input_texts, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                outputs = self._model(input_ids=encoded.input_ids.to(self._device),
                                      attention_mask=encoded.attention_mask.to(self._device))
                scores = torch.softmax(outputs['logits'], dim=-1)[:, -1].tolist() # ensure shape is (B)
            return scores[0]
        return 0
    

class FDRewardFunction(RewardFunction):
    def __init__(self, batch_size: int = 1) -> None:
        super().__init__()
        self._tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self._model = AutoModelForSequenceClassification.from_pretrained(
            "/p/fewshot/RL4LMs/ckpts/roberta-fd-reward-model/checkpoint-11000"
        )
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)
        self._batch_size = batch_size

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            truncated_prompt = ' '.join(next_observation.prompt_or_input_text.split()[:100])
            input_texts = [next_observation.context_text + '[SEP]' + truncated_prompt]
            encoded = self._tokenizer(input_texts, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                outputs = self._model(input_ids=encoded.input_ids.to(self._device),
                                      attention_mask=encoded.attention_mask.to(self._device))
                scores = torch.softmax(outputs['logits'], dim=-1)[:, -1].tolist() 
            return scores[0]
        return 0


class TokenKnowF1RewardFunction(RewardFunction):
    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            f1 = 0
            truncated_prompt = ' '.join(next_observation.prompt_or_input_text.split('context:')[-1].split()[:100])
            prompts = [truncated_prompt]
            predicted = [next_observation.context_text]
            f1 = f1_score(predicted[0], prompts[0])
            return f1
        return 0
    

class TokenRefF1RewardFunction(RewardFunction):
    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            f1 = f1_score(predicted[0], references[0][0])
            return f1
        return 0
    

class CombinedRewardFunction_v2(RewardFunction):
    def __init__(self, model_path: str = './ckpts/mdd_human_comparison.sav') -> None:
        super().__init__()
        self._info_metric = load_metric("sacrebleu")
        # load model weight
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            # accuracy score
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_results = self._info_metric.compute(predictions=predicted, references=references)
            info_score = metric_results["score"] / 100
            print('BLEU:', info_score)

            # faithfulness score
            f1 = 0
            truncated_prompt = ' '.join(next_observation.prompt_or_input_text.split('context:')[-1].split()[:100])
            prompts = [truncated_prompt]
            predicted = [next_observation.context_text]
            f1 = f1_score(predicted[0], prompts[0])
            print('K-F1:', f1)

            # total combined score
            score = self.model.predict_proba(np.array([[info_score, f1]]))[0][-1]
            print('Combined:', score)
            return score
        return 0


class BLEUKnowRewardFunction(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self._metric = load_metric("sacrebleu")

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            truncated_prompt = ' '.join(next_observation.prompt_or_input_text.split('context:')[-1].split()[:100])
            prompts = [truncated_prompt]
            predicted = [next_observation.context_text]
            metric_results = self._metric.compute(
                predictions=predicted, references=[prompts],
            )
            return metric_results["score"] / 100
        return 0
    

if __name__ == "__main__":
    knowledge = "hello there"
    predictions = "hello there general kenobi xxx"
    references = ["hello there general kenobi", "hello there!!"]
    observation = Observation(
        None, None, knowledge, None, None, predictions, references, None, None, None, None
    )

    # reward_fn = MeteorRewardFunction()
    # print(reward_fn(None, None, observation, True))

    # reward_fn = chrF()
    # print(reward_fn(None, None, observation, True))

    # reward_fn = RougeCombined()
    # print(reward_fn(None, None, observation, True))

    # reward_fn = RougeRewardFunction(rouge_type="rouge1")
    # print(reward_fn(None, None, observation, True))

    # reward_fn = RougeRewardFunction(rouge_type="rouge2")
    # print(reward_fn(None, None, observation, True))

    # reward_fn = RougeRewardFunction(rouge_type="rougeL")
    # print(reward_fn(None, None, observation, True))

    # reward_fn = BERTScoreRewardFunction(language="en")
    # print(reward_fn(None, None, observation, True))

    # reward_fn = BLEURewardFunction()
    # print(reward_fn(None, None, observation, True))

    # reward_fn = BLEURTRewardFunction()
    # print(reward_fn(None, None, observation, True))
    print("knowledge = hello there")
    print("references = ['hello there general kenobi', 'hello there!!']")
    print("predictions = hello there general kenobi xxx")
    # reward_fn = CombinedRewardFunction_v2()
    reward_fn = BLEUKnowRewardFunction()
    # reward_fn = TokenKnowF1RewardFunction()
    print(reward_fn(None, None, observation, True, None))
