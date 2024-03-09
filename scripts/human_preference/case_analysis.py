import json
import numpy as np
from datasets import load_metric
sacrebleu = load_metric("sacrebleu")

# with open('rl4lm_exps/t5-fd-supervised-ppo-full-sacre_bleu/epoch_399_test_split_predictions.json', 'r') as f:
#     bleu_list = json.load(f)
# with open('rl4lm_exps/t5-fd-supervised-ppo-full-bert_know_f1/epoch_1699_test_split_predictions.json', 'r') as f:
#     bert_list = json.load(f)
# with open('rl4lm_exps/t5-fd-supervised-ppo-full-combined092/epoch_499_test_split_predictions.json', 'r') as f:
#     ours_list = json.load(f)
with open('rl4lm_exps/t5-mdd-supervised-ppo-full-sacre_bleu/epoch_699_test_split_predictions.json', 'r') as f:
    bleu_list = json.load(f)
with open('rl4lm_exps/t5-mdd-supervised-ppo-full-bert_know_f1/epoch_4099_test_split_predictions.json', 'r') as f:
    bert_list = json.load(f)
with open('rl4lm_exps/t5-mdd-supervised-ppo-full-combined004/epoch_399_test_split_predictions.json', 'r') as f:
    ours_list = json.load(f)

def compute_bleu(generated_text, ref_text):
    bleu = sacrebleu.compute(predictions=[generated_text],
                             references=[[ref_text]])
    bleu = bleu['score']/100
    return bleu

for i in range(len(ours_list)):
    if i < 600: continue
    bleu = bleu_list[i]
    bleu_score = compute_bleu(bleu['generated_text'], bleu['ref_text']) + float(bleu['lexical/bert_know_f1'])
    
    bert = bert_list[i]
    bert_score = compute_bleu(bert['generated_text'], bert['ref_text']) + float(bert['lexical/bert_know_f1'])
    
    ours = ours_list[i]
    ours_score = compute_bleu(ours['generated_text'], ours['ref_text']) + float(ours['lexical/bert_know_f1'])
    
    max_id = np.argmax([bleu_score, bert_score, ours_score])
    knowledge = ours['prompt_text'].split('context:')[1].strip()
    if (max_id == 2):
        print('========== History ==========')
        print(ours['prompt_text'].split('context:')[0])
        print('========== Knowledge ==========')
        print(ours['prompt_text'].split('context:')[1].strip())
        print('========== ref_text ==========')
        print(ours['ref_text'][len('<START-1>'):-len('<END-1>')])
        print('========== bleu_generated ==========')
        print(bleu['generated_text'])
        print('========== bert_generated ==========')
        print(bert['generated_text'])
        print('========== ours_generated ==========')
        print(ours['generated_text'])
        break