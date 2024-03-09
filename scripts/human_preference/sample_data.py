import json
import random
from datasets import load_metric
from rl4lms.data_pools.custom_text_generation_pools import FaithDial
sacrebleu = load_metric("sacrebleu")
from rl4lms.envs.text_generation.reward import f1_score

with open('ckpts/t5-base-mdd-supervised/epoch_10_val_split_predictions.json', 'r') as f:
    lines = json.load(f)
random.shuffle(lines)

with open('data/mdd_span_only/dprft_val.json', 'r') as f:
    docs = json.load(f)
# docs = FaithDial.prepare("val")
print(docs[0])
print(len(docs))

outs = []
for i, line in enumerate(lines):
    doc_id = int(line['sample_id'].split('_')[-1])
    ref_text = docs[doc_id]['sp_text']
    span_text = docs[doc_id]['answers'][0]
    query = line['prompt_text'].split('[SEP]')[0]
    # span_text = docs[doc_id][0].references[0]
    # ref_text = docs[doc_id][0].meta_data['knowledge_passage']
    # query = docs[doc_id][0].prompt_or_input_text.split('[SEP]')[0]
    generated_text = line['generated_text']
    t5_sacrebleu = sacrebleu.compute(predictions=[generated_text],
                                     references=[[span_text]])
    t5_sacrebleu = t5_sacrebleu['score']/100
    tok_know_f1 = f1_score(generated_text, ref_text)
    if len(span_text.split()) > 10 and t5_sacrebleu > 0.3 and '?' in query and tok_know_f1 < 1.: #and span_text in line['prompt_text']:
        print(f"====================== {line['sample_id']} =========================")
        print(f"span_text: {span_text}")
        print(f"ref_text: {ref_text}")
        print(f"t5_generated_text: {generated_text}")
        print(f"t5_sacre_bleu: {t5_sacrebleu}")
        print(f"t5_token_know_f1: {tok_know_f1}")
        # tmp = docs[doc_id]
        # tmp = {'id': docs[doc_id][0].id,
        #        'ctxs': docs[doc_id][0].prompt_or_input_text.split('context:')[-1], 
        #        'question': docs[doc_id][0].prompt_or_input_text.split('context:')[0],
        #        'answers': docs[doc_id][0].references,
        #        'sp_text': ref_text} 
        tmp = {'id': doc_id,
               'ctxs': line['prompt_text'].split('context:')[-1], 
               'question': line['prompt_text'].split('context:')[0],
               'answers': docs[doc_id]['answers'],
               'sp_text': ref_text} 
        tmp['t5_ft_generated_text'] = generated_text
        tmp['t5_ft_sacre_bleu'] = t5_sacrebleu
        tmp['t5_token_know_f1'] = tok_know_f1
        outs.append(tmp)
    if len(outs) == 50: break

with open('data/human_preference/t5_multidoc2dial_2024.json', 'w') as f:
    json.dump(outs, f, indent=4)
print(len(outs))

with open('data/human_preference/t5_multidoc2dial_2024.json', 'r') as f:
    lines = json.load(f)
print(len(lines))