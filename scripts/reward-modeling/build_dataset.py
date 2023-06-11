import json

split = 'test'
with open(f'ckpts/t5-base-mdd-supervised/epoch_10_{split}_split_predictions.json', 'r') as f:
    lines = json.load(f)

outs = []
for line in lines:
    sos_len = len('<START-1>')
    eos_len = len('<END-1>')
    tmp_neg = {
        'prompt_text': line['prompt_text'],
        'utterance': line['generated_text'],
        'label': 0
    }
    tmp_pos = {
        'prompt_text': line['prompt_text'],
        'utterance': line['ref_text'][sos_len:-eos_len],
        'label': 1
    }
    if tmp_neg['utterance'].strip() != tmp_pos['utterance'].strip():
        outs.append(tmp_neg)
        outs.append(tmp_pos)
print(len(lines)*2)
print(len(outs))

with open(f'data/mdd_reward/{split}.json', 'w') as f:
    json.dump(outs, f, indent=4)