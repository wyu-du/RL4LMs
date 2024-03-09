import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/roberta-large-faithcritic", return_tensors="pt")
model = AutoModelForSequenceClassification.from_pretrained("McGill-NLP/roberta-large-faithcritic").to('cuda')

knowledge_list = []
with open('outputs/mdd_refs.jsonl', 'r') as f:
    lines = f.read().strip().split('\n')
    for line in lines:
        knowledge_text = json.loads(line)['knowledge_text']
        knowledge_text = ' '.join(knowledge_text.split(' ')[:200])
        knowledge_list.append(knowledge_text)
response_list = []
with open('outputs/mdd_alpha_025.json', 'r') as f:
    lines = json.load(f)
    # lines = f.read().strip().split('\n')
    for line in lines:
        response_list.append(line['generated_text'])
        # response_list.append(json.loads(line)['response'])

pred_list = []
for knowledge, response in zip(knowledge_list, response_list):
    input = tokenizer(knowledge, response, return_tensors="pt").to('cuda')
    pred = torch.argmax(model(**input).logits).item()
    pred_list.append(pred)
score = sum(pred_list)/float(len(pred_list))
print(score * 100)