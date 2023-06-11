from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from datasets.arrow_dataset import Dataset
import torch
from sklearn.metrics import classification_report
from tqdm import tqdm
import json

def get_batch(samples, batch_size: int):
    current_ix = 0
    n_samples = len(samples)
    while current_ix < n_samples:
        current_batch = samples[current_ix : current_ix + batch_size]
        yield current_batch
        current_ix += batch_size

def get_dataset(file_path, split):
    with open(f'{file_path}/{split}.json', 'r') as f:
        samples = json.load(f)

    # get the training data in text, label format
    texts = []
    labels = []
    for sample in samples:
        input_text = sample['utterance'] + '[SEP]' + sample['prompt_text']
        texts.append(input_text)
        labels.append(sample['label'])
    
    print(np.unique(labels, return_counts=True))

    dataset = Dataset.from_dict(
            {
                "text": texts,
                "labels": labels
            },
            split=split
        )
    return dataset


dataset = 'mdd'
results_folder = f"ckpts/roberta-{dataset}-reward-model/checkpoint-13000"

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(results_folder)
ds_test = get_dataset(f'data/{dataset}_reward', 'test')

all_pred_labels = []
all_target_labels = []
batches = list(get_batch(ds_test, 10))
for batch in tqdm(batches):
    encoded = tokenizer(
        batch["text"],
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256)
    with torch.no_grad():
        outputs = model(input_ids=encoded.input_ids,
                        attention_mask=encoded.attention_mask)
        pred_labels = torch.argmax(outputs.logits, dim=1).tolist()
        all_pred_labels.extend(pred_labels)
        all_target_labels.extend(batch["labels"])

print(classification_report(all_target_labels, all_pred_labels))