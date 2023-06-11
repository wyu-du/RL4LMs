from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from datasets.arrow_dataset import Dataset
import json

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


def main():

    # results folder
    dataset = 'fd'
    results_folder = f"ckpts/roberta-{dataset}-reward-model"

    # train and val dataset
    ds_train = get_dataset(f'data/{dataset}_reward', 'train')
    ds_test = get_dataset(f'data/{dataset}_reward', 'val')

    model_name = "roberta-base"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(examples):
        outputs = tokenizer(examples['text'], truncation=True, max_length=256)
        return outputs

    tokenized_ds_train = ds_train.map(tokenize, batched=True)
    tokenized_ds_test = ds_test.map(tokenize, batched=True)

    def compute_metrics(eval_preds):
        metric = load_metric("accuracy")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(num_train_epochs=10,
                                      output_dir=results_folder,
                                      per_device_train_batch_size=8,
                                      per_device_eval_batch_size=64,
                                      evaluation_strategy="steps",
                                      save_strategy='steps',
                                      logging_steps=50,
                                      save_total_limit=1,
                                      save_steps=500,
                                      lr_scheduler_type="constant",
                                      learning_rate=1e-6,
                                      )

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(model=model, 
                      tokenizer=tokenizer,
                      data_collator=data_collator,
                      args=training_args,
                      train_dataset=tokenized_ds_train,
                      eval_dataset=tokenized_ds_test,
                      compute_metrics=compute_metrics)

    trainer.train(resume_from_checkpoint=False)

if __name__ == '__main__':
    main()

