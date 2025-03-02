import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import DataCollatorWithPadding
from transformers import Trainer
import pandas as pd
import wandb
wandb.login(key="")#不登陆remote log无法显示，kaggle会进入死循环
model =AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-base',num_labels=2)
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
train = load_dataset("glue", "sst2", split="train")
test = load_dataset("glue", "sst2", split="test")
val = load_dataset("glue", "sst2", split="validation")
train = train.map(lambda e: tokenizer(e['sentence'],max_length=300,truncation=True, padding='max_length'), batched=True,remove_columns=['sentence'])
test = test.map(lambda e: tokenizer(e['sentence'],max_length=300,truncation=True, padding='max_length'), batched=True,remove_columns=['sentence'])
val = val.map(lambda e: tokenizer(e['sentence'],max_length=300,truncation=True, padding='max_length'), batched=True,remove_columns=['sentence'])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
def compute_metrics(eval_pred):
    predictions,labels = eval_pred
    predictions = predictions.argmax(-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=500,
    evaluation_strategy='steps',
    eval_steps=500,
    save_strategy="no",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train.select(range(64)),##修改此处即直接适配各个data_size
    eval_dataset=val,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)
trainer.train()
predictions_output = trainer.predict(test)
predictions = predictions_output.predictions.argmax(-1)
labels = predictions_output.label_ids
outputs = pd.DataFrame({'idx': range(len(predictions)), 'pred': predictions, 'label': labels})

outputs.to_csv('./results.csv', index=False, quoting=3)