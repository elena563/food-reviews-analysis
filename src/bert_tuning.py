from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
import evaluate
import torch
torch.set_num_threads(4)

# pretrained bert model and tokenizer loading
model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3 # reduced to three to achieve a reasonable performance level (positive, negative, neutral)
)

# dataset loading and splitting
dataset = load_dataset('csv', data_files='../data/Reviews_for_tuning.csv')['train']
dataset = dataset.class_encode_column("label")
split_dataset = dataset.train_test_split(test_size=0.1, seed=42, stratify_by_column="label")    # stratify for imbalanced dataset
train_set = split_dataset['train']
test_set = split_dataset['test']
train_set.to_csv("train_data.csv", index=False)
test_set.to_csv("test_data.csv", index=False)

def tokenize_function(batch): return tokenizer(batch["text"], padding="longest", truncation=True, max_length=64)

training_args = TrainingArguments(
  output_dir="./results",
  num_train_epochs=2,
  per_device_train_batch_size=8,
  per_device_eval_batch_size=8,
  learning_rate=3e-5,
  weight_decay=0.01,
  eval_strategy="epoch",
  save_strategy="no",
  load_best_model_at_end=False,
  metric_for_best_model="f1",
  save_total_limit=1,
  dataloader_num_workers=0
)

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")
    precision = precision_metric.compute(predictions=preds, references=labels, average="macro")
    recall = recall_metric.compute(predictions=preds, references=labels, average="macro")

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "precision": precision["precision"],
        "recall": recall["recall"]
    }

# sampled dataset for faster training
sampled_train_set = train_set.shuffle(seed=42).select(range(5000))
sampled_eval_set = test_set.shuffle(seed=42).select(range(1000))

sampled_train_set = sampled_train_set.map(tokenize_function, batched=True)
sampled_eval_set = sampled_eval_set.map(tokenize_function, batched=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=sampled_train_set,
    eval_dataset=sampled_eval_set,
    compute_metrics=compute_metrics,
)
trainer.train()

model_path = "./models/bert-reviews-tuned"

trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

trainer.save_state()