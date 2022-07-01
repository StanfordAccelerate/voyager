import argparse
import torch
import numpy as np
from transformers import MobileBertForSequenceClassification, MobileBertConfig, MobileBertTokenizer
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to output logs.",
    )
    args = parser.parse_args()

    raw_datasets = load_dataset("imdb")

    tokenizer = MobileBertTokenizer.from_pretrained("models/mobilebert")
    model = MobileBertForSequenceClassification.from_pretrained("models/mobilebert")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    full_train_dataset = tokenized_datasets["train"]
    full_eval_dataset = tokenized_datasets["test"]

    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=1)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=1)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    posit_metric = load_metric("accuracy")
    float_metric = load_metric("accuracy")

    for i, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}

        f = open(f"{args.output_dir}/run.{i}.log")
        lines = f.readlines()
        logits0 = lines[2].split()
        logits1 = lines[3].split()

        posit_pred = 1 if float(logits1[0]) > float(logits0[0]) else 0
        float_pred = 1 if float(logits1[1]) > float(logits0[1]) else 0

        posit_metric.add_batch(predictions=[posit_pred], references=batch["labels"])
        float_metric.add_batch(predictions=[float_pred], references=batch["labels"])

    posit_acc = posit_metric.compute()['accuracy']
    float_acc = float_metric.compute()['accuracy']

    print("HLS posit gold model accuracy: ", posit_acc)
    print("Floating-point gold model accuracy: ", float_acc)
