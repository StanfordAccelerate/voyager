import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, default_data_collator
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

    raw_datasets = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

    def preprocess_function(examples):
        result = tokenizer(examples["sentence"], padding="max_length", max_length=128, truncation=True)
        result["labels"] = examples["label"]
        return result

    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    eval_dataset = tokenized_datasets["validation"]
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=1)

    posit_metric = load_metric("glue", "sst2")
    float_metric = load_metric("glue", "sst2")

    for i, batch in enumerate(tqdm(eval_dataloader)):
        f = open(f"{args.output_dir}/run.{i}.log")
        lines = f.readlines()
        logits0 = lines[2].split()
        logits1 = lines[3].split()

        posit_pred = 1 if float(logits1[0]) > float(logits0[0]) else 0
        float_pred = 1 if float(logits1[1]) > float(logits0[1]) else 0

        posit_metric.add_batch(predictions=[posit_pred], references=batch["labels"])
        float_metric.add_batch(predictions=[float_pred], references=batch["labels"])

    posit_acc = posit_metric.compute()
    float_acc = float_metric.compute()

    print("HLS posit gold model accuracy: ", posit_acc['accuracy'])
    print("Floating-point gold model accuracy: ", float_acc['accuracy'])
