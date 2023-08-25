#!/usr/bin/env python3
import argparse
import os
from os import listdir
import struct
import sys
import re

import dill as pickle
import numpy as np
import torch

from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
    default_data_collator
)

''' @brief: Writes data of form torch.tensor dtype=float64 to binary data. '''
def write_fp64(filename, data):
    data = data.astype(np.float64)
    with open(filename, 'wb') as f:
        floatlist = []
        
        for i in np.nditer(data):
            floatlist.append(i)

        buf = struct.pack('%sd' % len(floatlist), *floatlist)
        f.write(buf)

def write_model_params(model, data_dir):
    parameters = {n: p.data.detach().float().cpu().numpy().T for n, p in model.named_parameters() if p.requires_grad}
    parameters["classifier.weight"] = np.concatenate([parameters["classifier.weight"].T, np.zeros((14, 512))])
    parameters["classifier.bias"] = np.concatenate([parameters["classifier.bias"], np.full(14, np.finfo(np.float32).min)])

    if not os.path.exists(os.path.join(data_dir, "params")):
        os.makedirs(os.path.join(data_dir, "params"))

    for name, param in parameters.items():
        outfile = os.path.join(data_dir, "params", name.replace('.', '_'))
        write_fp64(outfile, param.flatten())
        print(outfile, end="\r", flush=True)
    print("=" * 120)

def write_dataset(model, model_dir, data_dir):
    'NOTE: only supports SST-2 for now'

    raw_datasets = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

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

    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    for idx, batch in enumerate(tqdm(eval_dataloader, desc="Running through eval dataset")):
        inputs = {k: v.to(device) for k, v in batch.items()}

        embedding_output = model.mobilebert.embeddings(input_ids=inputs["input_ids"], token_type_ids=inputs["token_type_ids"])
        embedding_output = embedding_output.detach().numpy().astype(np.float64)
        attention_mask = (1.0 - inputs["attention_mask"]).cpu().numpy() * np.finfo(np.float32).min
        labels = inputs["labels"].detach().numpy().astype(np.float64)

        folder = os.path.join(data_dir, f"{idx}_{int(labels[0])}")
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # write embedding output and attention mask
        write_fp64(os.path.join(folder, "embedding_output"), embedding_output)
        write_fp64(os.path.join(folder, "attention_mask"), attention_mask)


def export_to_onnx(model, model_dir, output_onnx_path):
    
    # Create dummy inputs for the model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    dummy_sentence = "This is a sample sentence for ONNX export."
    tokens = tokenizer(dummy_sentence, return_tensors="pt")
    input_ids = tokens["input_ids"]
    token_type_ids = tokens["token_type_ids"]
    
    # Move model to the appropriate device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    # Set the model to evaluation mode
    model.eval()

    # # create two dummy inputs for the model with shapes (1, 128, 512) and (1, 128)
    input_ids = torch.ones((1, 128, 512), dtype=torch.float32)
    token_type_ids = torch.ones((1, 128), dtype=torch.float32)

    # # access only the encoder part of the model
    # model = model.mobilebert.encoder
    model = model

    # Export the model to ONNX
    torch.onnx.export(
        model,
        (input_ids.to(device), token_type_ids.to(device)),
        output_onnx_path,
        verbose=True,
        opset_version=11,
        input_names=['input', 'masks'],
        output_names=['output']
    )

    print(f"Model exported to {output_onnx_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate datafiles from model.")
    parser.add_argument("--model_dir", type=str, help="Path to model directory.", default=None, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to output datafiles.",
        required=True
    )
    parser.add_argument("--dump_model", action="store_true", help="Dump model parameters.")
    parser.add_argument("--dump_dataset", action="store_true", help="Dump dataset.")
    args = parser.parse_args()

    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, ignore_mismatched_sizes=True)

    if args.dump_model:
        write_model_params(model, args.output_dir)
    if args.dump_dataset:
        write_dataset(model, args.model_dir, args.output_dir)
    
    # export_to_onnx(model, args.model_dir, "model.onnx")
    

if __name__ == "__main__":
    main()
