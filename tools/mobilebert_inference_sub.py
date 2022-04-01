import argparse
import os
import sys
import struct
import torch
import numpy as np
import subprocess
import multiprocessing as mp
from subprocess import PIPE, STDOUT
from transformers import MobileBertForSequenceClassification, MobileBertConfig, MobileBertTokenizer, get_scheduler
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Reproducibility
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

def write_fp64fd(f, data):
    # data = data.type(torch.float64)
    floatlist = []
       
    for i in np.nditer(data):
        floatlist.append(i)

    buf = struct.pack('%sd' % len(floatlist), *floatlist)
    f.write(buf)

parser = argparse.ArgumentParser(description="Finetune a transformers model on a Text Classification task")
parser.add_argument(
    "--model_name_or_path",
    type=str,
    default="google/mobilebert-uncased",
    help="Path to pretrained model or model identifier from huggingface.co/models.",
)
parser.add_argument(
    "--start",
    type=int,
    help="Start of datasets to test.",
)
parser.add_argument(
    "--end",
    type=int,
    help="End of datasets to test.",
)
parser.add_argument(
    "--id",
    type=int,
    help="ID of the test session.",
)

args = parser.parse_args()

raw_datasets = load_dataset("imdb")

tokenizer = MobileBertTokenizer.from_pretrained(args.model_name_or_path)
model = MobileBertForSequenceClassification.from_pretrained(args.model_name_or_path)

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

def run_inference(start, end, id):
    dataset = small_eval_dataset.select(range(start, end))
    dataloader = DataLoader(dataset, batch_size=1)

    samples = 0.0
    hls_correct = 0.0
    gold_correct = 0.0
    comp_correct = 0.0

    model.eval()
    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        embedding_output = model.mobilebert.embeddings(input_ids=batch["input_ids"], token_type_ids=batch["token_type_ids"])
        embedding_output = embedding_output.detach().numpy().astype(np.float64)

        p = subprocess.Popen(["./build/MobilebertTest"], stdin=PIPE, stdout=PIPE)
        write_fp64fd(p.stdin, embedding_output)

        outs, _ = p.communicate()
        print("PyTorch model output: ", logits)
        print( str(outs, 'utf-8'))
        outs = str(outs, 'utf-8').split()
        hls_index = int(outs[2])
        uni_index = int(outs[-1])
        if hls_index == batch["labels"][0]:
            hls_correct += 1
        if uni_index == batch["labels"][0]:
            gold_correct += 1
        if predictions[0] == batch["labels"][0]:
            comp_correct += 1
        samples += 1

    with open(f'test_outputs/result{id}.txt', 'w') as f:
        f.write("PyTorch model correct predictions: {}/{}\n".format(comp_correct, samples))
        f.write("Gold model correct predictions:    {}/{}\n".format(gold_correct, samples))
        f.write("HLS model correct predictions:     {}/{}\n".format(hls_correct, samples))

if __name__ == '__main__':    
    run_inference(args.start, args.end, args.id)
