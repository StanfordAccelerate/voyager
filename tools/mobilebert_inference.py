import multiprocessing as mp
import os
import struct
import subprocess
from subprocess import PIPE

import argparse
from datasets import load_dataset, load_metric
import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import MobileBertForSequenceClassification, MobileBertConfig, MobileBertTokenizer, get_scheduler

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

def run_test(model_name_or_path, datapath, batch, results_folder, test_id):
    model = MobileBertForSequenceClassification.from_pretrained(model_name_or_path)
    model.eval()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    embedding_output = model.mobilebert.embeddings(input_ids=batch["input_ids"], token_type_ids=batch["token_type_ids"])
    embedding_output = embedding_output.detach().numpy().astype(np.float64)

    env_vars = os.environ.copy()
    env_vars['GROUP'] = "mobilebert"
    env_vars['TESTS'] = "inference"
    env_vars['SIMS'] = "fp32,hlsposit"
    env_vars['DATA_PATH'] = datapath

    stdout_file = open(f'{results_folder}/run.{test_id}.log', 'w')
    stderr_file = open(f'{results_folder}/run.{test_id}.out', 'w')

    sim_process = subprocess.Popen(
        './build/TestRunner',
        env=env_vars,
        stdin=PIPE,
        stdout=stdout_file,
        stderr=stderr_file
    )
    write_fp64fd(sim_process.stdin, embedding_output)

    sim_process.communicate()


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Text Classification task")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="google/mobilebert-uncased",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="data/mobilebert/datafile/",
        help="Path to dumped MobileBERT data.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index of test groups.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=1000,
        help="End index of test groups.",
    )
    args = parser.parse_args()

    raw_datasets = load_dataset("imdb")

    tokenizer = MobileBertTokenizer.from_pretrained(args.model_name_or_path)

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

    # Create directory with current time
    results_folder = 'logs/' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    os.makedirs(results_folder)

    # Create pool of multiprocessors
    pool = mp.Pool(32)

    dataset = small_eval_dataset.select(range(args.start, args.end))
    dataloader = DataLoader(dataset, batch_size=1)

    for pid, batch in enumerate(tqdm(dataloader)):
        pool.apply_async(run_test, (args.model_name_or_path, args.datapath, batch, results_folder, pid))

    pool.close()
    pool.join()
