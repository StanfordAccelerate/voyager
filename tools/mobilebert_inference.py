import datetime
import multiprocessing as mp
import os
import struct
import subprocess
from subprocess import PIPE

import argparse
from datasets import load_dataset, load_metric

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
    default_data_collator
)

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
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    model.eval()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    batch = {k: v.to(device) for k, v in batch.items()}

    embedding_output = model.mobilebert.embeddings(input_ids=batch["input_ids"], token_type_ids=batch["token_type_ids"])
    embedding_output = embedding_output.detach().numpy().astype(np.float64)
    attention_mask = batch["attention_mask"].detach().numpy().astype(np.float64)

    env_vars = os.environ.copy()
    env_vars['GROUP'] = "mobilebert"
    env_vars['SIMS'] = "fp32,customposit"
    env_vars['TESTS'] = "forward"
    env_vars['DATA'] = datapath

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
    write_fp64fd(sim_process.stdin, attention_mask)

    sim_process.communicate()


if __name__ == "__main__":    
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
    args = parser.parse_args()

    raw_datasets = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

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

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=1
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=1)

    # Create directory with current time
    results_folder = 'logs/' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    os.makedirs(results_folder)

    # Create pool of multiprocessors
    pool = mp.Pool(32)

    for pid, batch in enumerate(tqdm(eval_dataloader)):
        pool.apply_async(run_test, (args.model_name_or_path, args.datapath, batch, results_folder, pid))

    pool.close()
    pool.join()
