# We have GELU as an example, model: BERT
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, default_data_collator
from datasets import load_dataset
from tqdm import tqdm

class PiecewiseGELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.min_input = None
        self.max_input = None

    def forward(self, x):
        # Track min/max input range
        x_min, x_max = x.min().item(), x.max().item()
        self.min_input = x_min if self.min_input is None else min(self.min_input, x_min)
        self.max_input = x_max if self.max_input is None else max(self.max_input, x_max)

        out = torch.zeros_like(x)

        # Region masks
        mask1 = (x >= -2.5) & (x < -0.5)
        mask2 = (x >= -0.5) & (x < 0)
        mask3 = (x >= 0) & (x < 1)
        mask4 = (x >= 1) & (x <= 2)
        mask5 = x > 2

        x1 = x[mask1]
        x2 = x[mask2]
        x3 = x[mask3]
        x4 = x[mask4]
        x5 = x[mask5]

        # Region [-2.5, -0.5]
        out[mask1] = (
            -0.07092205 +
            0.30166385 * x1**1 +
            0.27434127 * x1**2 +
            0.05831527 * x1**3
        )
        # Region [-0.5, 0]
        out[mask2] = (
            0.00004909 +
            0.50208532 * x2 +
            0.41831297 * x2**2 +
            0.06245931 * x2**3
        )
        # Region [0, 1]
        out[mask3] = (
            0.00051412 +
            0.48847301 * x3 +
            0.45659731 * x3**2 +
           -0.10387582 * x3**3
        )
        # Region [1, 2]
        out[mask4] = (
           -0.11732868 +
            0.76605086 * x4 +
            0.24949524 * x4**2 +
           -0.05733813 * x4**3
        )
        # Region x > 2
        out[mask5] = x5
        # Region x < -2.5 already zero (initialized)
        return out


def main():
    # Use fine-tuned BERT for SST-2 from Hugging Face
    model_name_or_path = "textattack/bert-base-uncased-SST-2"
    task_name = "sst2"
    batch_size = 32

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        
    #gelu_trackers = []

    for i, layer in enumerate(model.bert.encoder.layer):
        custom_gelu = PiecewiseGELU()
        setattr(layer.intermediate, "intermediate_act_fn", custom_gelu)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model.eval()

    # Load and preprocess dataset
    raw_datasets = load_dataset("glue", task_name)
    sentence1_key = "sentence"
    sentence2_key = None

    def preprocess_function(examples):
        texts = (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        result = tokenizer(*texts, padding="max_length", max_length=128, truncation=True)
        result["labels"] = examples["label"]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
        desc="Running tokenizer on dataset",
    )

    eval_dataset = processed_datasets["validation"]
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size)

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Run inference and compute accuracy
    correct = 0
    total = 0

    for i, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
        print(f"\n Processing batch {i}")
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            labels = batch["labels"].to(device)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"\n PyTorch BERT accuracy on SST-2: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
