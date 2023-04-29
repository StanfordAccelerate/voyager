from datasets import load_dataset
import os

# Get path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load dataset from the hub
raw_datasets = load_dataset("glue", "sst2", num_proc=8)

# Save dataset to a local disk
raw_datasets.save_to_disk(os.path.join(script_dir, "sst2"))
