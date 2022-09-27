import argparse
import json
import os
import random

import torch
import torch.fx
from typing import Any, Callable, Dict, Optional, Tuple
from torch.fx.node import Node
import numpy as np
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PretrainedConfig,
    default_data_collator,
    get_scheduler
)
from tqdm.auto import tqdm

from modeling_mobilebert import MobileBertForSequenceClassification


class ModulePathTracer(torch.fx.Tracer):
    """
    ModulePathTracer is an FX tracer that--for each operation--also records
    the qualified name of the Module from which the operation originated.
    """

    # The current qualified name of the Module being traced. The top-level
    # module is signified by empty string. This is updated when entering
    # call_module and restored when exiting call_module
    current_module_qualified_name : str = ''
    # A map from FX Node to the qualname of the Module from which it
    # originated. This is recorded by `create_proxy` when recording an
    # operation
    node_to_originating_module : Dict[torch.fx.Node, str] = {}

    def call_module(self, m: torch.nn.Module, forward: Callable[..., Any],
                    args : Tuple[Any, ...], kwargs : Dict[str, Any]) -> Any:
        """
        Override of Tracer.call_module (see
        https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer.call_module).
        This override:
        1) Stores away the qualified name of the caller for restoration later
        2) Installs the qualified name of the caller in `current_module_qualified_name`
           for retrieval by `create_proxy`
        3) Delegates into the normal Tracer.call_module method
        4) Restores the caller's qualified name into current_module_qualified_name
        """
        old_qualname = self.current_module_qualified_name
        try:
            self.current_module_qualified_name = self.path_of_module(m)
            return super().call_module(m, forward, args, kwargs)
        finally:
            self.current_module_qualified_name = old_qualname

    def create_proxy(self, kind: str, target: torch.fx.node.Target, args: Tuple[Any, ...],
                     kwargs: Dict[str, Any], name: Optional[str] = None, type_expr: Optional[Any] = None):
        """
        Override of `Tracer.create_proxy`. This override intercepts the recording
        of every operation and stores away the current traced module's qualified
        name in `node_to_originating_module`
        """
        proxy = super().create_proxy(kind, target, args, kwargs, name, type_expr)
        self.node_to_originating_module[proxy.node] = self.current_module_qualified_name
        return proxy


# Reproducibility
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description="Finetune MobileBERT on IMDb Text Classification task.")
parser.add_argument(
    "--model_name_or_path",
    type=str,
    default="google/mobilebert-uncased",
    help="Path to pretrained model or model identifier from huggingface.co/models.",
)
args = parser.parse_args()


def main():
    raw_datasets = load_dataset("glue", "sst2")

    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = MobileBertForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels)

    model.eval()

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
    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    optimizer = SGD(model.parameters(), lr=2e-2)

    # def forward_hook(name):
    #     def hook(module, x, y):
    #         print(f'{name}: {[tuple(i.shape) for i in x]} -> {list(y.shape)}')
    #     return hook

    # for name, module in model.named_modules():
    #     if (len(list(module.children())) == 0):
    #         module.register_forward_hook(forward_hook(name))

    batch = {k: torch.unsqueeze(v, 0).to(device) for k, v in eval_dataset[0].items()}

    # with torch.autograd.profiler.profile(record_shapes=True, with_modules=True) as prof:
    #     outputs = model(**batch)

    # print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total"))

    # from torch.fx import symbolic_trace
    # from transformers.utils.fx import symbolic_trace
    # gm : torch.fx.GraphModule = symbolic_trace(model)

    tracer = ModulePathTracer()
    traced_model = tracer.trace(model)

    # Print (node, module qualified name) for every node in the Graph
    for node in traced_model.nodes:
        module_qualname = tracer.node_to_originating_module.get(node)
        print('Node', node, 'is from module', module_qualname)

if __name__ == "__main__":
    main()
