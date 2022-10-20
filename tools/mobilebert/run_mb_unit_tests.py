import datetime
import multiprocessing as mp
import os
import struct
import subprocess

import argparse
import numpy as np

inference_tests = [
    "bottleneck_input_dense",
    "bottleneck_input_LayerNorm",
    "bottleneck_attention_dense",
    "bottleneck_attention_LayerNorm",
    "attention_self_query_layer",
    "attention_self_key_layer",
    "attention_self_value_layer",
    "attention_self_attention_scores_0",
    "attention_self_attention_scores_1",
    "attention_self_attention_scores_2",
    "attention_self_attention_scores_3",
    "attention_self_attention_probs_0",
    "attention_self_attention_probs_1",
    "attention_self_attention_probs_2",
    "attention_self_attention_probs_3",
    "attention_self_context_layer_0",
    "attention_self_context_layer_1",
    "attention_self_context_layer_2",
    "attention_self_context_layer_3",
    "attention_output_dense",
    "attention_output_LayerNorm",
    "ffn_0_intermediate_dense",
    "ffn_0_output_dense",
    "ffn_0_output_LayerNorm",
    "intermediate_dense",
    "output_dense",
    "output_LayerNorm",
    "output_bottleneck_dense",
    "output_bottleneck_LayerNorm",
    "classifier"
]

backprop_tests = [
    "classifier",
    "output_bottleneck_LayerNorm",
    "output_bottleneck_dense",
    "output_LayerNorm",
    "output_dense",
    "intermediate_dense",
    "ffn_0_output_LayerNorm",
    "ffn_0_output_dense",
    "ffn_0_intermediate_dense",
    "attention_output_LayerNorm",
    "attention_output_dense",
    "bottleneck_input_dense",
    "attention_self_context_layer",
    "attention_self_value_layer_0",
    "attention_self_value_layer_1",
    "attention_self_value_layer_2",
    "attention_self_value_layer_3",
    "attention_self_attention_probs_0",
    "attention_self_attention_probs_1",
    "attention_self_attention_probs_2",
    "attention_self_attention_probs_3",
    "attention_self_attention_scores_0",
    "attention_self_attention_scores_1",
    "attention_self_attention_scores_2",
    "attention_self_attention_scores_3",
    "attention_self_query_layer_0",
    "attention_self_query_layer_1",
    "attention_self_query_layer_2",
    "attention_self_query_layer_3",
    "attention_self_key_layer_0",
    "attention_self_key_layer_1",
    "attention_self_key_layer_2",
    "attention_self_key_layer_3",
    "bottleneck_attention_dense",
]

gradient_tests = [
    "classifier_weight",
    "output_bottleneck_LayerNorm_weight",
    "output_bottleneck_LayerNorm_bias",
    "output_bottleneck_dense_weight",
    "output_bottleneck_dense_bias",
    "output_LayerNorm_weight",
    "output_LayerNorm_bias",
    "output_dense_weight",
    "output_dense_bias",
    "intermediate_dense_weight",
    "intermediate_dense_bias",
    "ffn_0_output_LayerNorm_weight",
    "ffn_0_output_LayerNorm_bias",
    "ffn_0_output_dense_weight",
    "ffn_0_output_dense_bias",
    "ffn_0_intermediate_dense_weight",
    "ffn_0_intermediate_dense_bias",
    "attention_output_LayerNorm_weight",
    "attention_output_LayerNorm_bias",
    "attention_output_dense_weight",
    "attention_output_dense_bias",
    "attention_self_value_weight",
    "attention_self_value_bias",
    "attention_self_query_weight",
    "attention_self_query_bias",
    "attention_self_key_weight",
    "attention_self_key_bias",
    "bottleneck_attention_dense_weight",
    "bottleneck_attention_dense_bias",
    "bottleneck_attention_LayerNorm_weight",
    "bottleneck_attention_LayerNorm_bias",
    "bottleneck_input_dense_weight",
    "bottleneck_input_dense_bias",
    "bottleneck_input_LayerNorm_weight",
    "bottleneck_input_LayerNorm_bias"
]

def run_test(results_folder, groups, task, test):
    env_vars = os.environ.copy()
    env_vars['NETWORK'] = "mobilebert"
    env_vars['SIMS'] = groups
    env_vars['TASK'] = task
    env_vars['TESTS'] = test

    stdout_file = open(f'{results_folder}/{test}.log', 'w')
    stderr_file = open(f'{results_folder}/{test}.out', 'w')

    sim_process = subprocess.Popen(
        './build/TestRunner',
        env=env_vars,
        stdout=stdout_file,
        stderr=stderr_file
    )

    sim_process.communicate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Text Classification task")
    parser.add_argument(
        "--datapath",
        type=str,
        default=None,
        help="Path to MobileBERT test data.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="forward",
        help="Group of tests to run.",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default="customposit,universal,fp32,accelerator",
        help="Models to run the tests.",
    )
    args = parser.parse_args()

    # Create directory with current time
    results_folder = 'logs/' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    os.makedirs(results_folder)

    # Create pool of multiprocessors
    pool = mp.Pool(32)

    tests = None
    if args.task == "forward":
        tests = inference_tests
    elif args.task == "backward":
        tests = backprop_tests
    elif args.task == "gradient":
        tests = gradient_tests
    else:
        raise SystemError("Unknown task: ", task)

    for test in tests:
        pool.apply_async(run_test, (results_folder, args.groups, args.task, test))

    pool.close()
    pool.join()
