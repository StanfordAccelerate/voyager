#!/usr/bin/env python3

import functools
import subprocess
import argparse
import time
import datetime
import sys
import os
import logging

NETWORKS = {
    'mobilebert':
    [
        "bottleneck_input_dense",
        "bottleneck_input_LayerNorm",
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
        "classifier",
    ],
    'mobilebert_activation_gradient': [
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
        "attention_self_value_layer_3",
        "attention_self_value_layer_2",
        "attention_self_value_layer_1",
        "attention_self_value_layer_0",
        "attention_self_attention_probs_0",
        "attention_self_attention_probs_1",
        "attention_self_attention_probs_2",
        "attention_self_attention_probs_3",
        "attention_self_attention_scores_0",
        "attention_self_attention_scores_1",
        "attention_self_attention_scores_2",
        "attention_self_attention_scores_3",
        "attention_self_query_layer_3",
        "attention_self_query_layer_2",
        "attention_self_query_layer_1",
        "attention_self_query_layer_0",
        "attention_self_key_layer_3",
        "attention_self_key_layer_2",
        "attention_self_key_layer_1",
        "attention_self_key_layer_0",
        # "bottleneck_attention_LayerNorm_q",
        # "bottleneck_attention_LayerNorm_k",
        "bottleneck_attention_dense",
        # "shared_attention_input_to_hidden_states",
        # "value_to_hidden_states",
        # "bottlenecked_hidden_states",
    ],
    'mobilebert_weight_gradient': [
        "classifier_weight",
        "classifier_bias",
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
        # "bottleneck_attention_LayerNorm_weight",
        # "bottleneck_attention_LayerNorm_bias",
        # "bottleneck_attention_dense_weight",
        # "bottleneck_attention_dense_bias",
        "bottleneck_input_LayerNorm_weight",
        "bottleneck_input_LayerNorm_bias",
        "bottleneck_input_dense_weight",
        "bottleneck_input_dense_bias",
    ],
    'resnet':
    [
        "conv1",
        "layer1_0_conv1",
        "layer1_0_conv2",
        "layer1_1_conv1",
        "layer1_1_conv2",
        "layer2_0_downsample",
        "layer2_0_conv1",
        "layer2_0_conv2",
        "layer2_1_conv1",
        "layer2_1_conv2",
        "layer3_0_downsample",
        "layer3_0_conv1",
        "layer3_0_conv2",
        "layer3_1_conv1",
        "layer3_1_conv2",
        "layer4_0_downsample",
        "layer4_0_conv1",
        "layer4_0_conv2",
        "layer4_1_conv1",
        "layer4_1_conv2",
        "fc",
    ]
}


def main():
    parser = argparse.ArgumentParser(
        description='MINOTAUR Test Runner. Dispatches and manages tests.')
    parser.add_argument('-sims', '--simulators',
                        type=str,
                        default='fp32,file',
                        help='Simulators to compare (accelerator, customposit, universal, fp32, file) [SIMS].')
    parser.add_argument('--model',
                        type=str,
                        default="resnet",
                        help='Model to run (simple, resnet, mobilebert) [NETWORK].')
    parser.add_argument('--task',
                        type=str,
                        default='forward',
                        help='Operation to run (only applicable to mobilebert) [TASK].')
    parser.add_argument('--tolerance',
                        type=float,
                        default=0.1,
                        help='Relative normalized error in % we allow [TOLERANCE].')
    parser.add_argument('--data_dir',
                        type=str,
                        default=None,
                        help='Path to binary input data [DATA_DIR].')
    parser.add_argument('--output_dir',
                        type=str,
                        default='./test_outputs/',
                        help='Path to output data [OUT_DIR].')
    parser.add_argument('--make_clean',
                        default=False,
                        action='store_true',
                        help='Run make clean before building.')
    parser.add_argument('--use_codegen',
                        default=False,
                        action='store_true',
                        help='Use results from automatic code generation flow.')
    parser.add_argument('--target_name',
                        type=str,
                        default='TestRunner',
                        help='Name of main target (compiled C++ binary).')
    parser.add_argument('--build_dir',
                        type=str,
                        default='build',
                        help='Name of build directory.')
    args = parser.parse_args()

    # Create output directories for both test value and console output
    script_output_dir = os.path.join(args.output_dir, "console_outputs")
    os.makedirs(script_output_dir, exist_ok=True)

    # TODO(fpedd): Move all prints to console to seperate logger, see here https://stackoverflow.com/a/9321890/8130394
    # Setup a logger to log to file
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        filename=f'{args.output_dir}/console_outputs/run_tests.log',
                        filemode='w')

    if args.data_dir is None:
        if args.model == "resnet":
            args.data_dir = './models/resnet/binary_data/'
            # Check if there are files in the binary_data dir, or whether we
            # need to go step deeper in the hierarchy
            sub_dir_info = list(os.walk(args.data_dir))
            assert len(
                sub_dir_info) > 0, f'Directory {args.data_dir} appears to be empty. Run data gen first.'
            if len(sub_dir_info[0][2]) == 0:
                args.data_dir = os.path.join(
                    args.data_dir, sub_dir_info[0][1][0]) + '/'
        elif args.model == "mobilebert":
            args.data_dir = './data/mobilebert_tiny/datafile/step0/'

    # Start timing before executing first "time-consuming" command
    start_time = time.time()

    if args.make_clean:
        subprocess.run(["make", "clean"], check=True)

    # Build SystemC code (running make twice because of linker issues on NFS)
    cmd = ["make", "-j"] + ["BASE_FLAGS='-DUSE_CODEGEN'"] * \
        args.use_codegen + [args.target_name]
    subprocess.run(cmd, check=True)
    subprocess.run(cmd, check=True)

    # Prepare and run all tests/layers simultaneously as different processes
    results = []
    if args.model == "resnet":
        all_tests = NETWORKS[args.model]
    elif args.model == "mobilebert" and args.task == "forward":
        all_tests = NETWORKS["mobilebert"]
    elif args.model == "mobilebert" and args.task == "backward":
        all_tests = NETWORKS["mobilebert_activation_gradient"]
    elif args.model == "mobilebert" and args.task == "gradient":
        all_tests = NETWORKS["mobilebert_weight_gradient"]
    else:
        raise SystemExit(1)

    for test in all_tests:
        file_name = os.path.join(
            script_output_dir, args.model + '_' + test + '.out')
        file = open(file_name, 'w')
        # Set environment variables
        env = os.environ.copy()
        env["NETWORK"] = args.model
        env["TESTS"] = test
        env["SIMS"] = args.simulators
        env["TASK"] = args.task
        env["TOLERANCE"] = str(args.tolerance)
        env["OUT_DIR"] = args.output_dir
        env["DATA_DIR"] = args.data_dir
        # Spawn an new subprocess and grab its name, handle, and output file
        p = subprocess.Popen([os.path.join(args.build_dir, args.target_name)],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        # Enable non-blocking read from process
        os.set_blocking(p.stdout.fileno(), False)
        results.append([args.model + "." + test, p, file])

    # Observe and manage running processes
    last_print_time = 0
    while True:
        running = 0
        failures = 0
        for res in results:
            # Read all lines from proc
            while line := res[1].stdout.readline():
                line = line.decode("utf-8")
                # Write to file
                res[2].write(line)
                # If format is right, print to console once per second
                nums = [int(s) for s in line.split() if s.isdigit()]
                if len(nums) == 3:
                    print("{} -> {} out of {} cycle ({:0.2f}%)".format(
                        res[0], nums[1], nums[2], nums[1]/nums[2]*100.0))
            # Check if proc is still running
            ret = res[1].poll()
            if ret is None:
                running = running + 1
            else:
                # Record number of failures
                failures = failures + bool(res[1].returncode)
                # Close file
                if not res[2].closed:
                    res[2].close()
        curr_start = str(datetime.timedelta(
            seconds=round(time.time()-start_time)))
        if time.time() - last_print_time >= 1.0:
            last_print_time = time.time()
            print("-> {}, Total {}, Running {}, Failed {}".format(
                curr_start, len(results), running, failures))
        # If all are done, exit
        if not running:
            break
        # Free-running while loops are not good
        time.sleep(0.1)

    print("--- Tests done ---")
    print('\n'.join(["{} returned with {}".format(
        res[0], res[1].returncode) for res in results]))
    failures = functools.reduce(
        (lambda acc, res: acc + bool(res[1].returncode)), results, 0)
    print("-> Total {} failed {}".format(len(results), failures))

    # Also log info to file
    logging.info('\n'.join(["{} returned with {}".format(
        res[0], res[1].returncode) for res in results]))
    logging.info("-> Total {} failed {}".format(len(results), failures))

    sys.exit(failures)


if __name__ == "__main__":
    main()
