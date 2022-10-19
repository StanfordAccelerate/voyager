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
                        default=5.0,
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
        elif args.model == "mobilebert":
            args.data_dir = './data/mobilebert_tiny/datafile/step0/'

    start_time = time.time()

    # Build SystemC code (running make twice because of linker issues when on NFS)
    if args.make_clean:
        subprocess.run(['make', 'clean', args.target_name, '-j'], check=True)
    else:
        subprocess.run(['make', args.target_name, '-j'], check=True)
    subprocess.run(['make', args.target_name, '-j'], check=True)

    # Prepare and run all tests/layers simultaneously as different processes
    results = []
    for test in NETWORKS[args.model]:
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
