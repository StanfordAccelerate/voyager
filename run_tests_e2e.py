#!/usr/bin/env python3

import functools
import subprocess
import argparse
import time
import datetime
import sys
import os
import re
import logging
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='MINOTAUR Test Runner. Dispatches and manages end2end tests.')
    parser.add_argument('-sims', '--simulators',
                        type=str,
                        default='fp32,file',
                        help='Simulators to compare (accelerator, customposit, universal, fp32, file) [SIMS].')
    parser.add_argument('--model',
                        type=str,
                        default="resnet",
                        help='Model to run (simple, resnet, mobilebert) [NETWORK].')
    parser.add_argument('--tolerance',
                        type=float,
                        default=0.1,
                        help='Relative normalized error in % we allow [TOLERANCE].')
    parser.add_argument('--accuracy_tolerance',
                        type=float,
                        default=0.65,
                        help='Minimum accuracy we expect on the provided dataset.')
    parser.add_argument('--data_dir',
                        type=str,
                        default=None,
                        help='Path to binary input data [DATA_DIR].')
    parser.add_argument('--output_dir',
                        type=str,
                        default='./test_outputs_e2e/',
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
    parser.add_argument('--labels_file',
                        type=str,
                        default='data/imagenet/labels.txt',
                        help='Path to file containing labels.')
    parser.add_argument('--samples',
                        type=int,
                        default=100,
                        help='Relative normalized error in % we allow [TOLERANCE].')
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

    dirpath, dirnames, _ = list(os.walk(args.data_dir))[0]

    assert len(
        dirnames) >= args.samples, f'Can not generate {args.samples}, only {len(dirnames)} samples available.'

    # Prepare and run all e2e tests simultaneously as different processes
    results = []
    for dirname in dirnames[:args.samples]:

        file_name = os.path.join(
            script_output_dir, args.model + '_' + dirname + '.out')
        file = open(file_name, 'w')
        # Set environment variables
        env = os.environ.copy()
        env["NETWORK"] = args.model
        env["TESTS"] = 'conv1,fc'
        env["SIMS"] = args.simulators
        env["TASK"] = args.task
        env["TOLERANCE"] = str(args.tolerance)
        env["OUT_DIR"] = os.path.join(args.output_dir, dirname)+'/'
        os.makedirs(env["OUT_DIR"], exist_ok=True)
        env["DATA_DIR"] = os.path.join(dirpath, dirname)+'/'
        # Spawn an new subprocess and grab its name, handle, and output file
        p = subprocess.Popen([os.path.join(args.build_dir, args.target_name)],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        # Enable non-blocking read from process
        os.set_blocking(p.stdout.fileno(), False)
        results.append([args.model + "." + dirname, p, file, env["OUT_DIR"]])

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

    print(f"--- Evaluating accuracy on {len(results)} samples ---")

    # Get the reference labels
    ref_labels = []
    with open(args.labels_file, "r") as f:
        ref_labels = [s.strip().split(',') for s in f.readlines()]

    x_corr = 0
    y_corr = 0
    for res in results:
        res_files = [x[0]+x[2][0] for x in os.walk(res[3])]
        assert (l := len(res_files)) == 1, f'More than one result file in dir.'
        res_file = res_files[0]
        with open(res_file, "r") as f:
            buf = f.readlines()
            # Get names
            names = buf[0].split(" vs. ")
            assert len(
                names) == 2, f"Could not locate two names in head {buf[0]} of file {res_file}."
            x_name = names[0]
            y_name = names[1][:-1]  # remove trailing newline
            # Remove names from buffer
            buf = buf[1:]
            # Match the numbers left and right of 'vs.'
            left = [re.findall(r"[-+]?(?:\d*\.\d+|\d+) vs.", s)[0][:-4]
                    for s in buf]
            right = [re.findall(r"vs. [-+]?(?:\d*\.\d+|\d+)", s)[0][4:]
                     for s in buf]
            left = left[:-8]  # remove last 8 zero lines
            assert (
                l := len(left)) == 1000, f'ResNet output should be 1000 entries, but is {l}.'
            right = right[:-8]  # remove last 8 zero lines
            assert (
                l := len(right)) == 1000, f'ResNet output should be 1000 entries, but is {l}.'
            # Convert strings to actualy numbers
            x = [float(x) for x in left]
            y = [float(x) for x in right]
            # Find top category and compare with ref labels
            top_x = np.argmax(x)
            top_y = np.argmax(y)
            image_label = re.findall('n[0-9]+', res_file)[0]
            if image_label == ref_labels[top_x][0]:
                x_corr += 1
            if image_label == ref_labels[top_y][0]:
                y_corr += 1
    x_acc = x_corr / len(results)
    y_acc = y_corr / len(results)
    print(f"Network {args.model} -> {x_name} accuracy: {x_acc} ({x_corr}/{len(results)}), {y_name} accuracy: {y_acc} ({y_corr}/{len(results)})")

    # Also log info to file
    logging.info(f"Network {args.model} -> {x_name} accuracy: {x_acc} ({x_corr}/{len(results)}), {y_name} accuracy: {y_acc} ({y_corr}/{len(results)})")

    # We expect both to match or exceed the accuracy tolerance
    if x_acc < args.accuracy_tolerance or y_acc < args.accuracy_tolerance:
        print(f"Did not match expected accuracy of {args.accuracy_tolerance}.")
        failures += 1

    sys.exit(failures)


if __name__ == "__main__":
    main()
