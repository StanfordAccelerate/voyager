#!/usr/bin/env python3

import functools
import subprocess
import argparse
import time
import datetime
import sys
import os

MODELS = {
    # 'mobilebert':
    # [
    #     # 'inputBottleneck',
    #     # 'qkvProjection',
    #     # 'qkAttention',
    #     # 'vAttention',
    #     # 'wProjection',
    #     # 'ffn1',
    #     # 'ffn2',
    #     # 'outputBottleneck'
    # ],
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
        "layer4_1_conv2",  # TODO(fpedd): Still failing...
        "fc",
    ]
}

# TODO(fpedd): Implement e2e runs by generating list of layers


def main():
    parser = argparse.ArgumentParser(
        description='MINOTAUR Test Runner. Dispatches and manages tests.')
    parser.add_argument('-sims', '--simulators',
                        type=str,
                        default='fp32,file',
                        help='Simulators to compare (accelerator, customposit, universal, fp32, file) [SIMS].')
    # TODO(fpedd): Implement commandline args and tests (should overwrite values from MODELS dict)
    # parser.add_argument('-mod', '--model',
    #                     type=str,
    #                     default=None,
    #                     help='Model to run (simple, resnet, mobilebert) [MODEL].')
    # parser.add_argument('-tst', '--tests',
    #                     type=str,
    #                     default=None,
    #                     help='Tests/layers to run (if "e2e", end-to-end will be run) [TESTS].')
    parser.add_argument('-tsk', '--task',
                        type=str,
                        default='forward',
                        help='Operation to run (only applicable to mobilebert) [TASK].')
    parser.add_argument('-dd', '--data_dir',
                        type=str,
                        default='./models/resnet/binary_data/',
                        help='Path to binary input data [DATA_DIR].')
    parser.add_argument('-od', '--output_dir',
                        type=str,
                        default='./test_outputs/',
                        help='Path to output data [OUT_DIR].')
    parser.add_argument('-nmc', '--no_make_clean',
                        default=False,
                        action='store_true',
                        help='Do not run make clean before building.')
    parser.add_argument('-tn', '--target_name',
                        type=str,
                        default='TestRunner',
                        help='Name of main target.')
    parser.add_argument('-bd', '--build_dir',
                        type=str,
                        default='build',
                        help='Name of build directory.')
    args = parser.parse_args()

    start_time = time.time()

    # Build SystemC code (running make twice because of linker issues when on NFS)
    if args.no_make_clean:
        subprocess.run(['make', args.target_name, '-j'])
    else:
        subprocess.run(['make', 'clean', args.target_name, '-j'])
    subprocess.run(['make', args.target_name, '-j'])

    # Create output directories for both test value and console output
    script_output_dir = os.path.join(args.output_dir, "console_outputs")
    os.makedirs(script_output_dir, exist_ok=True)

    # Prepare and run all tests/layers simultaneously as different processes
    results = []
    for model in MODELS:
        for test in MODELS[model]:
            file_name = os.path.join(
                script_output_dir, model + '_' + test + '.out')
            file = open(file_name, 'w')
            # Set environment variables
            env = os.environ.copy()
            env["MODEL"] = model
            env["TESTS"] = test
            env["SIMS"] = args.simulators
            env["DATA_DIR"] = args.data_dir
            env["OUT_DIR"] = args.output_dir
            # Spawn an new subprocess and grab its name, handle, and output file
            p = subprocess.Popen([os.path.join(args.build_dir, args.target_name)],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
            # Enable non-blocking read from process
            os.set_blocking(p.stdout.fileno(), False)
            results.append([model + "." + test, p, file])

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

    print("--- Simulation run done ---")
    print('\n'.join(["{} returned with {}".format(
        res[0], res[1].returncode) for res in results]))
    failures = functools.reduce(
        (lambda acc, res: acc + bool(res[1].returncode)), results, 0)
    print("-> Total {} failed {}".format(len(results), failures))

    sys.exit(failures)


if __name__ == "__main__":
    main()
