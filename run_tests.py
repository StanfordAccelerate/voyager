import functools
import subprocess
import argparse
import time
import datetime
import sys
import os
import logging
import shutil
import numpy as np
import tqdm
import re

# Import handwritten network definitions
from test.mobilebert import mobilebert_networks
from test.resnet import resnet_networks

# Try to import generated network definitions and warn user if they are not found
try:
    from test.resnet import resnet_networks_codegen
except ImportError:
    print("WARNING: Could not find and import resnet_networks_codegen.")
try:
    from test.mobilebert import mobilebert_networks_codegen
except ImportError:
    print("WARNING: Could not find and import mobilebert_networks_codegen.")


def build_code(args):
    """Build the SystemC code with the specified parameters."""

    # Clean build directory
    if args.make_clean:
        subprocess.run(["make", "clean"], check=True)

    # Build SystemC code (running make twice because of linker issues on NFS)
    # We multiply rram_banks (superbanks) by 4 because we have 4 RRAM banks per superbank
    cmd = (
        ["make", "-j"]
        + [
            f"BASE_FLAGS=-DNUM_SRAM_BANKS={args.sram_banks} -DNUM_RRAM_BANKS={args.rram_banks*4}"
        ]
        + [args.target_name]
    )
    subprocess.run(cmd, check=True)
    subprocess.run(cmd, check=True)
    if len(args.all_tests) == 1:
        print("Run this to compile manually:", " ".join(cmd))


def main():
    parser = argparse.ArgumentParser(
        description="MINOTAUR Test Runner. Dispatches and manages tests."
    )
    parser.add_argument(
        "-sims",
        "--simulators",
        type=str,
        default="fp32,file",
        help="Simulators to compare (accelerator, customposit, universal, fp32, file) [SIMS].",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="Model to run (simple, resnet18, mobilebert) [NETWORK].",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="inference",
        help="Operation to run (only applicable to mobilebert) [TASK].",
    )
    parser.add_argument(
        "--test_type",
        type=str,
        default="layerbylayer",
        help="Type of test to run (layerbylayer, end2end, accuracy).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="Relative normalized error in % we allow [TOLERANCE].",
    )
    parser.add_argument(
        "--accuracy_tolerance",
        type=float,
        default=0.60,
        help="Minimum accuracy we expect when running accuracy tests.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,  # None means we will lookup the default data dir below
        help="Path to binary input data [DATA_DIR].",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_outputs/",
        help="Path to output data [OUT_DIR].",
    )
    parser.add_argument(
        "--num_tests",
        type=int,
        default=-1,  # -1 means we will run all available tests
        help="Number of tests from the list of tests to run (-1 -> run all).",
    )
    parser.add_argument(
        "--make_clean",
        default=False,
        action="store_true",
        help="Run make clean before building.",
    )
    parser.add_argument(
        "--target_name",
        type=str,
        default="TestRunner",
        help="Name of main target (compiled C++ binary). See Makefile for details.",
    )
    parser.add_argument(
        "--build_dir", type=str, default="build", help="Name of build directory."
    )
    parser.add_argument(
        "--rram_banks",
        type=int,
        default=3,
        help="Configure the number of RRAM (super)banks.",
    )
    parser.add_argument(
        "--sram_banks", type=int, default=8, help="Configure the number of SRAM banks."
    )
    args = parser.parse_args()

    print(10 * "=" + " MINOTAUR Test Runner " + 10 * "=")

    # Create output directories for both test value and console output
    args.script_output_dir = os.path.join(args.output_dir, "console_outputs")
    os.makedirs(args.script_output_dir, exist_ok=True)

    # TODO(fpedd): Move all prints to console to separate logger, see here https://stackoverflow.com/a/9321890/8130394
    # Setup a logger to log to file
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        filename=f"{args.script_output_dir}/run_tests.log",
        filemode="w",
    )

    if args.data_dir is None:
        if args.model == "resnet18":
            args.data_dir = "./models/resnet/binary_data/"
            # Check if there are files in the binary_data dir, or whether we
            # need to step deeper in the hierarchy
            sub_dir_info = list(os.walk(args.data_dir))
            assert (
                len(sub_dir_info) > 0
            ), f"Directory {args.data_dir} appears to be empty. Run data gen first."
            if len(sub_dir_info[0][2]) == 0:
                args.data_dir = os.path.join(args.data_dir, sub_dir_info[0][1][0]) + "/"
        elif args.model == "mobilebert":
            args.data_dir = "./data/qnli/datafile/"
        else:
            raise ValueError(
                f"Could not find default data_dir for model {args.model}. Please provide data_dir."
            )

    # Check if data_dir exists
    assert os.path.isdir(
        args.data_dir
    ), f"Data dir {args.data_dir} does not exist. Please provide a valid data directory and/or run ZagZig first."

    assert args.num_tests == -1 or args.num_tests > 0, "num_tests must be -1 (all) or > 0."

    # Start timing before executing first "time-consuming" command
    args.start_time = time.time()

    # Prepare all tests/layers to be simultaneously run as individual processes
    args.all_tests = None
    # Default models (with handwritten config)
    if args.model == "resnet18":
        args.all_tests = resnet_networks.NETWORKS[args.model]
    elif args.model == "mobilebert":
        if args.task == "forward_with_weight_splitting":
            name = "mobilebert_inference"
        elif args.task == "backward_with_weight_splitting":
            name = "mobilebert_backward"
        else:
            name = f"mobilebert_{args.task}"

        if name not in mobilebert_networks.NETWORKS.keys():
            raise ValueError(f"{name} not supported on mobilebert.")

        args.all_tests = mobilebert_networks.NETWORKS[name]

    # If we can't find the model in any of the handwritten configs, we look in the codegen
    else:
        if "resnet" in args.model:
            args.all_tests = resnet_networks_codegen.NETWORKS[args.model]
        elif "mobilebert" in args.model:
            args.all_tests = mobilebert_networks_codegen.NETWORKS[args.model]
        else:
            raise ValueError(f"Model {args.model} not supported.")

    # Ensure we found tests for the model
    assert (
        args.all_tests is not None
    ), f"Could not find any tests for model {args.model}."

    # Compile the code
    build_code(args)

    if args.test_type == "layerbylayer":
        # If we only want to run a subset of tests, we truncate the list.
        # Ignore this (run all available tests) if num_tests is -1 (default)
        if args.num_tests > 0:
            assert args.num_tests <= len(
                args.all_tests
            ), f"Requested {args.num_tests} tests, but only {len(args.all_tests)} available."
            args.all_tests = args.all_tests[: args.num_tests]

    elif args.test_type == "end2end":
        assert args.num_tests == -1, "Cannot specify num_tests for end2end tests."
        args.layer_names = args.all_tests  # all_tests will be overwritten
        args.all_tests = ["end2end"]

    elif args.test_type == "accuracy":
        dirpath, dirnames, files = list(os.walk(args.data_dir))[0]
        tensor_names = [f for f in files if f not in ["input", "output"]]
        args.layer_names = args.all_tests  # all_tests will be overwritten
        args.all_tests = dirnames[: args.num_tests if args.num_tests > 0 else None]
        args.root_data_dir = args.data_dir  # data_dir will be overwritten

        # Copy the tensor files to every test directory
        print("Copying tensor files to test directories...")
        for test in tqdm.tqdm(args.all_tests):
            for tensor_name in tensor_names:
                source_tensor_path = os.path.join(dirpath, tensor_name)
                dest_tensor_path = os.path.join(dirpath, test, tensor_name)
                shutil.copyfile(
                    source_tensor_path,
                    dest_tensor_path,
                )

    else:
        raise ValueError(f"Unknown test type {args.test_type}.")

    results = []

    # Launch each test as a separate process
    for test in args.all_tests:
        # Create output file
        file_name = os.path.join(
            args.script_output_dir, args.model + "_" + test + ".out"
        )
        file = open(file_name, "w")

        # Set environment variables
        env = os.environ.copy()
        env["NETWORK"] = args.model
        env["TESTS"] = (
            f"{args.layer_names[0]},{args.layer_names[-1]}"
            if args.test_type in ["accuracy", "end2end"]
            else test
        )
        env["SIMS"] = args.simulators
        env["TASK"] = args.task
        env["TOLERANCE"] = str(args.tolerance)
        env["OUT_DIR"] = (
            os.path.join(args.output_dir, test) + "/"
            if args.test_type == "accuracy"
            else args.output_dir
        )
        if not os.path.exists(env["OUT_DIR"]):
            os.makedirs(env["OUT_DIR"], exist_ok=True)
        env["DATA_DIR"] = (
            os.path.join(dirpath, test) + "/"
            if args.test_type == "accuracy"
            else args.data_dir
        )

        # Set the executable path
        exec_path = os.path.join(args.build_dir, args.target_name)

        # If we only run a single test, print the command for easier reproducibility
        if len(args.all_tests) == 1:
            print("Run this to reproduce:")
            print(
                f"NETWORK={env['NETWORK']} TESTS={env['TESTS']} SIMS={env['SIMS']} TASK={env['TASK']} TOLERANCE={env['TOLERANCE']} OUT_DIR={env['OUT_DIR']} DATA_DIR={env['DATA_DIR']} ./{exec_path}\n"
            )

        # Spawn an new subprocess and grab its name, handle, and output file
        p = subprocess.Popen(
            [exec_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
        )

        # Enable non-blocking read from process
        os.set_blocking(p.stdout.fileno(), False)

        # Add to list of running processes
        results.append([args.model + "." + test, p, file])

    # Observe and manage running processes
    last_print_time = 0
    while True:
        # Reset running and failures
        running = 0
        failures = 0

        # Iterate over all running processes
        for res in results:
            # Read all lines from running process
            while line := res[1].stdout.readline():
                line = line.decode("utf-8")

                # Write line from process to output file
                res[2].write(line)

                # If format maches, print progress
                nums = [int(s) for s in line.split() if s.isdigit()]

                # Print progress only if we have 3 numbers and the second and third are not 0
                if len(nums) == 3 and nums[1] != 0 and nums[2] != 0:
                    print(
                        "{} -> {} out of {} cycle ({:0.2f}%)".format(
                            res[0], nums[1], nums[2], nums[1] / nums[2] * 100.0
                        )
                    )

            # Check if the process is done
            ret = res[1].poll()
            if ret is None:
                running = running + 1
            else:
                # Record number of failures
                failures = failures + bool(res[1].returncode)
                # And close the output file
                if not res[2].closed:
                    res[2].close()

        # Print progress every second
        if time.time() - last_print_time >= 1.0:
            last_print_time = time.time()
            running_time = str(
                datetime.timedelta(seconds=round(time.time() - args.start_time))
            )
            print(
                "-> {}, Total {}, Running {}, Failed {}".format(
                    running_time, len(results), running, failures
                )
            )

        # If all processes are done, break
        if not running:
            break
        # Free-running while loops are not good
        time.sleep(0.1)

    # Print a summary of the results
    left_justify = max([len(res[0]) for res in results])
    print("--- Tests done ---")
    print(
        "\n".join(
            [
                "{:<{x}} {y:>3}  {z}".format(
                    res[0],
                    x=left_justify,
                    y=res[1].returncode,
                    z="\u2714" if res[1].returncode == 0 else "\u2718",
                )
                for res in results
            ]
        )
    )
    failures = functools.reduce(
        (lambda acc, res: acc + bool(res[1].returncode)), results, 0
    )

    # Print a green check mark or a red cross depending on whether the tests passed or failed
    if failures == 0:
        print(
            "\u001b[32m"
            + 10 * "="
            + " Total {} Failed {} ".format(len(results), failures)
            + 10 * "="
            + "\u001b[0m"
        )
    else:
        print(
            "\u001b[31m"
            + 10 * "x"
            + " Total {} Failed {} ".format(len(results), failures)
            + 10 * "x"
            + " \u001b[0m"
        )

    # Also log info to logfile
    logging.info(
        "\n".join(
            [
                "{} returned with {} \t {}".format(
                    res[0],
                    res[1].returncode,
                    "\u2714" if res[1].returncode == 0 else "\u2718",
                )
                for res in results
            ]
        )
    )
    logging.info(
        10 * "=" + "Total {} failed {}".format(len(results), failures) + 10 * "="
    )

    # If we are running accuracy tests, parse the results and print them
    if args.test_type == "accuracy":
        x_corr = 0
        y_corr = 0

        print("Parsing accuracy results...")
        for test in tqdm.tqdm(args.all_tests):
            # Get the result directory
            res_dir = os.path.join(args.output_dir, test)

            # Get the result file path
            res_files = [os.path.join(x[0], x[2][0]) for x in os.walk(res_dir)]

            # Ensure there is only one result file
            assert (
                l := len(res_files)
            ) == 1, f"More than one result file in dir {res_dir}: {l} {res_files}"
            res_file = res_files[0]

            # Parse the result file
            with open(res_file, "r") as f:
                # Read the file into a buffer
                buf = f.readlines()

                # Parse the names from the first line
                names = buf[0].split(" vs. ")
                assert (
                    len(names) == 2
                ), f"Could not locate two names in head {buf[0]} of file {res_file}."

                x_name = names[0]
                y_name = names[1][:-1]  # remove trailing newline

                # Remove names from buffer
                buf = buf[1:]

                # Match the numbers left and right of 'vs.'
                left = [
                    re.findall(r"[-+]?(?:\d*\.\d+|\d+) vs.", s)[0][:-4] for s in buf
                ]
                right = [
                    re.findall(r"vs. [-+]?(?:\d*\.\d+|\d+)", s)[0][4:] for s in buf
                ]

                # Ensure the lengths match
                assert len(left) == len(
                    right
                ), f"Left {len(left)} != right {len(right)}"

                # ImageNet has 1000 classes
                if len(left) == 1008:
                    left = left[:-8]  # remove last 8 zero lines
                    assert (
                        l := len(left)
                    ) == 1000, f"ResNet output should be 1000 entries, but is {l}."
                    right = right[:-8]  # remove last 8 zero lines
                    assert (
                        l := len(right)
                    ) == 1000, f"ResNet output should be 1000 entries, but is {l}."

                # SST2 has 2 classes
                elif len(left) == 16:
                    left = left[:-14]  # remove last 14 zero lines
                    assert (
                        l := len(left)
                    ) == 2, f"ResNet output should be 2 entries, but is {l}."
                    right = right[:-14]  # remove last 14 zero lines
                    assert (
                        l := len(right)
                    ) == 2, f"ResNet output should be 2 entries, but is {l}."
                else:
                    raise RuntimeError(
                        f"Unknown results in {res_file} with {len(left)} entries."
                    )

                # Convert strings to actualy numbers
                x = [float(x) for x in left]
                y = [float(x) for x in right]

                # Find top category and compare with ref labels
                top_x = np.argmax(x)
                top_y = np.argmax(y)
                ref_label = int(re.findall(".[0-9]+_", res_file)[0][1:-1])

                # Check if the top category matches the reference label
                if ref_label == top_x:
                    x_corr += 1
                if ref_label == top_y:
                    y_corr += 1

        # Compute accuracy
        x_acc = x_corr / len(results)
        y_acc = y_corr / len(results)

        # Print accuracy results to console
        print(
            f"Network {args.model} -> {x_name} accuracy: {x_acc} ({x_corr}/{len(results)}), {y_name} accuracy: {y_acc} ({y_corr}/{len(results)})"
        )

        # Also log info to file
        logging.info(
            f"Network {args.model} -> {x_name} accuracy: {x_acc} ({x_corr}/{len(results)}), {y_name} accuracy: {y_acc} ({y_corr}/{len(results)})"
        )

        # We expect both to match or exceed the accuracy tolerance
        if x_acc < args.accuracy_tolerance or y_acc < args.accuracy_tolerance:
            print(f"Did not match expected accuracy of {args.accuracy_tolerance}.")
            failures += 1

    # Return the number of failed tests as exit code so that CI can detect failures
    sys.exit(failures)


if __name__ == "__main__":
    main()
