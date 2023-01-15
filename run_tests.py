import functools
import subprocess
import argparse
import time
import datetime
import sys
import os
import logging

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


def main():
    parser = argparse.ArgumentParser(
        description="MINOTAUR Test Runner. Dispatches and manages tests.")
    parser.add_argument("-sims", "--simulators",
                        type=str,
                        default="fp32,file",
                        help="Simulators to compare (accelerator, customposit, universal, fp32, file) [SIMS].")
    parser.add_argument("--model",
                        type=str,
                        default="resnet",
                        help="Model to run (simple, resnet, mobilebert) [NETWORK].")
    parser.add_argument("--task",
                        type=str,
                        default="inference",
                        help="Operation to run (only applicable to mobilebert) [TASK].")
    parser.add_argument("--tolerance",
                        type=float,
                        default=0.1,
                        help="Relative normalized error in % we allow [TOLERANCE].")
    parser.add_argument("--data_dir",
                        type=str,
                        default=None, # None means we will lookup the default data dir below
                        help="Path to binary input data [DATA_DIR].")
    parser.add_argument("--output_dir",
                        type=str,
                        default="./test_outputs/",
                        help="Path to output data [OUT_DIR].")
    parser.add_argument("--make_clean",
                        default=False,
                        action="store_true",
                        help="Run make clean before building.")
    parser.add_argument("--target_name",
                        type=str,
                        default="TestRunner",
                        help="Name of main target (compiled C++ binary).")
    parser.add_argument("--build_dir",
                        type=str,
                        default="build",
                        help="Name of build directory.")
    args = parser.parse_args()

    # Create output directories for both test value and console output
    script_output_dir = os.path.join(args.output_dir, "console_outputs")
    os.makedirs(script_output_dir, exist_ok=True)

    # TODO(fpedd): Move all prints to console to seperate logger, see here https://stackoverflow.com/a/9321890/8130394
    # Setup a logger to log to file
    logging.basicConfig(level=logging.INFO,
                        format="%(message)s",
                        filename=f"{args.output_dir}/console_outputs/run_tests.log",
                        filemode='w')

    if args.data_dir is None:
        if args.model == "resnet":
            args.data_dir = "./models/resnet/binary_data/"
            # Check if there are files in the binary_data dir, or whether we
            # need to step deeper in the hierarchy
            sub_dir_info = list(os.walk(args.data_dir))
            assert len(
                sub_dir_info) > 0, f"Directory {args.data_dir} appears to be empty. Run data gen first."
            if len(sub_dir_info[0][2]) == 0:
                args.data_dir = os.path.join(
                    args.data_dir, sub_dir_info[0][1][0]) + '/'
        elif args.model == "mobilebert":
            args.data_dir = "./data/qnli/datafile/"
        else:
            raise ValueError(
                f"Could not find default data_dir for model {args.model}. Please provide data_dir.")

    # Start timing before executing first "time-consuming" command
    start_time = time.time()

    if args.make_clean:
        subprocess.run(["make", "clean"], check=True)

    # Build SystemC code (running make twice because of linker issues on NFS)
    cmd = ["make", "-j"] + [args.target_name]
    subprocess.run(cmd, check=True)
    subprocess.run(cmd, check=True)

    # Prepare all tests/layers to be simultaneously run as individual processes
    results = []
    all_tests = None
    # Default models (with handwritten config)
    if args.model == "resnet":
        all_tests = resnet_networks.NETWORKS[args.model]
    elif args.model == "mobilebert":
        name = f'{args.model}_{args.task}'
        if name not in mobilebert_networks.NETWORKS.keys():
            raise ValueError(f"{name} not supported on mobilebert.")
        all_tests = mobilebert_networks.NETWORKS[name]
    # If we can't find the model in any of the handwritten configs, we look in the codegen
    else:
        if "resnet" in args.model:
            all_tests = resnet_networks_codegen.NETWORKS[args.model]
        elif "mobilebert" in args.model:
            all_tests = mobilebert_networks_codegen.NETWORKS[args.model]
        else:
            raise ValueError(f"Model {args.model} not supported.")

    # Ensure we found tests for the model
    assert all_tests is not None, f"Could not find any tests for model {args.model}."

    # Launch each test as a separate process
    for test in all_tests:
        file_name = os.path.join(
            script_output_dir, args.model + '_' + test + ".out")
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
        exec_path = os.path.join(args.build_dir, args.target_name)
        if len(all_tests) == 1:  # If we only run a single test, print the command for easier reproducibility
            print("Run this to reproduce:")
            print(f"NETWORK={env['NETWORK']} TESTS={env['TESTS']} SIMS={env['SIMS']} TASK={env['TASK']} TOLERANCE={env['TOLERANCE']} OUT_DIR={env['OUT_DIR']} DATA_DIR={env['DATA_DIR']} ./{exec_path}\n")
        # Spawn an new subprocess and grab its name, handle, and output file
        p = subprocess.Popen(
            [exec_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        # Enable non-blocking read from process
        os.set_blocking(p.stdout.fileno(), False)
        results.append([args.model + '.' + test, p, file])

    # Observe and manage running processes
    last_print_time = 0
    while True:
        running = 0
        failures = 0
        for res in results:
            # Read all lines from proc
            while line := res[1].stdout.readline():
                line = line.decode("utf-8")
                res[2].write(line)
                # If format is right, print to console once per second
                nums = [int(s) for s in line.split() if s.isdigit()]
                if len(nums) == 3 and nums[2] != 0:
                    print("{} -> {} out of {} cycle ({:0.2f}%)".format(
                        res[0], nums[1], nums[2], nums[1]/nums[2]*100.0))
            # Check if proc is still running
            ret = res[1].poll()
            if ret is None:
                running = running + 1
            else:
                # Record number of failures
                failures = failures + bool(res[1].returncode)
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

    # Also log info to logfile
    logging.info('\n'.join(["{} returned with {}".format(
        res[0], res[1].returncode) for res in results]))
    logging.info("-> Total {} failed {}".format(len(results), failures))

    sys.exit(failures)


if __name__ == "__main__":
    main()
