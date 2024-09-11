import argparse
import datetime
import multiprocessing as mp
import os
import subprocess
from collections import defaultdict
import pandas as pd
import re

def print_test_results(test_results, layers):
    results_by_model = defaultdict(list)
    for result in test_results:
        results_by_model[result[0]].append(result)

    all_tests_passed = True
    for model, results in results_by_model.items():
        # Sort results by layer order
        results.sort(key=lambda x: layers[model].index(x[1]))
        passed = [r for r in results if r[2] == True]
        failed = [r for r in results if r[2] == False]

        if len(failed) > 0:
            all_tests_passed = False

        print(f"Model: {model}")
        print("=" * 10 + " PASSED " + "=" * 10)
        print("\n".join(r[1] for r in passed) if passed else "None")
        print("=" * 10 + " FAILED " + "=" * 10)
        print("\n".join(r[1] for r in failed) if failed else "None")

        if len(results[0]) == 3:
            continue

        print("Runtime:")
        for result in results:
            print(f"{result[1]}: {result[3]}")

    return all_tests_passed


def check_environment_vars(required_vars):
    unset_vars = [var for var in required_vars if var not in os.environ]
    if len(unset_vars) > 0:
        raise ValueError(f"Please set {', '.join(unset_vars)} environment variables")


def run_fp32_unit_test(model, layer, output_folder):
    env_vars = os.environ.copy()
    env_vars["NETWORK"] = model
    env_vars["TESTS"] = layer
    env_vars["CLOCK_PERIOD"] = "1"
    env_vars["SIMS"] = "fp32,file"

    with open(f"{output_folder}/{model}_{layer}.log", "w") as stdout_file:
        try:
            subprocess.run(
                ["make", "sim"],
                env=env_vars,
                stdout=stdout_file,
                stderr=subprocess.STDOUT,
                timeout=1 * 30,
            )
        except subprocess.TimeoutExpired:
            print(f"Test {model}_{layer} timed out")
            stdout_file.write("Test timed out")

    # search if the test passed
    p = subprocess.Popen(
        ["grep", "Error count: 0", f"{output_folder}/{model}_{layer}.log"],
        stdout=subprocess.PIPE,
    )
    p.communicate()

    return (model, layer, p.returncode == 0)


def run_fp32_tests(layers, num_processes, results_folder):
    check_environment_vars(["IC_DIMENSION", "OC_DIMENSION"])

    # Build TestRunner binary
    # subprocess.run(["make", "clean"], env=env_vars)

    with open(f"{results_folder}/build.log", "w") as stdout_file:
        subprocess.run(
            ["make", "-j", "TestRunner"],
            env=os.environ,
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
        )

    pool = mp.Pool(num_processes)

    test_results = []

    for model, tests in layers.items():
        for test in tests:
            pool.apply_async(
                run_fp32_unit_test,
                args=(model, test, results_folder),
                callback=test_results.append,
            )

    pool.close()
    pool.join()

    return print_test_results(test_results, layers)


def run_systemc_unit_test(model, layer, output_folder):
    env_vars = os.environ.copy()
    env_vars["NETWORK"] = model
    env_vars["TESTS"] = layer
    env_vars["CLOCK_PERIOD"] = "1"
    env_vars["SIMS"] = "systemc,accelerator"

    with open(f"{output_folder}/{model}_{layer}.log", "w") as stdout_file:
        try:
            subprocess.run(
                ["make", "sim"],
                env=env_vars,
                stdout=stdout_file,
                stderr=subprocess.STDOUT,
                timeout=1 * 60 * 60,
            )
        except subprocess.TimeoutExpired:
            print(f"Test {model}_{layer} timed out")
            stdout_file.write("Test timed out")

    # search if the test passed
    p = subprocess.Popen(
        ["grep", "Error count: 0", f"{output_folder}/{model}_{layer}.log"],
        stdout=subprocess.PIPE,
    )
    p.communicate()

    return (model, layer, p.returncode == 0)


def run_systemc_tests(layers, num_processes, results_folder):
    check_environment_vars(["DATATYPE", "IC_DIMENSION", "OC_DIMENSION"])

    # Build TestRunner binary
    # subprocess.run(["make", "clean"], env=os.environ)

    with open(f"{results_folder}/build.log", "w") as stdout_file:
        subprocess.run(
            ["make", "-j", "TestRunner"],
            env=os.environ,
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
        )

    pool = mp.Pool(num_processes)

    test_results = []

    for model, tests in layers.items():
        for test in tests:
            pool.apply_async(
                run_systemc_unit_test,
                args=(model, test, results_folder),
                callback=test_results.append,
            )

    pool.close()
    pool.join()

    return print_test_results(test_results, layers)

def run_rtl_test(model, layer, output_folder):
    env_vars = os.environ.copy()
    env_vars["NETWORK"] = model
    env_vars["TESTS"] = layer
    env_vars["SIMS"] = "systemc,accelerator"
    # Workaround: vcs/catapult don't support GLIBCXX_3.4.30 in their libstdc++, and the tools hardcode the linker libraries in such an 
    # order that their libs are used over the user specified ones. We need the newer version in order to run dependencies installed from conda.
    env_vars["LD_PRELOAD"] = env_vars["CONDA_PREFIX"] + "/lib/libstdc++.so.6"

    with open(f"{output_folder}/{model}_{layer}.log", "w") as stdout_file:
        try:
            subprocess.run(
                ["make", "-f", "scverify/Verify_concat_sim_rtl_v_vcs.mk", "sim"],
                cwd=f"build/{env_vars['DATATYPE']}_{env_vars['IC_DIMENSION']}x{env_vars['OC_DIMENSION']}/Catapult/{env_vars['TECHNOLOGY']}/clock_{env_vars['CLOCK_PERIOD']}/Accelerator/Accelerator.v1",
                env=env_vars,
                stdout=stdout_file,
                stderr=subprocess.STDOUT,
                timeout=90 * 60,
            )
        except subprocess.TimeoutExpired:
            print(f"Test {model}_{layer} timed out")
            stdout_file.write("Test timed out")

    # search if the test passed
    p = subprocess.Popen(
        ["grep", "Error count: 0", f"{output_folder}/{model}_{layer}.log"],
        stdout=subprocess.PIPE,
    )
    p.communicate()
    success = p.returncode == 0

    if success:
        # capture number after "Runtime: " in the log file
        p = subprocess.Popen(
            [
                "grep",
                "-oP",
                "(?<=Runtime: ).\d+",
                f"{output_folder}/{model}_{layer}.log",
            ],
            stdout=subprocess.PIPE,
        )
        runtime = int(p.communicate()[0].decode("utf-8").strip())

        # capture number after "Ideal cycles: " in the log file
        p = subprocess.Popen(
            [
                "grep",
                "-oP",
                "(?<=Ideal cycles: ).\d+",
                f"{output_folder}/{model}_{layer}.log",
            ],
            stdout=subprocess.PIPE,
        )
        ideal = int(p.communicate()[0].decode("utf-8").strip())
    else:
        runtime = 0
        ideal = 0

    return (model, layer, success, runtime, ideal)


def run_rtl_tests(layers, num_processes, results_folder):
    check_environment_vars(
        ["DATATYPE", "IC_DIMENSION", "OC_DIMENSION", "TECHNOLOGY", "CLOCK_PERIOD"]
    )

    # clean old build
    subprocess.run(["make", "clean-catapult"], env=os.environ)

    # generate RTL
    with open(f"{results_folder}/rtl_generation.log", "w") as stdout_file:
        subprocess.run(
            ["make", "-j", "rtl"],
            env=os.environ,
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
        )

    # build VCS simulation binary
    with open(f"{results_folder}/vcs_build.log", "w") as stdout_file:
        env_vars = os.environ.copy()
        env_vars["NETWORK"] = "resnet18"
        env_vars["TESTS"] = "submodule_0"
        env_vars["SIMS"] = "systemc,accelerator"
        env_vars["LD_PRELOAD"] = env_vars["CONDA_PREFIX"] + "/lib/libstdc++.so.6"

        subprocess.run(
            ["make", "-f", "scverify/Verify_concat_sim_rtl_v_vcs.mk", "build"],
            cwd=f"build/{env_vars['DATATYPE']}_{env_vars['IC_DIMENSION']}x{env_vars['OC_DIMENSION']}/Catapult/{env_vars['TECHNOLOGY']}/clock_{env_vars['CLOCK_PERIOD']}/Accelerator/Accelerator.v1",
            env=env_vars,
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
        )

    pool = mp.Pool(num_processes)

    test_results = []

    for model, tests in layers.items():
        for test in tests:
            pool.apply_async(
                run_rtl_test,
                args=(model, test, results_folder),
                callback=test_results.append,
            )

    pool.close()
    pool.join()

    return print_test_results(test_results, layers)

def run_accuracy(model, dataset, num_processes, output_folder):
    check_environment_vars(["DATATYPE", "IC_DIMENSION", "OC_DIMENSION"])

    if len(model) > 1:
        print(f"Only testing accuracy for the first model: {model[0]}")
    model = model[0]

    # Build AccuracyTester binary
    subprocess.run(["make", "clean"], env=os.environ)

    with open(f"{output_folder}/build.log", "w") as stdout_file:
        subprocess.run(
            ["make", "-j", "AccuracyTester"],
            env=os.environ,
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
        )

    # Generate input samples from dataset
    if dataset == "imagenet":
        imagenet_path = "/sim2/shared/MINOTAUR/nn_data/imagenet_1000/data/"
        output_data_dir = "data/imagenet"
        subprocess.run(["python3", "test/script/dump_resnet_dataset.py", "--data_dir", imagenet_path, "--output_dir", output_data_dir, "--num_samples", "1000"])
    elif dataset == "sst2":
        output_data_dir = "data/sst2"
        subprocess.run(["python3", "test/script/dump_bert_dataset.py", "--dataset", "sst2", "--model_name_or_path", "models/mobilebert/mobilebert-tiny-sst2-bf16/", "--output_dir", output_data_dir])
    elif dataset == "squad":
        output_data_dir = "data/squad"
        subprocess.run(["python3", "test/script/dump_bert_dataset.py", "--dataset", "squad", "--model_name_or_path", "models/mobilebert/mobilebert-tiny-squad-bf16/", "--output_dir", output_data_dir])
    else:
        raise ValueError("Invalid dataset")

    # Run accuracy test
    additional_args = []
    if dataset == "squad":
        additional_args = ["--num_samples", "1000"]
    with open(f"{output_folder}/{model}_{dataset}.log", "w") as stdout_file:
        env_vars = os.environ.copy()

        try:
            subprocess.run(
                [
                    f"build/{env_vars['DATATYPE']}_{env_vars['IC_DIMENSION']}x{env_vars['OC_DIMENSION']}/cc/AccuracyTester",
                    model,
                    output_data_dir,
                    str(num_processes),
                    *additional_args,
                ],
                env=os.environ,
                stdout=stdout_file,
                stderr=subprocess.STDOUT,
                timeout=2 * 60 * 60,
            )
        except subprocess.TimeoutExpired:
            print(f"Test {model}_{dataset} timed out")
            stdout_file.write("Test timed out")
            return False

    # Extract accuracy from log file
    accuracy_regex = "Accuracy: \d+\/\d+ \((\d+\.+\d+)%\)"
    with open(f"{output_folder}/{model}_{dataset}.log", "r") as logfile:
        text = logfile.read()
    final_accuracy = re.findall(accuracy_regex, text)[-1]

    print(f"Final accuracy: {final_accuracy}%")

    # save results to dataframe
    df = pd.DataFrame(
        [(model, dataset, final_accuracy)], columns=["Model", "Dataset", "Accuracy"]
    )

    # dump dataframe to pickle
    df.to_pickle(f"{output_folder}/test_results.pkl")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        required=True,
        help="Model(s) to test for regression (resnet18, mobilebert)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="Dataset to use for accuracy test (imagenet, sst2)",
    )
    parser.add_argument(
        "--sims",
        choices=["fp32", "systemc", "rtl", "accuracy"],
        required=True,
        help="Simulation to run (fp32, systemc, rtl, accuracy)",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        required=True,
        help="Number of processes to run in parallel",
    )
    args = parser.parse_args()

    args.models = [s.strip() for s in args.models.split(",")]

    # Create directory with current time
    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    results_folder = "regression_results/" + current_time
    os.makedirs(results_folder)
    # create softlink to latest results (delete old if exists)
    os.system("rm -f regression_results/latest")
    os.system(f"cd regression_results && ln -sf {current_time} latest")

    # Add codegen layers
    subprocess.run(["make", "protos"])

    layers = {}
    for network in args.models:
        with open(f"test/compiler/networks/{network}/layers.txt", "r") as f:
            layers[network] = f.read().splitlines()

    if args.sims == "systemc":
        success = run_systemc_tests(layers, args.num_processes, results_folder)
    elif args.sims == "rtl":
        success = run_rtl_tests(layers, args.num_processes, results_folder)
    elif args.sims == "fp32":
        success = run_fp32_tests(layers, args.num_processes, results_folder)
    elif args.sims == "accuracy":
        run_accuracy(args.models, args.dataset, args.num_processes, results_folder)
        success = True
    else:
        raise ValueError("Invalid simulation type")

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
