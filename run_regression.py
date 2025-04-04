import argparse
import datetime
import multiprocessing as mp
import os
import subprocess
from collections import defaultdict
import pandas as pd
import re


def print_test_results(test_results, layers, output_folder):
    columns = ["Model", "Layer", "Status", "Runtime", "Ideal", "RuntimeType"]
    if len(test_results[0]) == 3:
        columns = columns[:3]

    # convert list of tuples to DataFrame
    df = pd.DataFrame(test_results, columns=columns)
    sorted_df = []

    # get models
    models = df["Model"].unique()

    for model in models:
        print("=" * 10 + f" {model} " + "=" * 10)

        model_df = df[df["Model"] == model]

        # sort according to order in layers
        model_df["Layer"] = pd.Categorical(model_df["Layer"], layers[model])
        model_df.sort_values("Layer", inplace=True)
        # turn categorial back to string
        model_df["Layer"] = model_df["Layer"].astype(str)
        sorted_df.append(model_df)

        passed = model_df[model_df["Status"] == True]
        failed = model_df[model_df["Status"] == False]

        print("Passed:")
        print(
            passed["Layer"].to_string(index=False) if not passed.empty else "None",
            flush=True,
        )
        print("Failed:")
        print(
            failed["Layer"].to_string(index=False) if not failed.empty else "None",
            flush=True,
        )

        # if runtime column exists, print runtime of each layer
        if "Runtime" in model_df.columns:
            print("Runtime:")
            print(
                model_df[["Layer", "Runtime", "Ideal", "RuntimeType"]].to_string(
                    index=False
                ),
                flush=True,
            )
            utilization = model_df["Ideal"].sum() / model_df["Runtime"].sum()
            matrix_runtime = model_df[model_df["RuntimeType"] == "matrix"][
                "Runtime"
            ].sum()
            matrix_ideal = model_df[model_df["RuntimeType"] == "matrix"]["Ideal"].sum()
            matri_utilization = matrix_ideal / matrix_runtime
            print(f"Utilization: {utilization:.3f}")
            print(f"Matrix Utilization: {matri_utilization:.3f}")

    # concatentate all sorted model DataFrames into a single DataFrame and save to pickle
    pd.concat(sorted_df).to_pickle(f"{output_folder}/test_results.pkl")

    # return True if all tests passed
    return len(df[df["Status"] == False]) == 0


def check_environment_vars(required_vars):
    unset_vars = [var for var in required_vars if var not in os.environ]
    if len(unset_vars) > 0:
        raise ValueError(f"Please set {', '.join(unset_vars)} environment variables")


def run_gold_model_unit_test(model, layer, output_folder):
    env_vars = os.environ.copy()
    env_vars["NETWORK"] = model
    env_vars["TESTS"] = layer
    env_vars["CLOCK_PERIOD"] = "1"
    env_vars["SIMS"] = "gold,pytorch"

    with open(f"{output_folder}/{model}_{layer}.log", "w") as stdout_file:
        try:
            subprocess.run(
                ["make", "sim"],
                env=env_vars,
                stdout=stdout_file,
                stderr=subprocess.STDOUT,
                timeout=5 * 60,
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


def run_gold_model_tests(layers, num_processes, results_folder):
    check_environment_vars(["DATATYPE", "IC_DIMENSION", "OC_DIMENSION"])

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
                run_gold_model_unit_test,
                args=(model, test, results_folder),
                callback=test_results.append,
            )

    pool.close()
    pool.join()

    return print_test_results(test_results, layers, results_folder)


def run_systemc_unit_test(model, layer, output_folder, fast):
    env_vars = os.environ.copy()
    env_vars["NETWORK"] = model
    env_vars["TESTS"] = layer
    env_vars["CLOCK_PERIOD"] = "1"
    env_vars["SIMS"] = "gold,accelerator"

    with open(f"{output_folder}/{model}_{layer}.log", "w") as stdout_file:
        try:
            subprocess.run(
                ["make", "fast-sim" if fast else "sim"],
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


def run_systemc_tests(layers, num_processes, results_folder, fast):
    check_environment_vars(["DATATYPE", "IC_DIMENSION", "OC_DIMENSION"])

    # Build TestRunner binary
    subprocess.run(["make", "clean"], env=os.environ)

    with open(f"{results_folder}/build.log", "w") as stdout_file:
        subprocess.run(
            ["make", "-j", "TestRunner-fast" if fast else "TestRunner"],
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
                args=(model, test, results_folder, fast),
                callback=test_results.append,
            )

    pool.close()
    pool.join()

    return print_test_results(test_results, layers, results_folder)


def run_rtl_test(model, layer, output_folder):
    env_vars = os.environ.copy()
    env_vars["NETWORK"] = model
    env_vars["TESTS"] = layer
    env_vars["SIMS"] = "gold,accelerator"
    # Workaround: vcs/catapult don't support GLIBCXX_3.4.30 in their libstdc++, and the tools hardcode the linker libraries in such an
    # order that their libs are used over the user specified ones. We need the newer version in order to run dependencies installed from conda.
    env_vars["LD_PRELOAD"] = env_vars["CONDA_PREFIX"] + "/lib/libstdc++.so.6"
    if "INPUT_BUFFER_SIZE" not in env_vars:
        env_vars["INPUT_BUFFER_SIZE"] = "1024"
    if "WEIGHT_BUFFER_SIZE" not in env_vars:
        env_vars["WEIGHT_BUFFER_SIZE"] = "1024"
    if "ACCUM_BUFFER_SIZE" not in env_vars:
        env_vars["ACCUM_BUFFER_SIZE"] = "1024"

    # we occasionally see the test fail due to filesystem issues ("no rule to make target", but the target exists), so we retry up to 3 times
    for attempt in range(3):
        with open(f"{output_folder}/{model}_{layer}.log", "w") as stdout_file:
            try:
                subprocess.run(
                    ["make", "-f", "scverify/Verify_concat_sim_rtl_v_vcs.mk", "sim"],
                    cwd=f"build/{env_vars['DATATYPE']}_{env_vars['IC_DIMENSION']}x{env_vars['OC_DIMENSION']}/Catapult/{env_vars['TECHNOLOGY']}/clock_{env_vars['CLOCK_PERIOD']}/Accelerator/Accelerator.v1",
                    env=env_vars,
                    stdout=stdout_file,
                    stderr=subprocess.STDOUT,
                    timeout=8 * 60 * 60,
                )
            except subprocess.TimeoutExpired:
                print(f"Test {model}_{layer} timed out")
                stdout_file.write("Test timed out")
                break

        with open(f"{output_folder}/{model}_{layer}.log", "r") as logfile:
            text = logfile.read()
            if "No rule to make target" not in text:
                break

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

        # capture ideal runtime number in the log file
        # can either be "Matrix unit ideal runtime: " or "Vector unit ideal runtime: "
        p = subprocess.Popen(
            [
                "grep",
                "-oP",
                "(?<=matrix unit ideal runtime: ).\d+",
                f"{output_folder}/{model}_{layer}.log",
            ],
            stdout=subprocess.PIPE,
        )

        match = p.communicate()[0]

        if match:
            runtime_type = "matrix"
        else:
            p = subprocess.Popen(
                [
                    "grep",
                    "-oP",
                    "(?<=vector unit ideal runtime: ).\d+",
                    f"{output_folder}/{model}_{layer}.log",
                ],
                stdout=subprocess.PIPE,
            )
            match, error = p.communicate()
            assert not error

            runtime_type = "vector"

        ideal = int(match.decode("utf-8").strip())
    else:
        runtime = 0
        ideal = 0
        runtime_type = ""

    return (model, layer, success, runtime, ideal, runtime_type)


def run_rtl_tests(layers, num_processes, results_folder, keep_build=False):
    check_environment_vars(
        ["DATATYPE", "IC_DIMENSION", "OC_DIMENSION", "TECHNOLOGY", "CLOCK_PERIOD"]
    )

    # clean old build
    if not keep_build:
        subprocess.run(["make", "clean-catapult"], env=os.environ)

    # generate RTL
    with open(f"{results_folder}/rtl_generation.log", "w") as stdout_file:
        subprocess.run(
            ["make", "-j", "rtl"],
            env=os.environ,
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
        )

    model, (test, *_) = next(iter(layers.items()))
    print(f"Running {model} {test}")

    # build VCS simulation binary
    with open(f"{results_folder}/vcs_build.log", "w") as stdout_file:
        env_vars = os.environ.copy()
        env_vars["NETWORK"] = model
        env_vars["TESTS"] = test
        env_vars["SIMS"] = "gold,accelerator"
        env_vars["LD_PRELOAD"] = env_vars["CONDA_PREFIX"] + "/lib/libstdc++.so.6"

        if "INPUT_BUFFER_SIZE" not in env_vars:
            env_vars["INPUT_BUFFER_SIZE"] = "1024"
        if "WEIGHT_BUFFER_SIZE" not in env_vars:
            env_vars["WEIGHT_BUFFER_SIZE"] = "1024"
        if "ACCUM_BUFFER_SIZE" not in env_vars:
            env_vars["ACCUM_BUFFER_SIZE"] = "1024"

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

    return print_test_results(test_results, layers, results_folder)


def run_accuracy(model, dataset, num_processes, output_folder):
    check_environment_vars(["DATATYPE", "IC_DIMENSION", "OC_DIMENSION"])

    if len(model) > 1:
        print(f"Only testing accuracy for the first model: {model[0]}")
    model = model[0]

    env_vars = os.environ.copy()
    env_vars["NETWORK"] = model

    if "INPUT_BUFFER_SIZE" not in env_vars:
        env_vars["INPUT_BUFFER_SIZE"] = "1024"
    if "WEIGHT_BUFFER_SIZE" not in env_vars:
        env_vars["WEIGHT_BUFFER_SIZE"] = "1024"
    if "ACCUM_BUFFER_SIZE" not in env_vars:
        env_vars["ACCUM_BUFFER_SIZE"] = "1024"

    # Build AccuracyTester binary
    subprocess.run(["make", "clean"], env=env_vars)

    with open(f"{output_folder}/build.log", "w") as stdout_file:
        subprocess.run(
            ["make", "-j", "AccuracyTester"],
            env=env_vars,
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
        )

    # Generate input samples from dataset
    if dataset == "imagenet":
        imagenet_path = "/sim2/shared/MINOTAUR/nn_data/imagenet_1000/data/"
        output_data_dir = "data/imagenet"
        subprocess.run(
            [
                "python",
                "test/script/dump_resnet_dataset.py",
                "--data_dir",
                imagenet_path,
                "--output_dir",
                output_data_dir,
                "--num_samples",
                "1000",
            ]
        )
    elif dataset == "sst2":
        output_data_dir = "data/sst2"
        subprocess.run(
            [
                "python",
                "test/script/dump_bert_dataset.py",
                "--dataset",
                "sst2",
                "--model_name_or_path",
                "models/mobilebert/mobilebert-tiny-sst2-bf16/",
                "--output_dir",
                output_data_dir,
            ]
        )
    elif dataset == "squad":
        output_data_dir = "data/squad"
        subprocess.run(
            [
                "python",
                "test/script/dump_bert_dataset.py",
                "--dataset",
                "squad",
                "--model_name_or_path",
                "models/mobilebert/mobilebert-tiny-squad-bf16/",
                "--output_dir",
                output_data_dir,
            ]
        )
    else:
        raise ValueError("Invalid dataset")

    # Dump model parameters
    if model == "resnet18":
        model_path = "models/resnet/resnet18_mp2_p8_qat.pth"
    elif model == "resnet50":
        model_path = "models/resnet/resnet50.pth"
    elif model == "mobilebert" and dataset == "sst2":
        model_path = "models/mobilebert/mobilebert-tiny-sst2-bf16/"
    elif model == "mobilebert" and dataset == "squad":
        model_path = "models/mobilebert/mobilebert-tiny-squad-bf16/"
    else:
        raise ValueError("Invalid model")

    block_size = max(os.environ['OC_DIMENSION'], os.environ['IC_DIMENSION'])

    if env_vars["DATATYPE"] == "E4M3":
        quantization_args = [
            "--activation",
            "fp8_e4m3",
            "--weight",
            "fp8_e4m3",
            "--bf16",
        ]
    elif env_vars["DATATYPE"] == "INT8":
        quantization_args = [
            "--activation",
            "int8,qs=per_tensor_symmetric",
            "--weight",
            "int8,qs=per_tensor_symmetric",
            "--bias",
            "int24",
            "--bf16",
        ]
    elif env_vars["DATATYPE"] == "P8_1":
        quantization_args = [
            "--activation",
            "posit8_1",
            "--weight",
            "posit8_1",
            "--bf16",
        ]
    elif env_vars["DATATYPE"] == "CFLOAT":
        quantization_args = []
    elif env_vars["DATATYPE"] == "MXINT8":
        quantization_args = [
            "--force_scale_power_of_two",
            "--activation",
            "int8,qs=microscaling,bs=" + block_size,
            "--weight",
            "int8,qs=microscaling,bs=" + block_size,
            "--bf16",
        ]
    else:
        raise ValueError("Invalid datatype")

    with open(f"{output_folder}/{model}_{dataset}_compiler.log", "w") as stdout_file:
        subprocess.run(
            [
                "python",
                "quantized-training/test/test_codegen.py",
                model,
                "--model_name_or_path",
                model_path,
                *quantization_args,
                "--output_dir",
                "test/compiler/networks/" + model + "/" + env_vars["DATATYPE"],
            ],
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
        )

    with open(f"{output_folder}/{model}_{dataset}_tiler.log", "w") as stdout_file:
        subprocess.run(
            [
                "python",
                "test/compiler/run_tiler.py",
                "--codegen_dir",
                f"test/compiler/networks/{model}/{env_vars['DATATYPE']}",
                "--IC_dimension",
                env_vars["IC_DIMENSION"],
                "--OC_dimension",
                env_vars["OC_DIMENSION"],
                "--input_buffer_size",
                env_vars["INPUT_BUFFER_SIZE"],
                "--weight_buffer_size",
                env_vars["WEIGHT_BUFFER_SIZE"],
                "--accum_buffer_size",
                env_vars["ACCUM_BUFFER_SIZE"],
            ],
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
        )

    # Run accuracy test
    additional_args = []
    if dataset == "squad":
        additional_args = ["1000"]  # limit number of samples to 1000 for squad dataset
    with open(f"{output_folder}/{model}_{dataset}.log", "w") as stdout_file:
        try:
            subprocess.run(
                [
                    f"build/{env_vars['DATATYPE']}_{env_vars['IC_DIMENSION']}x{env_vars['OC_DIMENSION']}/cc/AccuracyTester",
                    model,
                    output_data_dir,
                    str(num_processes),
                    *additional_args,
                ],
                env=env_vars,
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
    final_accuracy = float(re.findall(accuracy_regex, text)[-1])

    print(f"Final accuracy: {final_accuracy}%")

    # save results to dataframe
    df = pd.DataFrame(
        [(model, dataset, final_accuracy)], columns=["Model", "Dataset", "Accuracy"]
    )

    # dump dataframe to pickle
    df.to_pickle(f"{output_folder}/test_results.pkl")

    if model == "resnet18":
        return final_accuracy >= 70.0
    elif model == "resnet50":
        return final_accuracy >= 60.0
    elif model == "mobilebert" and dataset == "sst2":
        return final_accuracy >= 90.0
    else:
        return True


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
        choices=["gold_model", "systemc", "fast-systemc", "rtl", "accuracy"],
        required=True,
        help="Simulation to run (gold_model, systemc, rtl, accuracy)",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        required=True,
        help="Number of processes to run in parallel",
    )
    parser.add_argument(
        "--tests",
        default=None,
        help="Comma separated list of tests to run (e.g. test1,test2)",
    )
    parser.add_argument(
        "--keep_build",
        action="store_true",
        help="Keep the generated rtl and use it to run rtl tests",
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
    layers = {}
    if args.tests is None:
        for network in args.models:
            env_vars = os.environ.copy()
            env_vars["NETWORK"] = network
            subprocess.run(["make", "network-proto"], env=env_vars)
            with open(
                f"test/compiler/networks/{network}/{os.environ['DATATYPE']}/layers.txt",
                "r",
            ) as f:
                layers[network] = f.read().splitlines()
    else:
        assert (
            len(args.models) == 1
        ), "Only one model can be specified when using --tests"
        layers[args.models[0]] = args.tests.split(",")

    if args.sims == "systemc" or args.sims == "fast-systemc":
        success = run_systemc_tests(
            layers, args.num_processes, results_folder, args.sims == "fast-systemc"
        )
    elif args.sims == "rtl":
        success = run_rtl_tests(
            layers, args.num_processes, results_folder, args.keep_build
        )
    elif args.sims == "gold_model":
        success = run_gold_model_tests(layers, args.num_processes, results_folder)
    elif args.sims == "accuracy":
        success = run_accuracy(args.models, args.dataset, args.num_processes, results_folder)
    else:
        raise ValueError("Invalid simulation type")

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
