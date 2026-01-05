import argparse
import datetime
import json
import math
import multiprocessing as mp
import numpy as np
import os
import subprocess
import pandas as pd
import re
import signal
import sys
from deepdiff import DeepDiff

from google.protobuf import text_format
from google.protobuf.json_format import MessageToDict
from voyager_compiler.codegen import param_pb2

ACCURACY_RESULTS = {
    "resnet18": {
        "E4M3": 69.3,
        "CFLOAT": 70.8,
        "INT8": 69.5,
        "MXINT8": 70.3,
        "P8_1": 69.7,
    },
    "resnet50": {
        "E4M3": 79.5,
        "CFLOAT": 81.2,
        "INT8": 78.6,
        "MXINT8": 81.2,
        "P8_1": 80.2,
    },
    "mobilebert": {
        "E4M3": 90.94,
        "CFLOAT": 90.83,
        "INT8": 90.6,
        "MXINT8": 91.28,
        "P8_1": 90.02,
    },
    "bert": {
        "E4M3": 93.0,
        "CFLOAT": 92.89,
        "INT8": 92.32,
        "MXINT8": 93.0,
        "P8_1": 92.78,
    },
    "vit": {
        "E4M3": 84.5,
        "CFLOAT": 84.7,
        "INT8": 77.1,
        "MXINT8": 84.6,
        "P8_1": 84.0,
    },
    "mobilenet_v2": {
        "E4M3":  62.5,
        "CFLOAT": 70.0,
        "INT8": 66.0,
        "MXINT8": 72.1,
        "P8_1": 63.8,
    },
}


def set_default_env_vars(env_vars):
    env_vars.setdefault("INPUT_BUFFER_SIZE", "1024")
    env_vars.setdefault("WEIGHT_BUFFER_SIZE", "1024")
    env_vars.setdefault("ACCUM_BUFFER_SIZE", "1024")
    env_vars.setdefault("DOUBLE_BUFFERED_ACCUM_BUFFER", "false")
    env_vars.setdefault("SUPPORT_MVM", "false")
    env_vars.setdefault("SUPPORT_SPMM", "false")


def get_build_folder(env_vars):
    return (
        f"build/"
        f"{env_vars['DATATYPE']}_"
        f"{env_vars['IC_DIMENSION']}x{env_vars['OC_DIMENSION']}_"
        f"{env_vars['INPUT_BUFFER_SIZE']}x{env_vars['WEIGHT_BUFFER_SIZE']}x{env_vars['ACCUM_BUFFER_SIZE']}_"
        f"{env_vars['DOUBLE_BUFFERED_ACCUM_BUFFER']}_"
        f"{env_vars['SUPPORT_MVM']}_"
        f"{env_vars['SUPPORT_SPMM']}"
    )


def utilization(df):
    count = df["Count"].to_numpy()
    full_tiles = df["L2 Tiles"].to_numpy()
    actual_tiles = df["Actual Tiles"].to_numpy()
    ideal = df["Ideal"].to_numpy()
    runtime = df["Runtime"].to_numpy()

    # Precompute common factor
    weight = (full_tiles * count) / actual_tiles

    numerator = np.sum(ideal * weight)
    denominator = np.sum(runtime * weight)

    return numerator / denominator if denominator != 0 else np.nan


def print_test_results(test_results, layers, output_folder):
    columns = [
        "Model",
        "Layer",
        "Status",
        "Runtime",
        "Ideal",
        "RuntimeType",
        "Count",
        "L2 Tiles",
        "Actual Tiles",
    ]
    if len(test_results[0]) == 3:
        columns = columns[:3]

    # convert list of tuples to DataFrame
    df = pd.DataFrame(test_results, columns=columns)
    sorted_df = []

    # get models
    models = df["Model"].unique()

    for model in models:
        print("=" * 10 + f" {model} " + "=" * 10)

        # Create an explicit copy of the DataFrame
        model_df = df[df["Model"] == model].copy()

        # sort according to order in layers
        model_df["Layer"]= pd.Categorical(model_df["Layer"], layers[model])
        model_df.sort_values("Layer", inplace=True)
        # turn categorial back to string
        model_df["Layer"] = model_df["Layer"].astype(str)
        sorted_df.append(model_df)

        passed = model_df[model_df["Status"] == True]
        failed = model_df[model_df["Status"] == False]

        print("Passed:")
        print(passed["Layer"].to_string(index=False) if not passed.empty else "None")
        print("Failed:")
        print(failed["Layer"].to_string(index=False) if not failed.empty else "None")

        # if runtime column exists, print runtime of each layer
        if "Runtime" in model_df.columns:
            print("Runtime:")
            print(model_df[columns[1:]].to_string(index=False), flush=True)

            utilization_all    = utilization(model_df)
            utilization_matrix = utilization(model_df[model_df["RuntimeType"] == "matrix"])

            print(f"Utilization: {utilization_all:.3f}")
            print(f"Matrix Utilization: {utilization_matrix:.3f}")

    # concatentate all DataFrames into a single DataFrame and save to pickle and excel
    combined_df = pd.concat(sorted_df)
    combined_df.to_pickle(f"{output_folder}/test_results.pkl")
    combined_df.to_excel(f"{output_folder}/test_results.xlsx", index=False)

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

    def signal_handler(signum, frame):
        print(f"Receive signal {signum}, terminating pool...")
        pool.terminate()
        pool.join()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

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
                timeout=4 * 60 * 60,
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

    def signal_handler(signum, frame):
        print(f"Receive signal {signum}, terminating pool...")
        pool.terminate()
        pool.join()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

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


def run_rtl_test(model, layer, layer_count, num_tiles, output_folder, debug):
    env_vars = os.environ.copy()
    env_vars["NETWORK"] = model
    env_vars["TESTS"] = layer
    env_vars["SIMS"] = "gold,accelerator"

    if debug:
        env_vars["SIM_DUMP_FSDB"] = "1"

    # Workaround: vcs/catapult don't support GLIBCXX_3.4.30 in their libstdc++,
    # and the tools hardcode the linker libraries in such an order that their
    # libs are used over the user specified ones. We need the newer version in
    # order to run dependencies installed from conda.
    env_vars["LD_PRELOAD"] = env_vars["CONDA_PREFIX"] + "/lib/libstdc++.so.6"
    set_default_env_vars(env_vars)
    build_folder = get_build_folder(env_vars)

    # we occasionally see the test fail due to filesystem issues ("no rule to
    # make target", but the target exists), so we retry up to 3 times
    for attempt in range(3):
        with open(f"{output_folder}/{model}_{layer}.log", "w") as stdout_file:
            try:
                subprocess.run(
                    ["make", "-f", "scverify/Verify_concat_sim_rtl_v_vcs.mk", "sim"],
                    cwd=f"{build_folder}/Catapult/{env_vars['TECHNOLOGY']}/clock_{env_vars['CLOCK_PERIOD']}/Accelerator/Accelerator.v1",
                    env=env_vars,
                    stdout=stdout_file,
                    stderr=subprocess.STDOUT,
                    timeout=10 * 60 * 60,
                )
            except subprocess.TimeoutExpired:
                print(f"Test {model}_{layer} timed out")
                stdout_file.write("Test timed out")
                break

        with open(f"{output_folder}/{model}_{layer}.log", "r") as logfile:
            text = logfile.read()
            if "No rule to make target" not in text:
                break

    log_file = f"{output_folder}/{model}_{layer}.log"

    success = False
    total_runtime = 0
    ideal_runtime = 0
    runtime_type = ""

    if os.path.exists(log_file):
        with open(log_file, "r") as file:
            content = file.read()

        # Check if "Error count: 0" exists
        success = bool(re.search(r"Error\s+count:\s+0", content))

        if success:
            match = re.search(
                r"^Total Runtime:\s+(\d+)\s*ns",
                content,
                flags=re.IGNORECASE | re.MULTILINE,
            )

            total_runtime = int(match.group(1)) if match else 0

            # Capture runtime type and ideal runtime
            match = re.search(
                r"(matrix|vector)\s+unit\s+ideal\s+runtime:\s+(\d+)\s*ns",
                content,
                flags=re.IGNORECASE,
            )

            runtime_type = match.group(1).lower()
            ideal_runtime = int(match.group(2))

    # Calculate actual tiles (min of L2 tiles and max_tiles)
    max_tiles = int(os.environ.get("MAX_TILES", "1"))
    actual_tiles = min(num_tiles, max_tiles)

    return (
        model,
        layer,
        success,
        total_runtime,
        ideal_runtime,
        runtime_type,
        layer_count,
        num_tiles,
        actual_tiles,
    )


def run_rtl_tests(
    layers,
    layer_counts,
    tile_counts,
    num_processes,
    results_folder,
    keep_build=False,
    scverify_test=None,
    debug=False
):
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
    if scverify_test is not None:
        test = scverify_test
    print(f"Running {model} {test}")

    # build VCS simulation binary
    with open(f"{results_folder}/vcs_build.log", "w") as stdout_file:
        env_vars = os.environ.copy()
        env_vars["NETWORK"] = model
        env_vars["TESTS"] = test
        env_vars["SIMS"] = "gold,accelerator"
        env_vars["LD_PRELOAD"] = env_vars["CONDA_PREFIX"] + "/lib/libstdc++.so.6"
        set_default_env_vars(env_vars)
        build_folder = get_build_folder(env_vars)

        subprocess.run(
            ["make", "-f", "scverify/Verify_concat_sim_rtl_v_vcs.mk", "build"],
            cwd=f"{build_folder}/Catapult/{env_vars['TECHNOLOGY']}/clock_{env_vars['CLOCK_PERIOD']}/Accelerator/Accelerator.v1",
            env=env_vars,
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
        )

    pool = mp.Pool(num_processes)

    def signal_handler(signum, frame):
        print(f"Receive signal {signum}, terminating pool...")
        pool.terminate()
        pool.join()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    test_results = []

    for model, tests in layers.items():
        for test in tests:
            pool.apply_async(
                run_rtl_test,
                args=(
                    model,
                    test,
                    layer_counts[model][test],
                    tile_counts[model][test],
                    results_folder,
                    debug,
                ),
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
    set_default_env_vars(env_vars)
    build_folder = get_build_folder(env_vars)

    # Build AccuracyTester binary
    subprocess.run(["make", "clean"], env=env_vars)

    with open(f"{output_folder}/build.log", "w") as stdout_file:
        subprocess.run(
            ["make", "-j", "AccuracyTester"],
            env=env_vars,
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
        )

    # Dump model parameters
    if model == "resnet18":
        model_path = "IMAGENET1K_V1"
    elif model == "resnet50":
        model_path = "IMAGENET1K_V2"
    elif model == "mobilenet_v2":
        model_path = "IMAGENET1K_V2"
    elif model == "mobilebert" and dataset == "sst2":
        model_path = "models/mobilebert/mobilebert-tiny-sst2-bf16/"
    elif model == "mobilebert" and dataset == "squad":
        model_path = "models/mobilebert/mobilebert-tiny-squad-bf16/"
    elif model == "bert" and dataset == "sst2":
        model_path = "JeremiahZ/bert-base-uncased-sst2"
    elif model == "vit":
        model_path = "timm/vit_base_patch16_224.augreg2_in21k_ft_in1k"
    else:
        raise ValueError("Invalid model")

    # Generate input samples from dataset
    if dataset == "imagenet":
        output_data_dir = "data/imagenet"
    elif dataset == "sst2":
        output_data_dir = "data/sst2"
    elif dataset == "squad":
        output_data_dir = "data/squad"
        subprocess.run(
            [
                "python",
                "test/script/dump_bert_dataset.py",
                "--dataset",
                "squad",
                "--model_name_or_path",
                model_path,
                "--output_dir",
                output_data_dir,
            ]
        )
    else:
        raise ValueError("Invalid dataset")

    ic_unroll = int(os.environ["IC_DIMENSION"])
    oc_unroll = int(os.environ["OC_DIMENSION"])
    block_size = max(ic_unroll, oc_unroll)

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
            "--calibration_steps",
            "10",
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
    elif env_vars["DATATYPE"] == "BF16":
        quantization_args = [
            "--bf16",
        ]
    elif env_vars["DATATYPE"] == "MXINT8":
        quantization_args = [
            "--force_scale_power_of_two",
            "--activation",
            "int8,qs=microscaling,bs=" + str(block_size),
            "--weight",
            "int8,qs=microscaling,bs=" + str(block_size),
            "--bf16",
        ]
    elif env_vars["DATATYPE"] == "MXNF4":
        quantization_args = [
            "--activation",
            f"int6,qs=microscaling,bs={block_size},scale=fp8_e5m3",
            "--weight",
            f"int6,qs=microscaling,bs={block_size},scale=fp8_e5m3",
            "--bf16",
        ]
    else:
        raise ValueError("Invalid datatype")

    subprocess.run(
        [
            "mkdir",
            "-p",
            f"{env_vars['CODEGEN_DIR']}/networks/{model}/{env_vars['DATATYPE']}",
        ]
    )

    with open(f"{output_folder}/{model}_{dataset}_compiler.log", "w") as stdout_file:
        subprocess.run(
            [
                "python",
                "voyager-compiler/test/test_codegen.py",
                model,
                "--model_name_or_path",
                model_path,
                "--transpose_weight",
                "--hardware_unrolling",
                f"{ic_unroll},{oc_unroll}",
                *quantization_args,
                "--model_output_dir",
                "test/compiler/networks/" + model + "/" + env_vars["DATATYPE"],
                "--dump_dataset",
                "--dataset_output_dir",
                output_data_dir,
                "--dump_tensors",
            ],
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
        )

    subprocess.run(
        [
            "mkdir",
            "-p",
            f"{env_vars['CODEGEN_DIR']}/networks/{model}/{env_vars['DATATYPE']}/{env_vars['IC_DIMENSION']}x{env_vars['OC_DIMENSION']}_{env_vars['INPUT_BUFFER_SIZE']}x{env_vars['WEIGHT_BUFFER_SIZE']}x{env_vars['ACCUM_BUFFER_SIZE']}_{env_vars['DOUBLE_BUFFERED_ACCUM_BUFFER']}",
        ]
    )

    subprocess.run(
        [
            "protoc",
            "--proto_path=test/compiler/proto/",
            "--python_out=test/compiler/proto/",
            f"test/compiler/proto/tiling.proto",
        ]
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
                    f"{build_folder}/cc/AccuracyTester",
                    model,
                    output_data_dir,
                    str(num_processes),
                    *additional_args,
                ],
                env=env_vars,
                stdout=stdout_file,
                stderr=subprocess.STDOUT,
                timeout=10 * 60 * 60,
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

    if model in ACCURACY_RESULTS:
        gold_accuracy = ACCURACY_RESULTS[model][env_vars["DATATYPE"]]
    else:
        print(f"No gold accuracy specified for {model} {env_vars['DATATYPE']}")
        gold_accuracy = final_accuracy

    return abs(final_accuracy - gold_accuracy) < 1


def add_layers(network, layers, layer_counts, tile_counts, uniquify, skip_layers=None):
    layers[network] = []
    layer_counts[network] = {}
    tile_counts[network] = {}

    skip_layers = [re.compile(p) for p in skip_layers] if skip_layers else []

    if not uniquify:
        with open(
            f"test/compiler/networks/{network}/{os.environ['DATATYPE']}/layers.txt",
            "r",
        ) as f:
            all_layers = f.read().splitlines()
            # Filter out layers that should be skipped
            layers[network] = [
                layer for layer in all_layers
                if not any(p.fullmatch(layer) for p in skip_layers)
            ]
            layer_counts[network] = {layer: 1 for layer in layers[network]}
    else:
        # open the proto file
        with open(
            f"test/compiler/networks/{network}/{os.environ['DATATYPE']}/model.txt",
            "r",
        ) as f:
            contents = f.read()
        params = param_pb2.Model()
        text_format.Parse(contents, params)

        # convert to json
        params_dict = MessageToDict(params, preserving_proto_field_name=True)

        def delete_nested_keys(data, key):
            if isinstance(data, dict):
                for k in list(data.keys()):
                    if k == key:
                        del data[k]
                    else:
                        delete_nested_keys(data[k], key)
            elif isinstance(data, list):
                for item in data:
                    delete_nested_keys(item, key)

        unique_layers = {}
        for op in params_dict["ops"]:
            # skip nop layers
            if "op" in op and op["op"]["op"] == "nop":
                continue

            name = op["op"]["name"] if "op" in op else op["fused_op"]["name"]

            # Skip layers that are in the skip list
            if any(p.fullmatch(name) for p in skip_layers):
                print(f"Skipping layer {name}")
                continue

            # remove the name, memory, and node fields from the op
            delete_nested_keys(op, "name")
            delete_nested_keys(op, "memory")
            delete_nested_keys(op, "scratchpad")
            delete_nested_keys(op, "node")

            if "op" in op:
                kwargs = op["op"]["kwargs"]
            else:
                kwargs = op["fused_op"]["op_list"][0]["kwargs"]

            l2_values = kwargs.get("l2_tiling", {}).get("int_list", {}).get("values", [])
            l2_values = [int(v) for v in l2_values]
            l2_count = math.prod(l2_values) if l2_values else 1

            is_unique_layer = True
            for op_name, op_val in unique_layers.items():
                if not DeepDiff(op, op_val, ignore_order=True):
                    layer_counts[network][op_name] += 1
                    is_unique_layer = False
                    break

            if is_unique_layer:
                unique_layers[name] = op
                layers[network].append(name)
                layer_counts[network][name] = 1
                tile_counts[network][name] = l2_count


def matches(value, rule_value):
    if isinstance(rule_value, list):
        return value in rule_value
    else:
        if rule_value == "*":
            return True
        else:
            return value == rule_value


def get_skip_layers(skip_rules, model, datatype, sim_type, block_size):
    best_rule = None
    best_specificity = -1

    for rule in skip_rules:
        if (
            matches(model, rule["model"])
            and matches(datatype, rule["datatype"])
            and matches(sim_type, rule["sim_type"])
            and matches(block_size, rule["block_size"])
        ):
            # Calculate specificity score (higher is more specific)
            # Exact match = 2, list match = 1, wildcard = 0
            specificity = 0
            specificity += 2 if rule["datatype"] != "*" else 0
            specificity += 1 if isinstance(rule["model"], list) else (2 if rule["model"] != "*" else 0)
            specificity += 1 if isinstance(rule["sim_type"], list) else (2 if rule["sim_type"] != "*" else 0)
            specificity += 2 if rule["block_size"] != "*" else 0

            if specificity > best_specificity:
                best_specificity = specificity
                best_rule = rule

    return set(best_rule["layers"]) if best_rule else set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        required=True,
        help="Model(s) to test for regression (resnet18, mobilebert)",
    )
    parser.add_argument(
        "--uniquify_layers",
        action="store_true",
        help="Whether to remove duplicated layers in the model",
    )
    parser.add_argument(
        "--skip_layers",
        action="store_true",
        help="Whether to skip layers specified in the skip_rules.json",
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
    parser.add_argument(
        "--scverify_test",
        default=None,
        help="(Internal use) Name of scverify test to run",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to dump RTL simulation waveform",
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

    layers = {}
    layer_counts = {}
    tile_counts = {}
    skip_rules = []

    if args.skip_layers:
        with open("ci_skip_rules.json", "r") as f:
            skip_rules = json.load(f)

    # Add codegen layers
    if args.tests is None:
        all_models = []
        if args.models is not None:
            all_models.extend(args.models)

        for network in all_models:
            env_vars = os.environ.copy()
            env_vars["NETWORK"] = network
            if args.sims != "accuracy":
                subprocess.run(["make", "network-proto"], env=env_vars)
                datatype = os.environ["DATATYPE"]
                block_size = max(
                    int(os.environ["OC_DIMENSION"]), int(os.environ["IC_DIMENSION"])
                )
                skip_layers = get_skip_layers(
                    skip_rules, network, datatype, args.sims, block_size
                )
                add_layers(
                    network,
                    layers,
                    layer_counts,
                    tile_counts,
                    args.uniquify_layers,
                    skip_layers,
                )
    else:
        assert (
            len(args.models) == 1
        ), "Only one model can be specified when using --tests"
        layers[args.models[0]] = args.tests.split(",")
        layer_counts[args.models[0]] = {test: 1 for test in layers[args.models[0]]}
        tile_counts[args.models[0]] = {test: 1 for test in layers[args.models[0]]}

    if args.sims == "systemc" or args.sims == "fast-systemc":
        success = run_systemc_tests(
            layers,
            args.num_processes,
            results_folder,
            args.sims == "fast-systemc",
        )
    elif args.sims == "rtl":
        success = run_rtl_tests(
            layers,
            layer_counts,
            tile_counts,
            args.num_processes,
            results_folder,
            args.keep_build,
            args.scverify_test,
            args.debug
        )
    elif args.sims == "gold_model":
        success = run_gold_model_tests(layers, args.num_processes, results_folder)
    elif args.sims == "accuracy":
        success = run_accuracy(
            args.models, args.dataset, args.num_processes, results_folder
        )
    else:
        raise ValueError("Invalid simulation type")

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
