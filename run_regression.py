import argparse
import multiprocessing as mp
import subprocess
import sys
import os
import datetime

LAYERS = {
    "resnet18": [
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
    ],
    "mobilebert": [
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
        "classifier",
    ],
}


def print_test_results(test_results):
    passed_tests = []
    failed_tests = []
    for result in test_results:
        test_name = result[0]
        status = result[1]

        if status is True:
            passed_tests.append(test_name)
        else:
            failed_tests.append(test_name)

    print("=" * 10 + "Passed tests:" + "=" * 10)
    print(passed_tests)
    print("=" * 10 + "Failed tests:" + "=" * 10)
    print(failed_tests)

    return len(failed_tests) == 0


def check_environment_vars(required_vars):
    for var in required_vars:
        if var not in os.environ:
            print(f"Please set {var} environment variable")
            sys.exit(1)


def run_systemc_test(model, layer, output_folder):
    env_vars = os.environ.copy()
    env_vars["NETWORK"] = model
    env_vars["TESTS"] = layer
    env_vars["SIMS"] = "customposit,accelerator"
    env_vars["DATA_DIR"] = f"/sim2/shared/MINOTAUR/nn_data/{model}/"

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

    return (f"{model}.{layer}", p.returncode == 0)


def run_systemc_tests(models, num_processes, results_folder):
    check_environment_vars(["DATATYPE", "DIMENSION"])

    # Build TestRunner binary
    subprocess.run(["make", "clean"], env=os.environ)

    with open(f"{results_folder}/build.log", "w") as stdout_file:
        subprocess.run(
            ["make", "-j", "TestRunner"],
            env=os.environ,
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
        )

    pool = mp.Pool(num_processes)

    test_results = []

    for model in models:
        for layer in LAYERS[model]:
            pool.apply_async(
                run_systemc_test,
                args=(model, layer, results_folder),
                callback=test_results.append,
            )

    pool.close()
    pool.join()

    return print_test_results(test_results)


def run_rtl_test(model, layer, output_folder):
    env_vars = os.environ.copy()
    env_vars["NETWORK"] = model
    env_vars["TESTS"] = layer
    env_vars["SIMS"] = "customposit,accelerator"
    env_vars["DATA_DIR"] = f"/sim2/shared/MINOTAUR/nn_data/{model}/"

    with open(f"{output_folder}/{model}_{layer}.log", "w") as stdout_file:
        try:
            subprocess.run(
                ["make", "-f", "scverify/Verify_concat_sim_rtl_v_vcs.mk", "sim"],
                cwd=f"build/{env_vars['DATATYPE']}_{env_vars['DIMENSION']}x{env_vars['DIMENSION']}/Catapult/{env_vars['TECHNOLOGY']}/clock_{env_vars['CLOCK_PERIOD']}/Accelerator/Accelerator.v1",
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

    # TODO: get runtime from log file

    return (f"{model}.{layer}", p.returncode == 0)


def run_rtl_tests(models, num_processes, results_folder):
    check_environment_vars(["DATATYPE", "DIMENSION", "TECHNOLOGY", "CLOCK_PERIOD"])

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
        env_vars["TESTS"] = "layer2_0_downsample"
        env_vars["SIMS"] = "customposit,accelerator"
        env_vars["DATA_DIR"] = f"/sim2/shared/MINOTAUR/nn_data/resnet18/"

        subprocess.run(
            ["make", "-f", "scverify/Verify_concat_sim_rtl_v_vcs.mk", "build"],
            cwd=f"build/{env_vars['DATATYPE']}_{env_vars['DIMENSION']}x{env_vars['DIMENSION']}/Catapult/{env_vars['TECHNOLOGY']}/clock_{env_vars['CLOCK_PERIOD']}/Accelerator/Accelerator.v1",
            env=env_vars,
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
        )

    pool = mp.Pool(num_processes)

    test_results = []

    for model in models:
        for layer in LAYERS[model]:
            pool.apply_async(
                run_rtl_test,
                args=(model, layer, results_folder),
                callback=test_results.append,
            )

    pool.close()
    pool.join()

    return print_test_results(test_results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Model(s) to use for regression (resnet18, mobilebert)",
    )
    parser.add_argument(
        "--sims", type=str, required=True, help="Simulation to run (SystemC or RTL)"
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

    if args.sims == "SystemC":
        success = run_systemc_tests(args.models, args.num_processes, results_folder)
    elif args.sims == "RTL":
        success = run_rtl_tests(args.models, args.num_processes, results_folder)
    else:
        print("Invalid simulation type")
        success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
