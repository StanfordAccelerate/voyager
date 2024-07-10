import argparse
import multiprocessing as mp
import subprocess
import sys
import os
import datetime
import pandas as pd

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


def print_test_results(test_results, output_folder):
    columns = ["Model", "Layer", "Status", "Runtime", "Ideal"]
    if len(test_results[0]) == 3:
        columns = columns[:3]

    # convert list of tuples to DataFrame
    df = pd.DataFrame(test_results, columns=columns)

    # dump dataframe to pickle
    df.to_pickle(f"{output_folder}/test_results.pkl")

    # get models
    models = df["Model"].unique()

    for model in models:
        print("=" * 10 + f" {model} " + "=" * 10)

        model_df = df[df["Model"] == model]

        # sort according to order in LAYERS
        model_df["Layer"] = pd.Categorical(model_df["Layer"], LAYERS[model])
        model_df.sort_values("Layer", inplace=True)
        # turn categorial back to string
        model_df["Layer"] = model_df["Layer"].astype(str)

        passed = model_df[model_df["Status"] == True]
        failed = model_df[model_df["Status"] == False]

        print("Passed:")
        print(passed["Layer"].to_string(index=False) if not passed.empty else "None")
        print("Failed:")
        print(failed["Layer"].to_string(index=False) if not failed.empty else "None")

        # if runtime column exists, print runtime of each layer
        if "Runtime" in model_df.columns:
            print("Runtime:")
            print(model_df[["Layer", "Runtime"]].to_string(index=False))

    # return True if all tests passed
    return len(df[df["Status"] == False]) == 0


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
    env_vars["DATA_DIR"] = f"/sim2/shared/MINOTAUR/nn_data/unfused_maxpool/{model}/"

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


def run_systemc_tests(models, num_processes, results_folder):
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

    for model in models:
        for layer in LAYERS[model]:
            pool.apply_async(
                run_systemc_test,
                args=(model, layer, results_folder),
                callback=test_results.append,
            )

    pool.close()
    pool.join()

    return print_test_results(test_results, results_folder)


def run_rtl_test(model, layer, output_folder):
    env_vars = os.environ.copy()
    env_vars["NETWORK"] = model
    env_vars["TESTS"] = layer
    env_vars["SIMS"] = "customposit,accelerator"
    env_vars["DATA_DIR"] = f"/sim2/shared/MINOTAUR/nn_data/unfused_maxpool/{model}/"

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


def run_rtl_tests(models, num_processes, results_folder):
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
        env_vars["TESTS"] = "layer2_0_downsample"
        env_vars["SIMS"] = "customposit,accelerator"
        env_vars["DATA_DIR"] = (
            f"/sim2/shared/MINOTAUR/nn_data/unfused_maxpool/resnet18/"
        )

        subprocess.run(
            ["make", "-f", "scverify/Verify_concat_sim_rtl_v_vcs.mk", "build"],
            cwd=f"build/{env_vars['DATATYPE']}_{env_vars['IC_DIMENSION']}x{env_vars['OC_DIMENSION']}/Catapult/{env_vars['TECHNOLOGY']}/clock_{env_vars['CLOCK_PERIOD']}/Accelerator/Accelerator.v1",
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

    return print_test_results(test_results, results_folder)


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
