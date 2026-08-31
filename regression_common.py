"""Helpers shared by the regression drivers.

Two drivers import this module and stay separate on purpose:
  - run_regression.py (repo root): the standalone accelerator flows
    (gold model, SystemC, Catapult scverify RTL, accuracy).
  - test/soc/run_voyager.py: the SoC RTL flow (Chipyard simv, firmware on the
    Rocket core, DMA replay testbench).

Only genuinely flow-independent pieces live here: build-folder naming, the
layers.txt parsing and skip rules, the result table/utilization report, and
the log-scrape patterns both flows grade with.
"""

import os
import re
import signal
import subprocess

import numpy as np
import pandas as pd

# Every flow's checker prints this exact line on a clean comparison.
PASS_PATTERN = r"Error\s+count:\s+0"

# Printed by the gold walk (Simulation.cc); both result parsers scrape it.
IDEAL_RUNTIME_PATTERN = (
    r"(matrix|vector)\s+unit\s+ideal\s+runtime:\s+(\d+)\s*cycles"
)


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


def actual_tile_count(num_tiles):
    """Tiles a run actually executes: MAX_TILES bounds the tile loop; unset or
    0 means the whole loop runs (matching the interpreter, the C emitter, and
    the testbench)."""
    max_tiles = int(os.environ.get("MAX_TILES", "0"))
    return min(num_tiles, max_tiles) if max_tiles > 0 else num_tiles


def utilization(df):
    """Weighted ideal/actual ratio over the layer table. Callers must supply
    'Ideal' and 'Runtime' in the SAME unit (see the drivers' parsers)."""
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
        model_df["Layer"] = pd.Categorical(model_df["Layer"], layers[model])
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


def run_with_timeout(cmd, env, stdout_file, timeout, cwd=None):
    """Run `cmd`, and on timeout kill every process it started.

    `subprocess.run(timeout=...)` kills only the direct child. These tests
    launch the simulator through `make`, so killing `make` leaves the
    simulator running -- orphaned, holding a core, and invisible to the
    sweep, which has already moved on. Giving the child its own session puts
    the whole tree in one process group the timeout can signal.

    Returns True if the command finished, False if it timed out.
    """
    with subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=stdout_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    ) as process:
        try:
            process.communicate(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
            process.communicate()
            return False


def add_layers(network, layers, layer_counts, tile_counts, uniquify,
               skip_layers=None, codegen_root="test/compiler", datatype=None):
    """Populate the test list from the compiler's layers.txt.

    One line per bufferized layer: display name, first and last emitted op of
    the layer's extent, equivalence group from the compiler's bufferization
    cache, and the tile loop's trip count. Uniquify keeps the first layer of
    each group.
    """
    layers[network] = []
    layer_counts[network] = {}
    tile_counts[network] = {}

    skip_layers = [re.compile(p) for p in skip_layers] if skip_layers else []
    datatype = datatype or os.environ["DATATYPE"]

    seen_groups = {}
    with open(f"{codegen_root}/networks/{network}/{datatype}/layers.txt", "r") as f:
        for line in f:
            fields = line.split()
            if not fields:
                continue
            name, _start_op, _end_op, group, tiles = fields
            if any(p.fullmatch(name) for p in skip_layers):
                print(f"Skipping layer {name}")
                continue
            if uniquify and group in seen_groups:
                layer_counts[network][seen_groups[group]] += 1
                continue
            seen_groups[group] = name
            layers[network].append(name)
            layer_counts[network][name] = 1
            tile_counts[network][name] = int(tiles)


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
