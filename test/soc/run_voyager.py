# Script to run Voyager simulations
import os
import argparse
import datetime
import json
import multiprocessing as mp
import subprocess
import re
import sys
import time

# This file lives at <voyager repo>/test/soc/run_voyager.py; resolve through
# the symlink (sims/vcs/run_voyager.py) so both invocation paths work.
VOYAGER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
)
ACCELERATOR_DIR = VOYAGER_DIR
JTAG_DIR = os.path.join(VOYAGER_DIR, "test", "soc", "jtag")

# Shared regression helpers live at the voyager repo root, next to
# run_regression.py (the standalone-accelerator driver).
sys.path.insert(0, VOYAGER_DIR)
from regression_common import (
    IDEAL_RUNTIME_PATTERN,
    PASS_PATTERN,
    actual_tile_count,
    add_layers,
    get_build_folder,
    get_skip_layers,
    print_test_results,
)


def _wait_for_pattern_in_file(path, pattern, timeout=1800):
    """Poll a file until a regex pattern is found, then return the match object."""
    compiled = re.compile(pattern)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with open(path, "r") as f:
                for line in f:
                    m = compiled.search(line)
                    if m:
                        return m
        except FileNotFoundError:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Pattern '{pattern}' not found in {path} within {timeout}s")


def _terminate_process(proc: subprocess.Popen, timeout: float = 5) -> int:
    if proc.poll() is None:
        proc.terminate()

    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.wait()


def run_rtl_simulation(
    model,
    layer,
    layer_count,
    tile_count,
    output_folder,
    env_vars,
    config_path,
    debug,
    jtag_sim=None,
):
    env_vars["NETWORK"] = model
    env_vars["TESTS"] = layer

    cmd = [
        (
            f"./simv-chipyard.harness-{env_vars['CONFIG']}" if not debug
            else f"./simv-chipyard.harness-{env_vars['CONFIG']}-debug"
        ),
        "+permissive",
        "+dramsim",
        "+dramsim_ini_dir=../../generators/testchipip/src/main/resources/dramsim2_ini",
        "+ntb_random_seed_automatic",
        "+verbose"
    ]

    # Extra runtime args, e.g. EXTRA_SIM_FLAGS="+vcs+initreg+0" to zero-init
    # all state (mirrors the SystemC harness) instead of the default random.
    if env_vars.get("EXTRA_SIM_FLAGS"):
        cmd.extend(env_vars["EXTRA_SIM_FLAGS"].split())

    if debug:
        cmd.append("+fsdbfile=dump.fsdb")

    riscv_binary = f"../../tests/voyager/networks/{model}/{config_path}/{layer}.riscv"
    vcs_err = f"{output_folder}/{model}_{layer}.err"
    vcs_log = f"{output_folder}/{model}_{layer}.log"
    openocd_log = f"{output_folder}/{model}_{layer}_openocd.log"
    gdb_log = f"{output_folder}/{model}_{layer}_gdb.log"
    scratchpad_bin = f"{output_folder}/{model}_{layer}_scratchpad.bin"
    checker_log = f"{output_folder}/{model}_{layer}_checker.log"

    env_vars["RISCV_BINARY"] = riscv_binary

    if jtag_sim is None:
        cmd.extend([
            "+max-cycles=10000000",
            "+permissive-off",
            riscv_binary,
        ])

        with open(vcs_log, "w") as log_file, open(vcs_err, "w") as err_file:
            subprocess.run(
                cmd,
                env=env_vars,
                stdout=log_file,
                stderr=err_file,
                timeout=48 * 60 * 60,
            )
    else:
        cmd.extend([
            "+max-cycles=0",
            "+jtag_rbb_enable=1",
            "--rbb-port=9823",  # This is ignored by the simulator
            "+permissive-off",
            "none"
        ])

        with open(vcs_log, "w") as log_file, open(vcs_err, "w") as err_file:
            vcs_proc = subprocess.Popen(
                cmd, env=env_vars, stdout=log_file, stderr=err_file
            )

        try:
            # Step 1: Wait for VCS to announce its JTAG bitbang port
            m = _wait_for_pattern_in_file(vcs_err, r"Listening on port (\d+)")
            vcs_jtag_port = m.group(1)
            print(f"[JTAG] VCS listening on port {vcs_jtag_port}")

            # Step 2: Launch OpenOCD with VCS_JTAG_PORT in env
            openocd_env = {**env_vars, "VCS_JTAG_PORT": vcs_jtag_port}
            with open(openocd_log, "w") as ocd_log:
                openocd_proc = subprocess.Popen(
                    ["openocd", "-f", os.path.join(JTAG_DIR, "cemulator.cfg")],
                    env=openocd_env,
                    stdout=ocd_log,
                    stderr=subprocess.STDOUT,
                )

            try:
                # Step 3: Wait for OpenOCD to announce its GDB port
                m = _wait_for_pattern_in_file(
                    openocd_log,
                    r"Listening on port (\d+) for gdb connections",
                    timeout=4 * 60 * 60,
                )
                gdb_port = m.group(1)
                print(f"[JTAG] OpenOCD GDB server on port {gdb_port}")

                # Step 4: Run GDB (drives the simulation to completion)
                gdb_env = {
                    **env_vars,
                    "OPENOCD_GDB_PORT": gdb_port,
                    "SCRATCHPAD_DUMP_PATH": scratchpad_bin
                }
                with open(gdb_log, "w") as gdb_log_file:
                    subprocess.run(
                        [
                            "riscv64-unknown-elf-gdb",
                            "--batch",
                            riscv_binary,
                            "-x", os.path.join(JTAG_DIR, "run_jtag.gdb"),
                        ],
                        env=gdb_env,
                        stdout=gdb_log_file,
                        stderr=subprocess.STDOUT,
                    )
            except Exception as e:
                print(f"Error during GDB execution: {e}")
            finally:
                _terminate_process(openocd_proc)
        except Exception as e:
            print(f"Error starting OpenOCD initialization: {e}")
        finally:
            _terminate_process(vcs_proc)

        # Step 5: Run jtag_sim_checker on the scratchpad dump
        if jtag_sim == "full":
            build_dir = get_build_folder(env_vars)
            checker_bin = f"{ACCELERATOR_DIR}/{build_dir}/cc/jtag_sim_checker"
            with open(checker_log, "w") as clf:
                subprocess.run(
                    [checker_bin, scratchpad_bin],
                    env=env_vars,
                    stdout=clf,
                    stderr=subprocess.STDOUT,
                )

    return check_rtl_simulation_result(
        model,
        layer,
        layer_count,
        tile_count,
        output_folder,
        jtag_sim,
    )


def check_rtl_simulation_result(
    model,
    layer,
    layer_count,
    tile_count,
    results_folder,
    jtag_sim=None,
):
    success = False
    runtime = 0
    ideal = 0
    runtime_type = ""

    vcs_log = f"{results_folder}/{model}_{layer}.log"
    gdb_log = f"{results_folder}/{model}_{layer}_gdb.log"
    checker_log = f"{results_folder}/{model}_{layer}_checker.log"
    sim_log = checker_log if jtag_sim == "full" else vcs_log

    if os.path.exists(sim_log):
        with open(sim_log, "r") as file:
            content = file.read()

        # Check if test passed
        success = bool(re.search(PASS_PATTERN, content))

        # Capture runtime type (Matrix or Vector) and ideal runtime. Both the
        # compiler's ideal and the firmware's measurement are in accelerator
        # cycles, so the utilization report needs no conversion.
        match = re.search(IDEAL_RUNTIME_PATTERN, content, flags=re.IGNORECASE)
        if match:
            runtime_type = match.group(1).lower()
            ideal = int(match.group(2))

        # Capture actual runtime
        runtime_source = content
        if jtag_sim is not None and os.path.exists(gdb_log):
            with open(gdb_log, "r") as gdb_file:
                runtime_source = gdb_file.read()

        runtime_match = re.search(
            r"Accelerator\s+Runtime\s*:\s*(\d+)\s*cycles",
            runtime_source,
            flags=re.IGNORECASE,
        )
        if runtime_match:
            runtime = int(runtime_match.group(1))

    return (
        model,
        layer,
        success,
        runtime,
        ideal,
        runtime_type,
        layer_count,
        tile_count,
        actual_tile_count(tile_count),
    )


def run_rtl_simulations(
    layers,
    layer_counts,
    tile_counts,
    num_processes,
    results_folder,
    env_vars,
    config_path,
    debug,
    jtag_sim=None,
):
    pool = mp.Pool(num_processes)

    test_results = []

    for model, tests in layers.items():
        for test in tests:
            pool.apply_async(
                run_rtl_simulation,
                args=(
                    model,
                    test,
                    layer_counts[model][test],
                    tile_counts[model][test],
                    results_folder,
                    env_vars,
                    config_path,
                    debug,
                    jtag_sim,
                ),
                callback=test_results.append,
            )
    pool.close()
    pool.join()

    return print_test_results(test_results, layers, results_folder)


def check_rtl_simulation_results(
    layers,
    layer_counts,
    tile_counts,
    results_folder,
    jtag_sim=None,
):
    test_results = []

    for model, tests in layers.items():
        for test in tests:
            test_results.append(
                check_rtl_simulation_result(
                    model,
                    test,
                    layer_counts[model][test],
                    tile_counts[model][test],
                    results_folder,
                    jtag_sim,
                )
            )

    return print_test_results(test_results, layers, results_folder)


def inject_power_pins(testharness_sv_path):
    INJECT_SENTINEL = "dut_VSS"
    CHIPTOP_PATTERN = "ChipTop chiptop0 ("
    CLOSING_PREFIX = "  );"

    with open(testharness_sv_path, "r") as f:
        lines = f.readlines()

    if any(INJECT_SENTINEL in line for line in lines):
        print(f"Power pins already injected in {testharness_sv_path}, skipping.")
        return

    chiptop_idx = next(
        (i for i, line in enumerate(lines) if CHIPTOP_PATTERN in line), None
    )
    assert chiptop_idx is not None, (
        f"Could not find ChipTop instantiation in {testharness_sv_path}"
    )

    closing_idx = next(
        (
            i
            for i in range(chiptop_idx + 1, len(lines))
            if lines[i].startswith(CLOSING_PREFIX)
        ),
        None,
    )
    assert closing_idx is not None, (
        f"Could not find closing '); for ChipTop in {testharness_sv_path}"
    )

    supply_block = [
        "  // BEGIN: POWER PIN INJECT\n",
        "`ifdef GL_PWR_PINS\n",
        "  supply0 dut_VSS;\n",
        "  supply1 dut_VDDPST;\n",
        "  supply1 dut_VDD;\n",
        "`endif // GL_PWR_PINS\n",
        "  // END: POWER PIN INJECT\n",
    ]

    port_block = [
        "    // BEGIN: POWER PIN INJECT\n",
        "`ifdef GL_PWR_PINS\n",
        "   ,.VSS       (dut_VSS)\n",
        "   ,.VDDPST    (dut_VDDPST)\n",
        "   ,.VDD       (dut_VDD)\n",
        "`endif // GL_PWR_PINS\n",
        "    // END: POWER PIN INJECT\n",
    ]

    # Insert port_block first (higher index) to keep chiptop_idx valid
    new_lines = lines[:closing_idx] + port_block + lines[closing_idx:]
    new_lines = new_lines[:chiptop_idx] + supply_block + new_lines[chiptop_idx:]

    bak_path = testharness_sv_path + ".bak"
    with open(bak_path, "w") as f:
        f.writelines(lines)

    with open(testharness_sv_path, "w") as f:
        f.writelines(new_lines)

    print(f"Power pins injected into {testharness_sv_path}.")


def main():
    parser = argparse.ArgumentParser(description="Run Voyager simulations")
    parser.add_argument(
        "--models",
        required=True,
        help="Model(s) to test for regression (resnet18, mobilebert)",
    )
    parser.add_argument(
        "--tests",
        default=None,
        help="Comma separated list of tests to run (e.g. test1,test2)",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=16,
        help="Number of processes to run in parallel",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run the simulation in debug mode to generate waveforms",
    )
    parser.add_argument("--config", required=True, help="SoC config name")
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
        "--jtag-mode",
        choices=["full", "program", "vpi"],
        default=None,
        help=(
            "Run via JTAG interface. "
            "'full': program and data both loaded via JTAG. "
            "'program': program loaded via JTAG, data via VPI. "
            "'vpi': program and data both via VPI, JTAG for startup control only."
        ),
    )
    parser.add_argument(
        "--gl_netlist",
        default=None,
        help="Path to the GL netlist to use for GL simulation.",
    )
    parser.add_argument(
        "--gl_pwr_pins",
        action="store_true",
        help=(
            "Inject VSS/VDDPST/VDD supply declarations and power pin connections into "
            "TestHarness.sv for GL simulation with a power-aware netlist. "
            "Adds +define+GL_PWR_PINS to VCS compilation."
        ),
    )
    parser.add_argument(
        "--results_folder",
        default=None,
        help="Existing regression results folder to parse.",
    )
    parser.add_argument(
        "--semihosting",
        action="store_true",
        help=(
            "Route printf through RISC-V semihosting instead of suppressing it. "
            "Requires --jtag-mode. Output appears in the OpenOCD log."
        ),
    )
    args = parser.parse_args()
    args.models = [s.strip() for s in args.models.split(",")]

    if args.results_folder is None:
        # Create directory with current time
        current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        results_folder = "regression_results/" + current_time
        os.makedirs(results_folder)
        # create softlink to latest results (delete old if exists)
        os.system("rm -f regression_results/latest")
        os.system(f"cd regression_results && ln -sf {current_time} latest")
    else:
        results_folder = args.results_folder

    # Set environment variables
    env_vars = os.environ.copy()
    env_vars["CONFIG"] = args.config
    # The SoC flow always grades the DUT against the gold walk; force it so a
    # stale exported SIMS cannot leak into the simv process.
    env_vars["SIMS"] = "gold,accelerator"
    env_vars["PROJECT_ROOT"] = ACCELERATOR_DIR
    env_vars["CODEGEN_DIR"] = "test/compiler"
    env_vars["NETWORK"] = args.models[0]

    # SCRATCHPAD_SIZE is the whole physical L2 SRAM the compiler plans into
    # (software-pipelined buffers get slots of it); it must equal the SoC
    # scratchpad the Chipyard config instantiates.
    scratchpad_size = int(env_vars.setdefault("SCRATCHPAD_SIZE", "2097152"))
    scratchpad_offset = int(env_vars.setdefault("SCRATCHPAD_OFFSET", "0"))
    num_banks = int(env_vars.setdefault("NUM_BANKS", "16"))

    # The SoC targets require the IC=3 first conv lowered via im2col; the
    # channel-replication conv path is not supported here (see
    # README-soc-bringup.md). Overridable by exporting CONV2D_IM2COL=0.
    env_vars.setdefault("CONV2D_IM2COL", "1")

    # Shield load_memory and check_outputs functions from full JTAG mode
    # since we use GDB for data loading and output checking there
    if args.jtag_mode == "full":
        env_vars["JTAG_SIM"] = "1"
    if args.jtag_mode == "vpi":
        env_vars["SKIP_GDB_LOAD"] = "1"
    if args.jtag_mode is not None and args.semihosting:
        env_vars["SEMIHOSTING"] = "1"

    if args.results_folder is None:
        # Generate SoC verilog
        print("Generating verilog...")
        with open(f"{results_folder}/verilog_generation.log", "w") as log_file:
            subprocess.run(
                ["make", "verilog"],
                env=env_vars,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )

    if args.gl_pwr_pins:
        testharness_sv_path = os.path.join(
            "generated-src",
            f"chipyard.harness.TestHarness.{args.config}",
            "gen-collateral", "TestHarness.sv",
        )
        inject_power_pins(testharness_sv_path)

    # Set the environment variables from the sh file
    sim_cxxflags = []
    with open(f"{ACCELERATOR_DIR}/build/configs/{args.config}/config.sh", "r") as f:
        for line in f:
            if (match := re.match(r"export\s+(\w+)=(.*)", line.strip())) is None:
                continue
            var, val = match.groups()
            val = val.strip("'\"")
            env_vars[var] = val
            sim_cxxflags.append(f"-D{val}" if var == "DATATYPE" else f"-D{var}={val}")
    if args.gl_netlist is not None:
        sim_cxxflags.append("-DGL_SIM")
    env_vars.setdefault("EXTRA_SIM_CXXFLAGS", " ".join(sim_cxxflags))

    # We need to pass these preprocessor defines to the simulator during compilation
    preproc_defines = []
    if env_vars.get("SUPPORT_MVM") == "true":
        preproc_defines.append("+define+SUPPORT_MVM")
    if env_vars.get("SUPPORT_SPMM") == "true":
        preproc_defines.append("+define+SUPPORT_SPMM")
    # Overwrite the default Chipyard Verilog module definitions with the GL netlist if provided
    if args.gl_netlist is not None:
        preproc_defines.append("+define+GL_SIM")
        # TODO: replace with technology-specific preprocessor defines
        preproc_defines.append("+define+TSMC_PWR_AWARE +define+TSMC_CM_UNIT_DELAY")
        preproc_defines.append(f"-f {args.gl_netlist}")
    if args.gl_pwr_pins:
        preproc_defines.append("+define+GL_PWR_PINS")
    env_vars["EXTRA_SIM_SOURCES"] = " ".join(preproc_defines)

    # Set the memory banking configuration for the compiler
    # accesses need to be aligned to accelerator's data width
    # width of datatype * 2 * max(IC,OC)
    if env_vars["DATATYPE"] == "E4M3":
        datatype_width = 8
    elif env_vars["DATATYPE"] == "INT8":
        datatype_width = 8
    elif env_vars["DATATYPE"] == "MXINT8":
        datatype_width = 8
    elif env_vars["DATATYPE"] == "MXNF4":
        datatype_width = 4
    else:
        raise ValueError(
            f"Unknown width for DATATYPE: {env_vars['DATATYPE']}. Please add it to run_voyager.py"
        )

    # BANK_WIDTH here describes the SoC SRAM row width for the testbench's VPI
    # backdoor addressing (SoCMemory) — the compiler-side --bank_width is now
    # computed by the accelerator's codegen.mk and no longer injected from here.
    min_unroll = min(int(env_vars["IC_DIMENSION"]), int(env_vars["OC_DIMENSION"]))
    bank_width = int(datatype_width * min_unroll / 8)

    env_vars.setdefault("BANK_WIDTH", str(bank_width))
    env_vars.setdefault("SPDLOG_LEVEL", "DEBUG")

    # Compile the model(s) and get list of layers
    layers = {}
    layer_counts = {}
    tile_counts = {}
    skip_rules = []

    if args.skip_layers:
        with open(f"{ACCELERATOR_DIR}/ci_skip_rules.json", "r") as f:
            skip_rules = json.load(f)

    if args.tests is None:
        for network in args.models:
            env_vars["NETWORK"] = network
            subprocess.run(
                ["make", "-C", ACCELERATOR_DIR, "network-proto"],
                env=env_vars,
                check=True,
            )
            datatype = env_vars["DATATYPE"]
            block_size = max(
                int(env_vars["OC_DIMENSION"]), int(env_vars["IC_DIMENSION"])
            )
            skip_layers = get_skip_layers(
                skip_rules, network, datatype, "rtl", block_size
            )
            add_layers(
                network,
                layers,
                layer_counts,
                tile_counts,
                args.uniquify_layers,
                skip_layers,
                codegen_root=f"{ACCELERATOR_DIR}/test/compiler",
                datatype=env_vars["DATATYPE"],
            )
    else:
        assert (
            len(args.models) == 1
        ), "Only one model can be specified when using --tests"
        layers[args.models[0]] = args.tests.split(",")
        layer_counts[args.models[0]] = {layer: 1 for layer in layers[args.models[0]]}
        tile_counts[args.models[0]] = {test: 1 for test in layers[args.models[0]]}

    config_path = (
        f"{env_vars['DATATYPE']}/"
        f"{env_vars['IC_DIMENSION']}x{env_vars['OC_DIMENSION']}_"
        f"{env_vars['INPUT_BUFFER_SIZE']}x{env_vars['WEIGHT_BUFFER_SIZE']}x{env_vars['ACCUM_BUFFER_SIZE']}_"
        f"{env_vars['DOUBLE_BUFFERED_ACCUM_BUFFER']}_"
        f"{env_vars['SUPPORT_MVM']}_"
        f"{env_vars['SUPPORT_SPMM']}"
    )

    if args.results_folder is not None:
        success = check_rtl_simulation_results(
            layers,
            layer_counts,
            tile_counts,
            results_folder,
            args.jtag_mode,
        )
        exit(0 if success else 1)

    # Build the simulation binary
    print("Building the simulation binary...")
    with open(f"{results_folder}/sim_build.log", "w") as log_file:
        subprocess.run(
            ["make", "default" if not args.debug else "debug"],
            env=env_vars,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )

    # Build the JTAG sim checker binary
    if args.jtag_mode == "full":
        print("Building jtag_sim_checker...")
        with open(f"{results_folder}/jtag_sim_checker_build.log", "w") as log_file:
            subprocess.run(
                ["make", "-C", f"{VOYAGER_DIR}/test/soc/", "jtag_sim_checker"],
                env=env_vars,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )

    # Generate the SoC C driver code
    print("Generating the SoC C driver code...")
    with open(f"{results_folder}/driver_code_gen.log", "w") as log_file:
        for network in args.models:
            env_vars["NETWORK"] = network
            env_vars["TESTS"] = ",".join(layers[network])

            if args.jtag_mode == "full":
                # Provide tensor data directory for scratchpad pre-loading
                env_vars["DATA_DIR"] = (
                    f"{env_vars['PROJECT_ROOT']}/{env_vars['CODEGEN_DIR']}"
                    f"/networks/{network}/{env_vars['DATATYPE']}/tensor_files"
                )

            subprocess.run(
                ["make",  "-C", f"{VOYAGER_DIR}/test/soc/", "GenerateSoCBinaries"],
                env=env_vars,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )

            # Compile the SoC C code
            for layer in layers[network]:
                cmd = ["make", "-C", "../../tests/voyager/"]

                extra_cflags = []
                # Suppress printf which stalls without a frontend server (all JTAG modes).
                if args.jtag_mode is not None and not args.semihosting:
                    extra_cflags.append("-DSUPPRESS_PRINTF")
                # Skip the semaphore wait in full JTAG mode due to the disabled Voyager backend,
                # or in GL simulation due to unreliable semaphore register path resolution.
                if args.jtag_mode == "full" or args.gl_netlist is not None:
                    extra_cflags.append("-DDISABLE_SEMAPHORE_WAIT")
                # PLIC ids run in device order, so a config without the UART
                # (which takes id 1) moves Voyager down to it. The firmware
                # defaults to the id the UART-bearing configs give it.
                if (int_id := env_vars.get("VOYAGER_INT_ID")) is not None:
                    extra_cflags.append(f"-DVOYAGER_INT_ID={int_id}")
                if extra_cflags:
                    cmd.append(f"EXTRA_CFLAGS={' '.join(extra_cflags)}")

                cmd.append(f"networks/{network}/{config_path}/{layer}.riscv")

                subprocess.run(
                    cmd,
                    env=env_vars,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=True,
                )

    success = run_rtl_simulations(
        layers,
        layer_counts,
        tile_counts,
        args.num_processes,
        results_folder,
        env_vars,
        config_path,
        args.debug,
        args.jtag_mode,
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
