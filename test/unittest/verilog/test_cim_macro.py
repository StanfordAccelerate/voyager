#!/usr/bin/env python3
"""Macro-wrapper-level CIMIntMacroWrapper unittest suite definition."""

from __future__ import annotations

from dataclasses import dataclass

from cim_unittest_runner import (
    CIM_SRC_DIR,
    SCRIPT_DIR,
    RunContext,
    SuiteConfig,
    init_seed_for,
    run_seed_matrix,
    run_single,
    run_suite,
    stimulus_seed_for,
)

MODE_PARALLEL = "CIM_MODE_BIT_PARALLEL"
MODE_SERIAL = "CIM_MODE_BIT_SERIAL"
IMPL_MODEL = "CIM_MACRO_WRAPPER_IMPL_MODEL"
IMPL_MACRO_1 = "CIM_MACRO_WRAPPER_IMPL_CIM_MACRO_1"
TEST_NORMAL = 0
TEST_EXCEPTION = 1
TOP = "CIMIntMacroWrapperTb"
MACRO_1_SRC = CIM_SRC_DIR / "cim_macro_1.sv"
MODEL_SRC = CIM_SRC_DIR / "cim_macro_model.sv"
WRAPPER_SRC = CIM_SRC_DIR / "cim_macro_wrapper.sv"
TB_COMMON = SCRIPT_DIR / "cim_model_tb_common.svh"
TB_HELPERS = SCRIPT_DIR / "cim_tb_helpers.svh"


# One immutable macro test instance with Python-owned structural knobs
@dataclass(frozen=True)
class Inst:
    name: str
    mode: str
    a_signed: bool
    b_signed_mask: int
    mac_latency: int
    write_input_lanes: int
    iterations: int
    mclk_period: int
    wclk_period: int
    input_lanes: int = 4
    output_lanes: int = 2
    weight_sets: int = 2
    a_width: int = 4
    b_width: int = 4
    c_width: int = 16
    test_kind: int = TEST_NORMAL
    impl: str = IMPL_MODEL


CASES = (
    Inst(
        name="generic_parallel_unsigned_lat1_wb1",
        mode=MODE_PARALLEL,
        a_signed=False,
        b_signed_mask=0,
        mac_latency=1,
        write_input_lanes=1,
        iterations=24,
        mclk_period=5,
        wclk_period=7,
    ),
    Inst(
        name="generic_parallel_unsigned_lat4_wb2",
        mode=MODE_PARALLEL,
        a_signed=False,
        b_signed_mask=0,
        mac_latency=4,
        write_input_lanes=2,
        iterations=28,
        mclk_period=7,
        wclk_period=11,
    ),
    Inst(
        name="generic_parallel_signed_lat2_wb4",
        mode=MODE_PARALLEL,
        a_signed=True,
        b_signed_mask=0b11,
        mac_latency=2,
        write_input_lanes=4,
        iterations=24,
        mclk_period=5,
        wclk_period=9,
    ),
    Inst(
        name="generic_parallel_signed_rows4_lat3_wb2",
        mode=MODE_PARALLEL,
        a_signed=True,
        b_signed_mask=0b11,
        mac_latency=3,
        write_input_lanes=2,
        iterations=36,
        mclk_period=6,
        wclk_period=11,
        weight_sets=4,
    ),
    Inst(
        name="generic_parallel_unsigned_single_row_lat2_wb2",
        mode=MODE_PARALLEL,
        a_signed=False,
        b_signed_mask=0,
        mac_latency=2,
        write_input_lanes=2,
        iterations=16,
        mclk_period=5,
        wclk_period=7,
        weight_sets=1,
    ),
    Inst(
        name="generic_parallel_equal_clocks_mixed_b",
        mode=MODE_PARALLEL,
        a_signed=False,
        b_signed_mask=0b01,
        mac_latency=2,
        write_input_lanes=2,
        iterations=20,
        mclk_period=6,
        wclk_period=6,
    ),
    Inst(
        name="generic_parallel_divisible_clocks_signed_a",
        mode=MODE_PARALLEL,
        a_signed=True,
        b_signed_mask=0,
        mac_latency=2,
        write_input_lanes=2,
        iterations=20,
        mclk_period=4,
        wclk_period=8,
    ),
    Inst(
        name="generic_parallel_rows3_chout4_mixed_b_min_c",
        mode=MODE_PARALLEL,
        a_signed=False,
        b_signed_mask=0b1010,
        mac_latency=3,
        write_input_lanes=2,
        iterations=30,
        mclk_period=5,
        wclk_period=8,
        output_lanes=4,
        weight_sets=3,
        c_width=10,
    ),
    Inst(
        name="generic_parallel_chin1_chout3_min_c",
        mode=MODE_PARALLEL,
        a_signed=False,
        b_signed_mask=0,
        mac_latency=2,
        write_input_lanes=1,
        iterations=12,
        mclk_period=6,
        wclk_period=6,
        input_lanes=1,
        output_lanes=3,
        c_width=9,
    ),
    Inst(
        name="generic_parallel_chin3_mixed_signed_bwidth5",
        mode=MODE_PARALLEL,
        a_signed=True,
        b_signed_mask=0b101,
        mac_latency=3,
        write_input_lanes=1,
        iterations=18,
        mclk_period=5,
        wclk_period=9,
        input_lanes=3,
        output_lanes=3,
        b_width=5,
        c_width=11,
    ),
    Inst(
        name="generic_parallel_rows5_wb5_latency5",
        mode=MODE_PARALLEL,
        a_signed=True,
        b_signed_mask=0b01,
        mac_latency=5,
        write_input_lanes=5,
        iterations=24,
        mclk_period=7,
        wclk_period=11,
        input_lanes=5,
        weight_sets=5,
        c_width=11,
    ),
    Inst(
        name="generic_serial_unsigned_lat1_wb2",
        mode=MODE_SERIAL,
        a_signed=False,
        b_signed_mask=0,
        mac_latency=1,
        write_input_lanes=2,
        iterations=10,
        mclk_period=5,
        wclk_period=7,
    ),
    Inst(
        name="generic_serial_unsigned_lat3_wb1",
        mode=MODE_SERIAL,
        a_signed=False,
        b_signed_mask=0,
        mac_latency=3,
        write_input_lanes=1,
        iterations=10,
        mclk_period=7,
        wclk_period=10,
    ),
    Inst(
        name="generic_serial_signed_lat3_wb4",
        mode=MODE_SERIAL,
        a_signed=True,
        b_signed_mask=0b11,
        mac_latency=3,
        write_input_lanes=4,
        iterations=10,
        mclk_period=6,
        wclk_period=11,
    ),
    Inst(
        name="generic_serial_a1_mixed_b_lat2",
        mode=MODE_SERIAL,
        a_signed=True,
        b_signed_mask=0b10,
        mac_latency=2,
        write_input_lanes=2,
        iterations=12,
        mclk_period=5,
        wclk_period=10,
        a_width=1,
    ),
    Inst(
        name="generic_serial_chin1_a1_min_c",
        mode=MODE_SERIAL,
        a_signed=False,
        b_signed_mask=0,
        mac_latency=1,
        write_input_lanes=1,
        iterations=8,
        mclk_period=5,
        wclk_period=7,
        input_lanes=1,
        a_width=1,
        c_width=6,
    ),
    Inst(
        name="generic_serial_chin3_bwidth5_mixed",
        mode=MODE_SERIAL,
        a_signed=True,
        b_signed_mask=0b10,
        mac_latency=3,
        write_input_lanes=1,
        iterations=8,
        mclk_period=6,
        wclk_period=10,
        input_lanes=3,
        b_width=5,
        c_width=11,
    ),
    Inst(
        name="generic_serial_latency5_rows5_wb5",
        mode=MODE_SERIAL,
        a_signed=True,
        b_signed_mask=0b11,
        mac_latency=5,
        write_input_lanes=5,
        iterations=8,
        mclk_period=7,
        wclk_period=11,
        input_lanes=5,
        weight_sets=5,
        c_width=11,
    ),
    Inst(
        name="macro1_serial_unsigned_lat3_wb1_equal_clocks",
        mode=MODE_SERIAL,
        a_signed=False,
        b_signed_mask=0,
        mac_latency=3,
        write_input_lanes=1,
        iterations=10,
        mclk_period=5,
        wclk_period=5,
        weight_sets=1,
        c_width=10,
        impl=IMPL_MACRO_1,
    ),
    Inst(
        name="macro1_serial_unsigned_chin8_lat3_wb1_equal_clocks",
        mode=MODE_SERIAL,
        a_signed=False,
        b_signed_mask=0,
        mac_latency=3,
        write_input_lanes=1,
        iterations=8,
        mclk_period=6,
        wclk_period=6,
        input_lanes=8,
        output_lanes=3,
        weight_sets=1,
        c_width=11,
        impl=IMPL_MACRO_1,
    ),
    Inst(
        name="generic_illegal_suite",
        mode=MODE_SERIAL,
        a_signed=False,
        b_signed_mask=0,
        mac_latency=1,
        write_input_lanes=2,
        iterations=1,
        mclk_period=5,
        wclk_period=7,
        test_kind=TEST_EXCEPTION,
    ),
)


# Convert a Python boolean into a SystemVerilog one-bit literal
def sv_bit(value: bool) -> str:
    # True means the generated localparam should be driven high
    if value:
        return "1'b1"

    return "1'b0"


# Create a SystemVerilog literal for the per-output-channel B signedness mask
def sv_mask(width: int, value: int) -> str:
    hex_digits = max(1, (width + 3) // 4)
    return f"{width}'h{value & ((1 << width) - 1):0{hex_digits}x}"


# Return a short implementation label for test-run status lines
def impl_label(case: Inst) -> str:
    if case.impl == IMPL_MACRO_1:
        return "macro1"

    return "model"


# Catch parameter combinations that the shared SV harness cannot model correctly
def validate_case(case: Inst) -> None:
    if case.input_lanes <= 0 or case.output_lanes <= 0 or case.weight_sets <= 0:
        raise ValueError(f"{case.name}: dimensions must be positive")
    if case.a_width <= 0 or case.b_width <= 0 or case.c_width <= 0:
        raise ValueError(f"{case.name}: widths must be positive")
    if case.write_input_lanes <= 0 or case.mac_latency <= 0:
        raise ValueError(f"{case.name}: WRITE_INPUT_LANES and MAC_LATENCY must be positive")
    if case.iterations <= 0:
        raise ValueError(f"{case.name}: iterations must be positive")
    if case.mclk_period <= 1 or case.wclk_period <= 1:
        raise ValueError(f"{case.name}: clock periods must be greater than one tick")
    if case.b_signed_mask < 0 or case.b_signed_mask >= (1 << case.output_lanes):
        raise ValueError(f"{case.name}: B signedness mask exceeds OUTPUT_LANES")
    if case.test_kind not in {TEST_NORMAL, TEST_EXCEPTION}:
        raise ValueError(f"{case.name}: unknown test kind {case.test_kind}")
    if case.impl not in {IMPL_MODEL, IMPL_MACRO_1}:
        raise ValueError(f"{case.name}: unknown implementation {case.impl}")
    if case.impl == IMPL_MACRO_1:
        if case.mode != MODE_SERIAL:
            raise ValueError(f"{case.name}: macro1 requires bit-serial mode")
        if case.input_lanes <= 1:
            raise ValueError(f"{case.name}: macro1 requires INPUT_LANES greater than one")
        if case.a_signed or case.b_signed_mask != 0:
            raise ValueError(f"{case.name}: macro1 wrapper cases must be unsigned")
        if case.weight_sets != 1:
            raise ValueError(f"{case.name}: macro1 requires WEIGHT_SETS=1")
        if case.write_input_lanes != 1:
            raise ValueError(f"{case.name}: macro1 requires WRITE_INPUT_LANES=1")
        if case.mac_latency < 3:
            raise ValueError(f"{case.name}: macro1 requires MAC_LATENCY >= 3")
        if case.mclk_period != case.wclk_period:
            raise ValueError(f"{case.name}: macro1 requires equal mclk and wclk periods")


# Emit a tiny top module that binds one parameter set to the shared SV harness
def tb_text(case: Inst) -> str:
    validate_case(case)

    is_serial = case.mode == MODE_SERIAL
    return f"""module {TOP};
  localparam string CASE_NAME = \"{case.name}\";
  localparam int unsigned INPUT_LANES = {case.input_lanes};
  localparam int unsigned OUTPUT_LANES = {case.output_lanes};
  localparam int unsigned WEIGHT_SETS = {case.weight_sets};
  localparam int unsigned A_WIDTH = {case.a_width};
  localparam int unsigned B_WIDTH = {case.b_width};
  localparam int unsigned C_WIDTH = {case.c_width};
  localparam int unsigned WRITE_INPUT_LANES = {case.write_input_lanes};
  localparam int unsigned MAC_LATENCY = {case.mac_latency};
  localparam cim_mode_t INST_MODE = {case.mode};
  localparam cim_macro_wrapper_impl_t INST_IMPL = {case.impl};
  localparam bit A_SIGNED = {sv_bit(case.a_signed)};
  localparam bit [OUTPUT_LANES-1:0] B_SIGNED_MASK = {sv_mask(case.output_lanes, case.b_signed_mask)};
  localparam bit IS_SERIAL = {sv_bit(is_serial)};
  localparam int unsigned NUM_ITERS = {case.iterations};
  localparam int unsigned MCLK_PERIOD = {case.mclk_period};
  localparam int unsigned WCLK_PERIOD = {case.wclk_period};
  localparam int unsigned TEST_KIND = {case.test_kind};

`include \"cim_model_tb_common.svh\"
endmodule
"""


# Report the fields printed before a macro case runs
def run_fields(case: Inst, init_seed_count: int, base_seed: int) -> dict[str, object]:
    stimulus_seed = stimulus_seed_for(base_seed, case)
    init_seed = init_seed_for(base_seed, case, 0)
    if case.test_kind == TEST_EXCEPTION:
        return {
            "impl": impl_label(case),
            "kind": "illegal",
            "stimulus_seed": f"0x{stimulus_seed:08x}",
            "init_seed": f"0x{init_seed:08x}",
        }
    return {
        "impl": impl_label(case),
        "kind": "normal",
        "init_seed_count": init_seed_count,
        "stimulus_seed": f"0x{stimulus_seed:08x}",
        "first_init_seed": f"0x{init_seed:08x}",
    }


# Run one macro case with either normal seed sweep or collected illegal checks
def run_case(context: RunContext) -> int:
    if context.case.test_kind == TEST_EXCEPTION:
        return run_single(context, "illegal.log", ["+collect_exception_types"])
    return run_seed_matrix(context)


# Return expected simulator invocations for summary width and accounting
def case_run_count(case: Inst, init_seed_count: int) -> int:
    if case.test_kind == TEST_EXCEPTION:
        return 1
    return init_seed_count


# Return whether a case is expected to self-check illegal behavior
def is_illegal(case: Inst) -> bool:
    return case.test_kind == TEST_EXCEPTION


SUITE = SuiteConfig(
    name="cim_model",
    description=__doc__ or "Run CIMIntMacroWrapper unit tests.",
    top_module=TOP,
    cases=CASES,
    source_files=(WRAPPER_SRC, MODEL_SRC, MACRO_1_SRC),
    include_dirs=(SCRIPT_DIR, CIM_SRC_DIR),
    build_inputs=(
        WRAPPER_SRC,
        MODEL_SRC,
        MACRO_1_SRC,
        CIM_SRC_DIR / "cim_typedefs.svh",
        TB_COMMON,
        TB_HELPERS,
    ),
    tb_text=tb_text,
    validate_case=validate_case,
    run_fields=run_fields,
    run_case=run_case,
    case_run_count=case_run_count,
    is_illegal=is_illegal,
)


# Keep import side effects small so ad hoc probes can import case definitions
if __name__ == "__main__":
    raise SystemExit(run_suite(SUITE))
