#!/usr/bin/env python3
"""Element-level CIMIntElement unittest suite definition."""

from __future__ import annotations

from dataclasses import dataclass

from cim_unittest_runner import (
    CIM_SRC_DIR,
    SCRIPT_DIR,
    RunContext,
    SuiteConfig,
    init_seed_for,
    run_seed_matrix,
    run_suite,
    stimulus_seed_for,
)

MODE_PARALLEL = "CIM_MODE_BIT_PARALLEL"
MODE_SERIAL = "CIM_MODE_BIT_SERIAL"
IMPL_MODEL = "CIM_MACRO_WRAPPER_IMPL_MODEL"
IMPL_MACRO_1 = "CIM_MACRO_WRAPPER_IMPL_CIM_MACRO_1"
TEST_NORMAL = 0
TEST_RESET_MID_OP = 1
TOP = "CIMIntElementTb"
MACRO_1_SRC = CIM_SRC_DIR / "cim_macro_1.sv"
MODEL_SRC = CIM_SRC_DIR / "cim_macro_model.sv"
WRAPPER_SRC = CIM_SRC_DIR / "cim_macro_wrapper.sv"
ELEMENT_SRC = CIM_SRC_DIR / "cim_element.sv"
TB_COMMON = SCRIPT_DIR / "cim_element_tb_common.svh"
TB_HELPERS = SCRIPT_DIR / "cim_tb_helpers.svh"


# One immutable unit test instance with Python-owned structural knobs
@dataclass(frozen=True)
class Inst:
    name: str
    mode: str
    signed: bool
    mac_latency: int
    write_input_lanes: int
    iterations: int
    mclk_period: int
    wclk_period: int
    input_lanes: int = 4
    output_lanes: int = 2
    weight_sets: int = 2
    base_a_width: int = 4
    base_b_width: int = 4
    base_c_width: int = 10
    a_width: int = 4
    b_width: int = 4
    expect_dropped_issue: bool = False
    test_kind: int = TEST_NORMAL
    impl: str = IMPL_MODEL


CASES = (
    Inst(
        name="generic_element_parallel_base_unsigned",
        mode=MODE_PARALLEL,
        signed=False,
        mac_latency=1,
        write_input_lanes=2,
        iterations=8,
        mclk_period=5,
        wclk_period=7,
    ),
    Inst(
        name="generic_element_parallel_wide_a_unsigned",
        mode=MODE_PARALLEL,
        signed=False,
        mac_latency=2,
        write_input_lanes=2,
        iterations=8,
        mclk_period=5,
        wclk_period=7,
        a_width=9,
        expect_dropped_issue=True,
    ),
    Inst(
        name="generic_element_parallel_wide_b_unsigned",
        mode=MODE_PARALLEL,
        signed=False,
        mac_latency=2,
        write_input_lanes=2,
        iterations=8,
        mclk_period=5,
        wclk_period=7,
        output_lanes=2,
        b_width=8,
        expect_dropped_issue=False,
    ),
    Inst(
        name="generic_element_parallel_wide_a_b_signed_partial",
        mode=MODE_PARALLEL,
        signed=True,
        mac_latency=3,
        write_input_lanes=1,
        iterations=10,
        mclk_period=6,
        wclk_period=11,
        input_lanes=5,
        output_lanes=3,
        weight_sets=3,
        base_c_width=12,
        a_width=10,
        b_width=8,
        expect_dropped_issue=True,
    ),
    Inst(
        name="generic_element_parallel_k1_equal_clocks_min_c",
        mode=MODE_PARALLEL,
        signed=False,
        mac_latency=1,
        write_input_lanes=1,
        iterations=6,
        mclk_period=6,
        wclk_period=6,
        input_lanes=1,
        base_c_width=9,
    ),
    Inst(
        name="generic_element_parallel_single_set_signed_b16",
        mode=MODE_PARALLEL,
        signed=True,
        mac_latency=2,
        write_input_lanes=2,
        iterations=6,
        mclk_period=5,
        wclk_period=7,
        output_lanes=1,
        weight_sets=1,
        b_width=16,
        expect_dropped_issue=False,
    ),
    Inst(
        name="generic_element_parallel_b12_n2_unsigned",
        mode=MODE_PARALLEL,
        signed=False,
        mac_latency=2,
        write_input_lanes=2,
        iterations=8,
        mclk_period=5,
        wclk_period=7,
        output_lanes=2,
        b_width=12,
        expect_dropped_issue=False,
    ),
    Inst(
        name="generic_element_parallel_a3_one_less_unsigned",
        mode=MODE_PARALLEL,
        signed=False,
        mac_latency=2,
        write_input_lanes=2,
        iterations=8,
        mclk_period=5,
        wclk_period=7,
        a_width=3,
        expect_dropped_issue=False,
    ),
    Inst(
        name="generic_element_parallel_a5_one_more_signed",
        mode=MODE_PARALLEL,
        signed=True,
        mac_latency=2,
        write_input_lanes=2,
        iterations=8,
        mclk_period=5,
        wclk_period=7,
        a_width=5,
        expect_dropped_issue=True,
    ),
    Inst(
        name="generic_element_parallel_latency5_k5_signed",
        mode=MODE_PARALLEL,
        signed=True,
        mac_latency=5,
        write_input_lanes=5,
        iterations=10,
        mclk_period=7,
        wclk_period=11,
        input_lanes=5,
        output_lanes=2,
        weight_sets=5,
        base_c_width=12,
        a_width=5,
        b_width=8,
        expect_dropped_issue=True,
    ),
    Inst(
        name="generic_element_parallel_divisible_clocks_b16_unsigned",
        mode=MODE_PARALLEL,
        signed=False,
        mac_latency=3,
        write_input_lanes=2,
        iterations=8,
        mclk_period=4,
        wclk_period=8,
        output_lanes=1,
        b_width=16,
        expect_dropped_issue=False,
    ),
    Inst(
        name="generic_element_serial_base_signed",
        mode=MODE_SERIAL,
        signed=True,
        mac_latency=1,
        write_input_lanes=2,
        iterations=6,
        mclk_period=5,
        wclk_period=7,
        expect_dropped_issue=True,
    ),
    Inst(
        name="generic_element_serial_window_unsigned",
        mode=MODE_SERIAL,
        signed=False,
        mac_latency=2,
        write_input_lanes=2,
        iterations=6,
        mclk_period=5,
        wclk_period=7,
        base_a_width=2,
        base_c_width=12,
        a_width=6,
        expect_dropped_issue=True,
    ),
    Inst(
        name="generic_element_serial_multi_slice_signed",
        mode=MODE_SERIAL,
        signed=True,
        mac_latency=2,
        write_input_lanes=1,
        iterations=6,
        mclk_period=7,
        wclk_period=10,
        input_lanes=5,
        output_lanes=2,
        weight_sets=3,
        base_a_width=3,
        base_c_width=12,
        a_width=11,
        b_width=8,
        expect_dropped_issue=True,
    ),
    Inst(
        name="generic_element_serial_a1_unsigned",
        mode=MODE_SERIAL,
        signed=False,
        mac_latency=1,
        write_input_lanes=2,
        iterations=5,
        mclk_period=5,
        wclk_period=7,
        a_width=1,
    ),
    Inst(
        name="generic_element_serial_slice_exact_signed",
        mode=MODE_SERIAL,
        signed=True,
        mac_latency=2,
        write_input_lanes=2,
        iterations=5,
        mclk_period=5,
        wclk_period=7,
        base_a_width=3,
        base_c_width=12,
        a_width=6,
        expect_dropped_issue=True,
    ),
    Inst(
        name="generic_element_serial_slice_one_less_unsigned",
        mode=MODE_SERIAL,
        signed=False,
        mac_latency=2,
        write_input_lanes=2,
        iterations=5,
        mclk_period=5,
        wclk_period=7,
        base_a_width=3,
        base_c_width=12,
        a_width=5,
        expect_dropped_issue=True,
    ),
    Inst(
        name="generic_element_serial_slice_one_more_signed",
        mode=MODE_SERIAL,
        signed=True,
        mac_latency=2,
        write_input_lanes=2,
        iterations=5,
        mclk_period=5,
        wclk_period=7,
        base_a_width=3,
        base_c_width=12,
        a_width=7,
        expect_dropped_issue=True,
    ),
    Inst(
        name="generic_element_serial_b12_n2_signed",
        mode=MODE_SERIAL,
        signed=True,
        mac_latency=2,
        write_input_lanes=2,
        iterations=5,
        mclk_period=5,
        wclk_period=7,
        output_lanes=2,
        b_width=12,
        expect_dropped_issue=True,
    ),
    Inst(
        name="generic_element_serial_k1_single_set_unsigned",
        mode=MODE_SERIAL,
        signed=False,
        mac_latency=2,
        write_input_lanes=1,
        iterations=4,
        mclk_period=5,
        wclk_period=7,
        input_lanes=1,
        weight_sets=1,
        base_c_width=9,
        expect_dropped_issue=True,
    ),
    Inst(
        name="generic_element_reset_mid_op",
        mode=MODE_PARALLEL,
        signed=True,
        mac_latency=2,
        write_input_lanes=2,
        iterations=4,
        mclk_period=5,
        wclk_period=7,
        output_lanes=2,
        base_c_width=12,
        a_width=9,
        b_width=8,
        expect_dropped_issue=True,
        test_kind=TEST_RESET_MID_OP,
    ),
    Inst(
        name="generic_element_reset_mid_serial",
        mode=MODE_SERIAL,
        signed=True,
        mac_latency=2,
        write_input_lanes=2,
        iterations=4,
        mclk_period=5,
        wclk_period=7,
        output_lanes=2,
        base_c_width=12,
        a_width=7,
        b_width=8,
        expect_dropped_issue=True,
        test_kind=TEST_RESET_MID_OP,
    ),
    Inst(
        name="generic_element_serial_latency5_wide_a_b_signed",
        mode=MODE_SERIAL,
        signed=True,
        mac_latency=5,
        write_input_lanes=1,
        iterations=5,
        mclk_period=7,
        wclk_period=10,
        input_lanes=5,
        output_lanes=2,
        weight_sets=5,
        base_a_width=3,
        base_c_width=12,
        a_width=11,
        b_width=12,
        expect_dropped_issue=True,
    ),
    Inst(
        name="macro1_element_serial_unsigned_lat3_bk1_equal_clocks",
        mode=MODE_SERIAL,
        signed=False,
        mac_latency=3,
        write_input_lanes=1,
        iterations=6,
        mclk_period=5,
        wclk_period=5,
        weight_sets=1,
        base_c_width=10,
        expect_dropped_issue=True,
        impl=IMPL_MACRO_1,
    ),
    Inst(
        name="macro1_element_serial_unsigned_k8_lat3_bk1_equal_clocks",
        mode=MODE_SERIAL,
        signed=False,
        mac_latency=3,
        write_input_lanes=1,
        iterations=5,
        mclk_period=6,
        wclk_period=6,
        input_lanes=8,
        output_lanes=3,
        weight_sets=1,
        base_c_width=11,
        expect_dropped_issue=True,
        impl=IMPL_MACRO_1,
    ),
)


# Convert a Python boolean into a SystemVerilog one-bit literal
def sv_bit(value: bool) -> str:
    if value:
        return "1'b1"
    return "1'b0"


# Return a short implementation label for test-run status lines
def impl_label(case: Inst) -> str:
    if case.impl == IMPL_MACRO_1:
        return "macro1"

    return "model"


# Return the number of physical B slices represented by one logical output
def weight_slices(case: Inst) -> int:
    return case.b_width // case.base_b_width


# Return the minimum legal base C width for the macro wrapper shape
def minimum_base_c_width(case: Inst) -> int:
    guard_width = 1 if case.input_lanes <= 1 else (case.input_lanes - 1).bit_length()
    return case.base_a_width + case.base_b_width + guard_width


# Catch parameter combinations that the shared SV harness cannot model correctly
def validate_case(case: Inst) -> None:
    if case.input_lanes <= 0 or case.output_lanes <= 0 or case.weight_sets <= 0:
        raise ValueError(f"{case.name}: dimensions must be positive")
    if case.base_a_width <= 0 or case.base_b_width <= 0 or case.base_c_width <= 0:
        raise ValueError(f"{case.name}: base widths must be positive")
    if case.a_width <= 0 or case.b_width <= 0:
        raise ValueError(f"{case.name}: logical widths must be positive")
    if case.write_input_lanes <= 0 or case.mac_latency <= 0:
        raise ValueError(f"{case.name}: WRITE_INPUT_LANES and MAC_LATENCY must be positive")
    if case.iterations <= 0:
        raise ValueError(f"{case.name}: iterations must be positive")
    if case.mclk_period <= 1 or case.wclk_period <= 1:
        raise ValueError(f"{case.name}: clock periods must be greater than one tick")
    if (case.input_lanes % case.write_input_lanes) != 0:
        raise ValueError(f"{case.name}: INPUT_LANES must be divisible by WRITE_INPUT_LANES")
    if (case.b_width % case.base_b_width) != 0:
        raise ValueError(f"{case.name}: B_WIDTH must be divisible by BASE_B_WIDTH")
    if weight_slices(case) <= 0:
        raise ValueError(f"{case.name}: B_WIDTH must be at least BASE_B_WIDTH")
    if case.base_c_width < minimum_base_c_width(case):
        raise ValueError(f"{case.name}: BASE_C_WIDTH is too small for the macro wrapper")
    if case.mode == MODE_SERIAL:
        guard_width = 1 if case.input_lanes <= 1 else (case.input_lanes - 1).bit_length()
        if case.base_c_width <= case.base_b_width + guard_width:
            raise ValueError(f"{case.name}: serial slice capacity must be positive")
    if case.test_kind not in {TEST_NORMAL, TEST_RESET_MID_OP}:
        raise ValueError(f"{case.name}: unknown test kind {case.test_kind}")
    if case.impl not in {IMPL_MODEL, IMPL_MACRO_1}:
        raise ValueError(f"{case.name}: unknown implementation {case.impl}")
    if case.impl == IMPL_MACRO_1:
        if case.mode != MODE_SERIAL:
            raise ValueError(f"{case.name}: macro1 requires bit-serial mode")
        if case.input_lanes <= 1:
            raise ValueError(f"{case.name}: macro1 requires INPUT_LANES greater than one")
        if case.signed:
            raise ValueError(f"{case.name}: macro1 element cases must be unsigned")
        if case.weight_sets != 1:
            raise ValueError(f"{case.name}: macro1 requires WEIGHT_SETS=1")
        if case.write_input_lanes != 1:
            raise ValueError(f"{case.name}: macro1 requires WRITE_INPUT_LANES=1")
        if case.mac_latency < 3:
            raise ValueError(f"{case.name}: macro1 requires MAC_LATENCY >= 3")
        if case.mclk_period != case.wclk_period:
            raise ValueError(f"{case.name}: macro1 requires equal mclk and wclk periods")
        if case.a_width != case.base_a_width:
            raise ValueError(f"{case.name}: macro1 first-pass element cases require A_WIDTH == BASE_A_WIDTH")
        if case.b_width != case.base_b_width:
            raise ValueError(f"{case.name}: macro1 first-pass element cases require B_WIDTH == BASE_B_WIDTH")
        if case.base_c_width != minimum_base_c_width(case):
            raise ValueError(f"{case.name}: macro1 first-pass element cases require exact BASE_C_WIDTH")
        if case.test_kind != TEST_NORMAL:
            raise ValueError(f"{case.name}: macro1 first-pass element cases must be normal tests")


# Emit a tiny top module that binds one parameter set to the element SV harness
def tb_text(case: Inst) -> str:
    validate_case(case)

    return f"""module {TOP};
  localparam string CASE_NAME = \"{case.name}\";
  localparam int unsigned INPUT_LANES = {case.input_lanes};
  localparam int unsigned OUTPUT_LANES = {case.output_lanes};
  localparam int unsigned WRITE_INPUT_LANES = {case.write_input_lanes};
  localparam int unsigned WEIGHT_SETS = {case.weight_sets};
  localparam int unsigned BASE_A_WIDTH = {case.base_a_width};
  localparam int unsigned BASE_B_WIDTH = {case.base_b_width};
  localparam int unsigned BASE_C_WIDTH = {case.base_c_width};
  localparam int unsigned MAC_LATENCY = {case.mac_latency};
  localparam cim_mode_t INST_MODE = {case.mode};
  localparam cim_macro_wrapper_impl_t INST_IMPL = {case.impl};
  localparam bit SIGNED = {sv_bit(case.signed)};
  localparam int unsigned A_WIDTH = {case.a_width};
  localparam int unsigned B_WIDTH = {case.b_width};
  localparam int unsigned NUM_ITERS = {case.iterations};
  localparam int unsigned MCLK_PERIOD = {case.mclk_period};
  localparam int unsigned WCLK_PERIOD = {case.wclk_period};
  localparam bit EXPECT_DROPPED_ISSUE = {sv_bit(case.expect_dropped_issue)};
  localparam int unsigned TEST_KIND = {case.test_kind};

`include \"cim_element_tb_common.svh\"
endmodule
"""


# Report the fields printed before an element case runs
def run_fields(case: Inst, init_seed_count: int, base_seed: int) -> dict[str, object]:
    stimulus_seed = stimulus_seed_for(base_seed, case)
    init_seed = init_seed_for(base_seed, case, 0)
    return {
        "impl": impl_label(case),
        "kind": "normal",
        "init_seed_count": init_seed_count,
        "stimulus_seed": f"0x{stimulus_seed:08x}",
        "first_init_seed": f"0x{init_seed:08x}",
    }


# Run one element case through the common seed matrix
def run_case(context: RunContext) -> int:
    return run_seed_matrix(context)


# Return expected simulator invocations for summary width and accounting
def case_run_count(_case: Inst, init_seed_count: int) -> int:
    return init_seed_count


SUITE = SuiteConfig(
    name="cim_element",
    description=__doc__ or "Run CIMIntElement unit tests.",
    top_module=TOP,
    cases=CASES,
    source_files=(WRAPPER_SRC, MODEL_SRC, MACRO_1_SRC, ELEMENT_SRC),
    include_dirs=(SCRIPT_DIR, CIM_SRC_DIR),
    build_inputs=(
        WRAPPER_SRC,
        MODEL_SRC,
        MACRO_1_SRC,
        ELEMENT_SRC,
        CIM_SRC_DIR / "cim_typedefs.svh",
        TB_COMMON,
        TB_HELPERS,
    ),
    tb_text=tb_text,
    validate_case=validate_case,
    run_fields=run_fields,
    run_case=run_case,
    case_run_count=case_run_count,
)


# Keep import side effects small so ad hoc probes can import case definitions
if __name__ == "__main__":
    raise SystemExit(run_suite(SUITE))
