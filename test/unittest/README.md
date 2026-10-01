# CIM simulation

From the checkout root:

```sh
source ./.envrc
make -C test/unittest -j4 test-systemc
```

This requires the existing Catapult installation (`CATAPULT_ROOT`) for SystemC,
AC datatypes, Connections, and its C++ compiler, plus Verilator from Conda.
These tests use event-driven SystemC; do not enable `CONNECTIONS_FAST_SIM`.

- `test-systemc-element`: three fixtures compare the SystemC element with
  Verilated RTL cycle by cycle, covering the packed interface, ready/busy/retire
  timing, reset, signed serial arithmetic, and operand slicing.
- `test-systemc-array`: four fixtures check routing and reduction across tiles,
  resident weight sets, narrow weight ports, both result layouts, mixed
  completions, output backpressure, and reset recovery. This also exercises
  `CIMTile`; there is no separate tile suite or geometry sweep.
- `test-systemc-processor`: three configurations cover bit-parallel and
  bit-serial macros and double-buffered accumulation. They check resident-set
  replay and ring refills, bias and IC/FX/FY reductions, local-context reuse,
  SRAM fallback, independent read/write progress, and output backpressure.

The CIM processor supports signed and unsigned integer operands and emits
output-major results. It consumes ordered weight-sequence descriptors alongside
packed weight beats. Backend defaults and the descriptor are declared in
`src/cim/CIMConfig.h` and `src/cim/CIMTypes.h`.

SRAM-backed reductions require enough independent output contexts between
dependent operations to cover the accumulation-memory feedback latency. The
processor diagnoses stale partial-sum reads in simulation; this check does
not add a hardware interlock. These fixtures emulate memory backpressure and
check that dependent reads follow completed writes.

Use each target independently. Generated models, dependency files, and
executables stay under `test/unittest/build/systemc`. Run `make -C test/unittest
clean` after changing toolchains.

The Verilog suites remain available through `make -C test/unittest test-verilog
TEST_ARGS='--jobs 4'`, or by invoking `verilog/test_cim_macro.py` and
`verilog/test_cim_element.py` directly with Python.
