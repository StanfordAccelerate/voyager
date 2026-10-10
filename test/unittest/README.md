# Datapath simulation

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
  Two pending-request fixtures also fill result storage, stall a MAC, and check
  that its weight set stays protected while another set remains writable, in
  both parallel and serial macro modes.
- `test-systemc-processor`: three configurations cover bit-parallel and
  bit-serial macros and double-buffered accumulation. They check resident-set
  replay and ring refills, bias and IC/FX/FY reductions, local-context reuse,
  SRAM fallback, independent read/write progress, and output backpressure.
- `test-systemc-matrix`: both matrix backends run against an independent
  signed-integer GEMM/convolution calculation. They use the actual controllers,
  parameter deserializer, and SRAMs. Three CIM 8x2 builds cover bit-parallel and bit-serial macros,
  plus double-buffered accumulation with narrow memory ports. Each build runs
  thirteen queued commands without intervening resets. Coverage includes
  resident-sequence replay, ring refills, local and SRAM reductions, bias
  followed by no bias, convolution and stride-2 halos, two L1 loop orders,
  outer partial-output contexts, full-buffer addressing, memory and vector
  output, bank handoff, and memory/output backpressure.
- Two systolic MatrixUnit builds cover single and double-buffered accumulation,
  weight reuse, SRAM reductions, bias, and memory output. Counter checks cover
  issued vectors, SRAM accesses, completion snapshots, and reset on both backends.

Run the MatrixUnit regression independently with:

```sh
source ./.envrc
make -C test/unittest -j2 test-systemc-matrix
```

To isolate a shared job, pass `CIM_MATRIX_TEST_ARGS='--case reuse'`.
For CIM-only jobs, run a CIM executable with `--case ring_refill` (or another
job name from `systemc/MatrixUnitTb.cc`). Executables accept `--case NAME` directly.
Selecting a job retains its operand seed and memory layout. The test checks
output values and addresses, memory request addresses/bursts and counts,
completion handshakes, extra traffic, and a simulation watchdog.
The MatrixUnit target generates its access-counter protobuf header under the
test build directory using the existing Conda `protoc` and protobuf libraries.
Use `ENABLE_PERF_COUNTERS=0` to exercise the same datapath tests without counters;
these builds use a separate directory.

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

# Scratchpad port sharing

`make -f test/unittest/Makefile test-systemc-scratchpad` exercises the shared
standalone-harness arbiter: read/read and read/write contention, independent
banks, partial and unaligned words, bank boundaries, backpressure, and
configuration validation. It requires the usual sourced `./.envrc`.

# Gold model

From the checkout root, `make test-gold-model DATATYPE=INT8 IC_DIMENSION=16
OC_DIMENSION=16` checks BF16 vector GEMV with and without bias in a build whose
matrix accumulator is integer. This covers bias-free reduction tiles such as
ResNet18's fully connected layer. Source `./.envrc` first.
