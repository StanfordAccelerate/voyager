# Matrix backend builds

`MATRIX_BACKEND=0` selects the systolic backend (the default), and
`MATRIX_BACKEND=1` selects CIM. `config.mk` contains the shared hardware
defaults and supplies the same defines to native C++ builds, Catapult, and
SCVerify. Override settings through Make arguments or environment variables.

Make's network code-generation rules pass `MATRIX_BACKEND` and the CIM settings
to the compiler's `AcceleratorConfig` and select the matching layout policy.
When changing hardware settings, use a fresh `CODEGEN_DIR` or regenerate the
model with `make -B`; existing `model.txt` targets are otherwise reused.

Build the native accelerator with the default CIM geometry:

```bash
source ./.envrc && make -j2 TestRunner DATATYPE=INT8 \
    IC_DIMENSION=64 OC_DIMENSION=16 MATRIX_BACKEND=1
```

`DATATYPE=INT8` selects 8-bit operands and 24-bit accumulation;
`DATATYPE=INT8_32` uses the same operands with 32-bit accumulation.

Supply matching compiler settings through `AcceleratorConfig` or CLI flags.
For the build above, use `--matrix_backend 1 --pe_array_size 64,16
--layout_policy cim`. The compiler maps integer GEMMs and dense convolutions
with dilation equal to 1, estimates weight reuse and buffer use,
and emits schedules through `transform()` and `compile()`.

The instruction mapper accepts explicit L1/L2 schedules in the IR, including
manually specified schedules. CIM requires a complete schedule and rejects
the C++ fallback selected by `MANUAL_TILING=1` or a missing schedule. Explicit
matrix `l2_tiling` counts alone do not supply the internal schedule and are
also rejected by the CIM compiler path. Use the compiler's tiling search or
supply a complete schedule in the IR.

## Performance counters

Both matrix backends enable `ENABLE_PERF_COUNTERS=1` by default. Set it to 0
to remove the counters and read ports. Native and Catapult builds use the same
setting, and build directories distinguish the two configurations.
The `TestRunner-fast` and `TestRunner-checker` targets disable these counters:
their transaction-level Connections channels do not expose cycle handshakes.

Each backend reports ten counters. Eight are common: processor-active cycles,
issued vectors (`array_issue_cycles`), vector admission backpressure
(`input_backpressure_cycles`), result backpressure, and read/write accesses to
the input and accumulation buffers. Systolic adds weight-buffer reads/writes;
CIM adds scheduler waits for resident weights and accepted weight-write beats
(`cim_weight_load_cycles`). Counter indices are stable across backends;
inapplicable indices read zero and are omitted from reports.

For systolic, vector admission is at the input skewer and result backpressure
is after result deskewing. CIM observes its array request and result channels.
`array_issue_cycles` counts vectors, not scalar MACs. Processor-active time
ends at write-back; the harness measures complete accelerator runtime separately.
Stall counters can overlap and do not partition active time. CIM's resident-set
wait diagnostic is not used for systolic PE weight-loading stalls.

Buffer counters increment on actual SRAM accesses across all banks and clients,
including the output controller. Synthetic zero responses that bypass SRAM and
CIM reductions retained in local registers do not count. Stored padding does
count. These totals cover the matrix data buffers; optional MX scale SRAMs are
separate. Each access moves one complete buffer word, and the runner derives
bytes from that word's width, retaining fractional bytes for non-byte-aligned
words. CIM weight bytes and completed set fills are derived from weight-write
beats, preserving partial sets across reports.
There are no separate hardware byte or set-fill counters.
The native runner's `Access counts:` summary is a separate diagnostic. It
reports input and weight buffer reads in elements, which is the unit of the
compiler's tiling estimate.

Counters accumulate modulo 2^32 until reset. Each matrix-unit completion,
including output drain, publishes a coherent snapshot and advances
`snapshot_sequence`. The runner prints `MatrixPerfHardware:` differences between
snapshots, derived byte counts, and `completed_commands`. It retries if the
snapshot changes while reading. Reporting does not gate command execution, and
pending reads finish after total runtime is recorded.

Commands overlap and weights can be prefetched, so a reporting interval is not
exclusive to one command. CIM scheduler weight-wait counts are published after
issue finishes; intermediate snapshots may lag, but final totals include the waits.
Slow readers may combine completions without losing totals, provided fewer than
2^32 events occur per counter between accepted reads. For comparison with an
individual compiler estimate, run that command in isolation.

## Geometry and configuration

`IC_DIMENSION` and `OC_DIMENSION` describe the complete array's hardware input
and output lanes. The CIM macro settings describe each physical macro:

| Setting | Default | Meaning |
| --- | --- | --- |
| `CIM_MACRO_INPUT_LANES` | 64 | Physical macro input lanes |
| `CIM_MACRO_OUTPUT_LANES` | 8 | Physical macro output lanes |
| `CIM_WEIGHT_SETS` | 18 | Resident weight sets per macro |
| `CIM_BASE_A_WIDTH`, `CIM_BASE_B_WIDTH` | 4, 4 | Macro input and weight widths |
| `CIM_BASE_C_WIDTH` | 20 | Macro accumulation width |
| `CIM_MACRO_WRITE_INPUT_LANES` | 1 | Input positions filled by each weight write |
| `CIM_MODE` | 0 | Bit-parallel (0) or bit-serial (1) operation |
| `CIM_SIGNED` | true | Signed operands; false selects unsigned operands |
| `CIM_TILE_INPUT_AXIS_ELEMENTS` | 1 | Elements reduced along a tile's input axis |
| `CIM_TILE_OUTPUT_AXIS_ELEMENTS` | 4 | Elements along a tile's output axis |
| `CIM_INPUT_AXIS_TILES`, `CIM_OUTPUT_AXIS_TILES` | 1, 1 | Array tile counts |

The base widths are configurable. Wider operands are split into slices, and
their partial results are combined with shifts and accumulation. Each weight
uses `weight width / CIM_BASE_B_WIDTH` macro output lanes. The weight width
must be a multiple of `CIM_BASE_B_WIDTH`, and the slice count must divide
`CIM_MACRO_OUTPUT_LANES`. With the table's defaults and 8-bit weights, each
weight uses two macro lanes, giving 64 input lanes and `(8 / 2) * 4 = 16`
output lanes across the array.

`CIM_A_PORT_TILES` defaults to the full input axis. `CIM_B_PORT_TILES` and
`CIM_C_PORT_TILES` default to the full output axis. Narrowing the B port splits
one logical weight row into multiple writes. The processor requires one
complete reduced result in each output-major C beat (`CIM_C_BEAT_LAYOUT=1`).
`CIM_ARRAY_RESULT_SLOTS` defaults to `CIM_INPUT_AXIS_TILES`, and
`CIM_LOCAL_ACCUM_CONTEXTS` defaults to 4.

`IC_PORT_WIDTH` and `OC_PORT_WIDTH` override the external memory port widths
in bits. Leaving them unset retains the datatype's derived defaults.
Non-MXNF4 builds warn when a width is implicit; their defaults use the full
datatype width per lane. MXNF4 retains its four-bit-per-lane port defaults.
`DOUBLE_BUFFERED_ACCUM_BUFFER` selects one or two accumulation SRAM banks.

## HLS and RTL simulation

Available standalone targets are `CIMElement`, `CIMArray`, `CIMProcessor`,
`CIMWeightController`, and `MatrixUnit`. The CIM targets require
`MATRIX_BACKEND=1`. Accelerator consumes the synthesized `MatrixUnit` and
`VectorUnit` libraries. MatrixUnit selects the CIM or systolic child libraries
and owns the input, weight, and accumulation SRAM mappings. Its block script
is `scripts/blocks/MatrixUnit.tcl`, and it can also be built independently.
These builds require a working Catapult license and the selected technology
libraries.

```bash
source ./.envrc && make CIMProcessor DATATYPE=INT8 \
    IC_DIMENSION=64 OC_DIMENSION=16 MATRIX_BACKEND=1 \
    TECHNOLOGY=generic CLOCK_PERIOD=5
```

`CIMElement` exposes both `wclk` and `mclk`; the other targets use `clk`.
CIM builds enable SystemVerilog and the `src/cim` include directory. Make rules
target `concat_rtl.sv` for blocks containing CIM blackboxes and `concat_rtl.v`
for the CIM weight controller and systolic blocks.
`rtl-sim` and the regression runner select the SCVerify flow for the backend.
SCVerify builds run serially so library setup completes before compilation.
MatrixUnit and Accelerator import each child's current synthesized library
explicitly, avoiding stale revisions retained by other parent projects.

Build directories use literal configuration values. Systolic builds with
counters disabled retain upstream's names; enabled performance ports add
`_perf1`; enabled depthwise convolution, explicit external port
widths, and an explicit clock period add fields when supplied. CIM builds also
encode macro dimensions, weight sets, macro base widths, write width,
latency, mode, signedness, element and tile layout, tile ports, beat layout,
result slots, and local accumulation contexts. Changing any CIM hardware
setting selects a separate directory.
Native and SoC Makefiles share the build name from `config.mk`; regression
asks Make to evaluate the same name without running build recipes or copying
configuration defaults into Python. Native regression uses `CLOCK_PERIOD`
from the environment; the harness defaults to 1 ns when it is unset.
`BUILD_DIR` and `CATAPULT_BUILD_DIR`
remain overridable.

## Scratchpad bank timing

The SystemC and RTL harnesses read `memory_config.txt` beside `model.txt`, in
the protobuf text format of `voyager_ir.proto`. The compiler writes the
scratchpad size, bank count, bank width, reserved low-address region and
compiler frequency into it. In banked mode, each bank serves one read or write
word per accelerator cycle. All matrix, vector, bias, scale and sparse ports
share this address-based arbitration; different banks operate concurrently.
The harness prints its configuration and the per-bank and per-port counts.

`SCRATCHPAD_MODEL=independent` replays with independent streams, as programs
without `memory_config.txt` do. To share banks in such a program, set
`SCRATCHPAD_MODEL=banked`, `SCRATCHPAD_SIZE`, `NUM_BANKS` and `BANK_WIDTH`;
`SCRATCHPAD_OFFSET` defaults to 0. Overrides that disagree with
`memory_config.txt` are rejected. `SCRATCHPAD_TRACE=<file>` records every bank
grant.

Host DMA copies stay untimed. The harness does not model DRAM, the SoC
interconnect or the SRAM latency. Compile with `--frequency 0.1` to compare
with generic RTL at `CLOCK_PERIOD=10`. Native SystemC and RTL cycles at the
same clock come from different pipelines; report them separately.
