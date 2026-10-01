# Matrix backend builds

`MATRIX_BACKEND=0` selects the systolic backend (the default), and
`MATRIX_BACKEND=1` selects CIM. `config.mk` contains the shared hardware
defaults and supplies the same defines to native C++ builds, Catapult, and
SCVerify. Override settings through Make arguments or environment variables.

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
with stride and dilation equal to 1, estimates weight reuse and buffer use,
and emits schedules through `transform()` and `compile()`.

The instruction mapper accepts explicit L1/L2 schedules in the IR, including
manually specified schedules. The `MANUAL_TILING=1` fallback is not implemented
for CIM yet.

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
CIM SCVerify builds run serially so library setup completes before compilation.
MatrixUnit and Accelerator import each child's current synthesized library
explicitly, avoiding stale revisions retained by other parent projects.

Build directories use literal configuration values. Default systolic builds
retain upstream's names; enabled depthwise convolution, explicit external port
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
