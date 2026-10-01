# Matrix backend build directories

Build directories use literal configuration values. Default systolic builds
retain upstream's names; enabled depthwise convolution, explicit external port
widths, and an explicit clock period add fields when supplied. CIM builds also
encode macro dimensions, weight sets, operand and result widths, write width,
latency, mode, signedness, element and tile layout, tile ports, beat layout,
result slots, and local accumulation contexts. Changing any CIM hardware
setting selects a separate directory.
Native and SoC Makefiles share the build name from `config.mk`; regression
asks Make to evaluate the same name without running build recipes or copying
configuration defaults into Python. Native regression uses `CLOCK_PERIOD`
from the environment; the harness defaults to 1 ns when it is unset.
`BUILD_DIR` and `CATAPULT_BUILD_DIR`
remain overridable.
