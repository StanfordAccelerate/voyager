source scripts/architecture.tcl

set block "ProcessingElement"
set full_block_name "ProcessingElement<P8D, P8D, P16D>"

source scripts/common.tcl

go libraries

set clocks {clk {-CLOCK_PERIOD 10 -CLOCK_EDGE rising -CLOCK_HIGH_TIME 5 -CLOCK_OFFSET 0.000000 -CLOCK_UNCERTAINTY 0.0 -RESET_KIND async -RESET_SYNC_NAME rst -RESET_SYNC_ACTIVE high -RESET_ASYNC_NAME arst_n -RESET_ASYNC_ACTIVE low -ENABLE_NAME {} -ENABLE_ACTIVE high}}

directive set -CLOCKS $clocks

go assembly

go extract
