source scripts/architecture.tcl

set block "VectorOpUnit"
set full_block_name "VectorOpUnit<$IO_DATATYPE, $ACCUM_DATATYPE, $DIMENSION>"

source scripts/common.tcl

go libraries

# set clocks {clk {-CLOCK_PERIOD 10 -CLOCK_EDGE rising -CLOCK_HIGH_TIME 5 -CLOCK_OFFSET 0.000000 -CLOCK_UNCERTAINTY 0.0 -RESET_KIND async -RESET_SYNC_NAME rst -RESET_SYNC_ACTIVE high -RESET_ASYNC_NAME arst_n -RESET_ASYNC_ACTIVE low -ENABLE_NAME {} -ENABLE_ACTIVE high}}

directive set -CLOCKS $slower_clock

go assembly

go extract
