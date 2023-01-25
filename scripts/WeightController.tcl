source scripts/architecture.tcl

set block "WeightController"
set full_block_name "WeightController<$IO_DATATYPE, $ACCUM_DATATYPE, $DIMENSION, $DIMENSION>"
set weight_controller_stripped [string map {" " ""} $full_block_name]

source scripts/common.tcl

go libraries

set clocks {clk {-CLOCK_PERIOD 10 -CLOCK_EDGE rising -CLOCK_HIGH_TIME 5 -CLOCK_OFFSET 0.000000 -CLOCK_UNCERTAINTY 0.0 -RESET_KIND async -RESET_SYNC_NAME rst -RESET_SYNC_ACTIVE high -RESET_ASYNC_NAME arst_n -RESET_ASYNC_ACTIVE low -ENABLE_NAME {} -ENABLE_ACTIVE high}}

directive set -CLOCKS $clocks

go assembly

directive set /$weight_controller_stripped/$weight_controller_stripped:transposer/transposer/while:if:transposeBuffer.bits:rsc -MAP_TO_MODULE {[Register]}
directive set /$weight_controller_stripped/$weight_controller_stripped:transposeGrads/transposeGrads/while:if:transposeBuffer.bits:rsc -MAP_TO_MODULE {[Register]}

go extract
