source scripts/architecture.tcl

set block "WeightController"
set full_block_name "WeightController<$IO_DATATYPE, $DIMENSION, $DIMENSION>"
set weight_controller_stripped [string map {" " ""} $full_block_name]

source scripts/common.tcl

go libraries

directive set -CLOCKS $clocks

go assembly

directive set /$weight_controller_stripped/$weight_controller_stripped:transposer/transposer/while:if:transposeBuffer.bits:rsc -MAP_TO_MODULE {[Register]}

go extract
