source scripts/architecture.tcl

set block "MaxpoolUnit"
set full_block_name "MaxpoolUnit<$ACCUM_DATATYPE, $IO_DATATYPE, $DIMENSION>"
set full_block_name_stripped [string map {" " ""} $full_block_name]

source scripts/common.tcl

go libraries

directive set -CLOCKS $clocks

go assembly

directive set /$full_block_name_stripped/run/while:maxpool_comparator.value.$C_DATA_REP_NAME:rsc -MAP_TO_MODULE {[Register]}

go extract
