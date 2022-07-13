source scripts/architecture.tcl

set block "VectorUnit"
set full_block_name "VectorUnit<$IO_DATATYPE, $ACCUM_DATATYPE, $DIMENSION>"

source scripts/common.tcl

go libraries

directive set -CLOCKS $clocks

go assembly

directive set /VectorUnit<P8,P16,16>/MaxpoolUnit<P8,16>/run/while:maxpool_comparator.value.bits:rsc -MAP_TO_MODULE {[Register]}

go extract
