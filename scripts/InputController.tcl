source scripts/architecture.tcl


set block "InputController"
set full_block_name "InputController<$IO_DATATYPE, $DIMENSION>"

source scripts/common.tcl

go libraries

directive set -CLOCKS $clocks

go assembly

go extract
