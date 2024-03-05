source scripts/architecture.tcl


set block "InputController"
set full_block_name "InputController<$IO_DATATYPE, $DIMENSION>"
set input_controller_stripped [string map {" " ""} $full_block_name]

source scripts/common.tcl

go libraries

directive set -CLOCKS $clocks

go assembly
directive set /$input_controller_stripped/$input_controller_stripped:transposer/transposer/while:if#1:transposeBuffer.$C_DATA_REP_NAME:rsc -MAP_TO_MODULE {[Register]}

go extract
