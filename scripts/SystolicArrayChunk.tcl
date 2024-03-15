source scripts/architecture.tcl

set block "SystolicArrayChunk"
set full_block_name "SystolicArrayChunk<$PE_INPUT_DATATYPE, $PE_WEIGHT_DATATYPE, $PE_PSUM_DATATYPE, 4, $DIMENSION>"
set full_block_name_stripped [string map {" " ""} $full_block_name]
set row_name "SystolicArrayRow<$PE_INPUT_DATATYPE, $PE_WEIGHT_DATATYPE, $PE_PSUM_DATATYPE, $DIMENSION>"
set row_name_stripped [string map {" " ""} $row_name]

source scripts/common.tcl

solution library add {[Block] SystolicArrayRow.v1}

go libraries

directive set /$full_block_name_stripped/$row_name_stripped -MAP_TO_MODULE {[Block] SystolicArrayRow.v1}

directive set -CLOCKS $clocks

go assembly

go extract
