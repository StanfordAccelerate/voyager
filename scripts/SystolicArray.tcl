source scripts/architecture.tcl

set block "SystolicArray"
set full_block_name "SystolicArray<$PE_INPUT_DATATYPE, $PE_WEIGHT_DATATYPE, $PE_PSUM_DATATYPE, $DIMENSION, $DIMENSION>"
set full_block_name_stripped [string map {" " ""} $full_block_name]
set chunk_name "SystolicArrayChunk<$PE_INPUT_DATATYPE, $PE_WEIGHT_DATATYPE, $PE_PSUM_DATATYPE, 4, $DIMENSION>"
set chunk_name_stripped [string map {" " ""} $chunk_name]

source scripts/common.tcl

solution library add {[Block] SystolicArrayChunk.v1}

go libraries

directive set /$full_block_name_stripped/$chunk_name_stripped -MAP_TO_MODULE {[Block] SystolicArrayChunk.v1}

directive set -CLOCKS $clocks

go assembly

go extract
