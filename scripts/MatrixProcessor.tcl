source scripts/architecture.tcl

set block "MatrixProcessor"
set full_block_name "MatrixProcessor<$IO_DATATYPE, $ACCUM_DATATYPE, $DIMENSION, $DIMENSION, 1024>"
set full_block_name_stripped [string map {" " ""} $full_block_name]

source scripts/common.tcl

go libraries

directive set -CLOCKS $clocks

go assembly

directive set /$full_block_name_stripped/$full_block_name_stripped:run/run/while:accumulation_buffer.value.bits -WORD_WIDTH [expr $ACCUM_DATATYPE_WIDTH * $DIMENSION]
if {[info exists env(DEBUG)] == 0} {
  directive set /$full_block_name_stripped/$full_block_name_stripped:run/run/while:accumulation_buffer.value.bits:rsc -MAP_TO_MODULE mem_1024x402.custom1024x402
}

go architect

ignore_memory_precedences -from WRITE_ACC_BUFFER* -to READ_ACC_BUFFER*

go extract
