source scripts/architecture.tcl

set block "MatrixProcessor"
set full_block_name "MatrixProcessor<$IO_DATATYPE, $ACCUM_DATATYPE, $DIMENSION, $DIMENSION, 1024>"
set full_block_name_stripped [string map {" " ""} $full_block_name]

set systolic_array_name "SystolicArray<$PE_INPUT_DATATYPE, $PE_WEIGHT_DATATYPE, $PE_PSUM_DATATYPE, $DIMENSION, $DIMENSION>"
set systolic_array_name_stripped [string map {" " ""} $systolic_array_name]

source scripts/common.tcl

solution library add {[Block] SystolicArray.v1}

go libraries

directive set /$full_block_name_stripped/$systolic_array_name_stripped -MAP_TO_MODULE {[Block] SystolicArray.v1}

directive set -CLOCKS $clocks

go assembly

directive set /$full_block_name_stripped/$full_block_name_stripped:run/run/while:accumulation_buffer.value.$C_DATA_REP_NAME -WORD_WIDTH [expr $ACCUM_DATATYPE_WIDTH * $DIMENSION]
if {[info exists env(DEBUG)] == 0} {
  if {[expr $ACCUM_DATATYPE_WIDTH * $DIMENSION] == 128} {
    directive set /$full_block_name_stripped/$full_block_name_stripped:run/run/while:accumulation_buffer.value.$C_DATA_REP_NAME:rsc -MAP_TO_MODULE mem_1024x128.custom1024x128
  } elseif {[expr $ACCUM_DATATYPE_WIDTH * $DIMENSION] == 256} {
    directive set /$full_block_name_stripped/$full_block_name_stripped:run/run/while:accumulation_buffer.value.$C_DATA_REP_NAME:rsc -MAP_TO_MODULE mem_1024x256.custom1024x256
  } elseif {[expr $ACCUM_DATATYPE_WIDTH * $DIMENSION] == 512} {
    directive set /$full_block_name_stripped/$full_block_name_stripped:run/run/while:accumulation_buffer.value.$C_DATA_REP_NAME:rsc -MAP_TO_MODULE mem_1024x512.custom1024x512
  } elseif {[expr $ACCUM_DATATYPE_WIDTH * $DIMENSION] == 1024} {
    directive set /$full_block_name_stripped/$full_block_name_stripped:run/run/while:accumulation_buffer.value.$C_DATA_REP_NAME:rsc -MAP_TO_MODULE mem_1024x1024.custom1024x1024
  } else {
    error "No memory for width [expr $ACCUM_DATATYPE_WIDTH * $DIMENSION]"
  }
}

go architect

ignore_memory_precedences -from WRITE_ACC_BUFFER* -to READ_ACC_BUFFER*

# to prevent stuttering issues, schedule inputDin and psumIn to happen in the same cycle
cycle set inputSkewerDin.Push()#1 -from psumInSkewerDin.Push()#1 -equal 0

go extract
