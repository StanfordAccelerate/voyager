set block "MatrixProcessor"
set full_block_name "MatrixProcessor<$IO_DATATYPE, $ACCUM_DATATYPE, $IC_DIMENSION, $OC_DIMENSION, $ACCUM_BUFFER_SIZE>"
set full_block_name_stripped [string map {" " ""} $full_block_name]

proc pre_compile {} {
  global PE_INPUT_DATATYPE PE_WEIGHT_DATATYPE PE_PSUM_DATATYPE IC_DIMENSION OC_DIMENSION
  solution design set "SystolicArray<$PE_INPUT_DATATYPE, $PE_WEIGHT_DATATYPE, $PE_PSUM_DATATYPE, $IC_DIMENSION, $OC_DIMENSION>" -mapped
}

proc pre_libraries {} {
  solution library add {[Block] SystolicArray.v1}
}

proc pre_assembly {} {
  global full_block_name_stripped
  global PE_INPUT_DATATYPE PE_WEIGHT_DATATYPE PE_PSUM_DATATYPE IC_DIMENSION OC_DIMENSION
  set systolic_array_name "SystolicArray<$PE_INPUT_DATATYPE, $PE_WEIGHT_DATATYPE, $PE_PSUM_DATATYPE, $IC_DIMENSION, $OC_DIMENSION>"
  set systolic_array_name_stripped [string map {" " ""} $systolic_array_name]

  directive set /$full_block_name_stripped/$systolic_array_name_stripped -MAP_TO_MODULE {[Block] SystolicArray.v1}
}

proc pre_architect {} {
  global full_block_name_stripped ACC_BUF_C_DATA_REP_NAME ACCUM_DATATYPE_WIDTH OC_DIMENSION TECHNOLOGY memories
  directive set /$full_block_name_stripped/$full_block_name_stripped:run/run/while:accumulation_buffer.value.$ACC_BUF_C_DATA_REP_NAME -WORD_WIDTH [expr $ACCUM_DATATYPE_WIDTH * $OC_DIMENSION]

  if {$TECHNOLOGY != "generic"} {
    set memory_width [expr $ACCUM_DATATYPE_WIDTH * $OC_DIMENSION]
    directive set /$full_block_name_stripped/$full_block_name_stripped:run/run/while:accumulation_buffer.value.$ACC_BUF_C_DATA_REP_NAME:rsc -MAP_TO_MODULE $memories(1r1w)
  }
}

proc pre_extract {} {
  ignore_memory_precedences -from WRITE_ACC_BUFFER* -to READ_ACC_BUFFER*

  # to prevent stuttering issues, schedule inputDin and psumIn to happen in the same cycle
  cycle set inputSkewerDin.Push() -from psumInSkewerDin.Push() -equal 0

  cycle set *INCR_OUT_STEP:* -from *WRITE_ACC_BUFFER:* -equal 0
}
