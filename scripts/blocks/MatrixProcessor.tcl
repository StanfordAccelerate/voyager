set block "MatrixProcessor"
set full_block_name "MatrixProcessor<$IO_DATATYPE, $ACCUM_DATATYPE, $DIMENSION, $DIMENSION, 1024>"
set full_block_name_stripped [string map {" " ""} $full_block_name]

proc pre_compile {} {
  global PE_INPUT_DATATYPE PE_WEIGHT_DATATYPE PE_PSUM_DATATYPE DIMENSION
  solution design set "SystolicArray<$PE_INPUT_DATATYPE, $PE_WEIGHT_DATATYPE, $PE_PSUM_DATATYPE, $DIMENSION, $DIMENSION>" -mapped
}

proc pre_libraries {} {
  solution library add {[Block] SystolicArray.v1}
}

proc pre_assembly {} {
  global full_block_name_stripped
  global PE_INPUT_DATATYPE PE_WEIGHT_DATATYPE PE_PSUM_DATATYPE DIMENSION
  set systolic_array_name "SystolicArray<$PE_INPUT_DATATYPE, $PE_WEIGHT_DATATYPE, $PE_PSUM_DATATYPE, $DIMENSION, $DIMENSION>"
  set systolic_array_name_stripped [string map {" " ""} $systolic_array_name]

  directive set /$full_block_name_stripped/$systolic_array_name_stripped -MAP_TO_MODULE {[Block] SystolicArray.v1}
}

proc pre_architect {} {
  global full_block_name_stripped C_DATA_REP_NAME ACCUM_DATATYPE_WIDTH DIMENSION TECHNOLOGY memories
  directive set /$full_block_name_stripped/$full_block_name_stripped:run/run/while:accumulation_buffer.value.$C_DATA_REP_NAME -WORD_WIDTH [expr $ACCUM_DATATYPE_WIDTH * $DIMENSION]

  if {$TECHNOLOGY != "generic"} {
    set memory_width [expr $ACCUM_DATATYPE_WIDTH * $DIMENSION]
    if {[info exists memories(1024,$memory_width)]} {
      directive set /$full_block_name_stripped/$full_block_name_stripped:run/run/while:accumulation_buffer.value.$C_DATA_REP_NAME:rsc -MAP_TO_MODULE $memories(1024,$memory_width)
    } else {
      error "No memory specified in technology file for depth=1024, width=$memory_width"
    }

    # if {[expr $ACCUM_DATATYPE_WIDTH * $DIMENSION] == 128} {
    #   directive set /$full_block_name_stripped/$full_block_name_stripped:run/run/while:accumulation_buffer.value.$C_DATA_REP_NAME:rsc -MAP_TO_MODULE mem_1024x128.custom1024x128
    # } elseif {[expr $ACCUM_DATATYPE_WIDTH * $DIMENSION] == 256} {
    #   directive set /$full_block_name_stripped/$full_block_name_stripped:run/run/while:accumulation_buffer.value.$C_DATA_REP_NAME:rsc -MAP_TO_MODULE mem_1024x256.custom1024x256
    # } elseif {[expr $ACCUM_DATATYPE_WIDTH * $DIMENSION] == 512} {
    #   directive set /$full_block_name_stripped/$full_block_name_stripped:run/run/while:accumulation_buffer.value.$C_DATA_REP_NAME:rsc -MAP_TO_MODULE mem_1024x512.custom1024x512
    # } elseif {[expr $ACCUM_DATATYPE_WIDTH * $DIMENSION] == 1024} {
    #   directive set /$full_block_name_stripped/$full_block_name_stripped:run/run/while:accumulation_buffer.value.$C_DATA_REP_NAME:rsc -MAP_TO_MODULE mem_1024x1024.custom1024x1024
    # } else {
    #   error "No memory for width [expr $ACCUM_DATATYPE_WIDTH * $DIMENSION]"
    # }
  }
}

proc pre_extract {} {
  ignore_memory_precedences -from WRITE_ACC_BUFFER* -to READ_ACC_BUFFER*

  # to prevent stuttering issues, schedule inputDin and psumIn to happen in the same cycle
  cycle set inputSkewerDin.Push() -from psumInSkewerDin.Push() -equal 0
}
