set block "MatrixProcessor"
set full_block_name "MatrixProcessor<$IO_DATATYPE, $ACCUM_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SUPPORT_MX, $IC_DIMENSION, $OC_DIMENSION, 1024>"
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
  global full_block_name_stripped ACC_BUF_C_DATA_REP_NAME ACCUM_DATATYPE_WIDTH OC_DIMENSION TECHNOLOGY memories SUPPORT_MX

  if {$SUPPORT_MX == true} {
    set accumulation_buffer_path "/$full_block_name_stripped/$full_block_name_stripped:process_accumulation/process_accumulation/constexpr_if.if:while:accumulation_buffer.value.$ACC_BUF_C_DATA_REP_NAME"
  } else {
    set accumulation_buffer_path "/$full_block_name_stripped/$full_block_name_stripped:run/run/while:accumulation_buffer.value.$ACC_BUF_C_DATA_REP_NAME"
  }
  directive set $accumulation_buffer_path -WORD_WIDTH [expr $ACCUM_DATATYPE_WIDTH * $OC_DIMENSION]

  if {$TECHNOLOGY != "generic"} {
    set memory_width [expr $ACCUM_DATATYPE_WIDTH * $OC_DIMENSION]
    directive set $accumulation_buffer_path:rsc -MAP_TO_MODULE $memories(1r1w)
  }

  # Unroll loops that were not unrolled
  if {$SUPPORT_MX == true} {
    for {set index 0} { $index < 6 } { incr index } {
      directive set /$full_block_name_stripped/process_accumulation/UNROLL_$index -UNROLL yes
    }
  }
}

proc pre_extract {} {
  ignore_memory_precedences -from WRITE_ACC_BUFFER* -to READ_ACC_BUFFER*

  # to prevent stuttering issues, schedule inputDin and psumIn to happen in the same cycle
  cycle set inputSkewerDin.Push() -from psumInSkewerDin.Push() -equal 0
}
