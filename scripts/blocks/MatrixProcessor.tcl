set block "MatrixProcessor"
set full_block_name "MatrixProcessor<InputTypeList, WeightTypeList, $SA_INPUT_TYPE, $SA_WEIGHT_TYPE, $ACCUM_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SCALE_DATATYPE, $IC_DIMENSION, $OC_DIMENSION, $ACCUM_BUFFER_SIZE>"
set full_block_name_stripped [string map {" " ""} $full_block_name]

proc pre_analyze {} {
  global CATAPULT_BUILD_DIR ROOT IC_DIMENSION

  set path $ROOT/$CATAPULT_BUILD_DIR/ProcessingElement.log
  puts $path
  if [file exist $path] {
        if {[catch {exec grep -P "Info:.+Final schedule of SEQUENTIAL '/ProcessingElement.+:run/run" $path} res] == 0} {
            puts "$res"
            regexp {Latency = (\d+),} $res temp pe_latency
            puts "PE Latency: $pe_latency"
            solution options set Input/CompilerFlags  "[solution options get Input/CompilerFlags] -DPE_LATENCY=$pe_latency"
        } else {
            puts "\033\[31mERROR: PE latency not found\033\[0m"
        }
  } else {
    puts "\033\[31mERROR: $path not found\033\[0m"
  }
}

proc pre_compile {} {
  global SA_INPUT_TYPE SA_WEIGHT_TYPE ACCUM_DATATYPE IC_DIMENSION OC_DIMENSION
  solution design set "SystolicArray<$SA_INPUT_TYPE, $SA_WEIGHT_TYPE, $ACCUM_DATATYPE, $IC_DIMENSION, $OC_DIMENSION>" -mapped
}

proc pre_libraries {} {
  solution library add {[Block] SystolicArray.v1}
}

proc pre_assembly {} {
  global full_block_name_stripped
  global SA_INPUT_TYPE SA_WEIGHT_TYPE ACCUM_DATATYPE IC_DIMENSION OC_DIMENSION
  set systolic_array_name "SystolicArray<$SA_INPUT_TYPE, $SA_WEIGHT_TYPE, $ACCUM_DATATYPE, $IC_DIMENSION, $OC_DIMENSION>"
  set systolic_array_name_stripped [string map {" " ""} $systolic_array_name]

  directive set /$full_block_name_stripped/$systolic_array_name_stripped -MAP_TO_MODULE {[Block] SystolicArray.v1}
}

proc pre_extract {} {
  global DOUBLE_BUFFERED_ACCUM_BUFFER
  cycle set accumulation_buffer_read_data.Pop() -from accumulation_buffer_read_address.Push() -equal 2
  if {$DOUBLE_BUFFERED_ACCUM_BUFFER == true} {
    cycle set accumulation_buffer_read_data.Pop()#1 -from accumulation_buffer_read_address.Push()#1 -equal 2
  }
}
