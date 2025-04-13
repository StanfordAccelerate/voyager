set block "MatrixProcessor"
set full_block_name "MatrixProcessor<$IO_DATATYPE, $IO_DATATYPE, $ACCUM_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SCALE_DATATYPE, $IC_DIMENSION, $OC_DIMENSION, $ACCUM_BUFFER_SIZE>"
set full_block_name_stripped [string map {" " ""} $full_block_name]


proc pre_analyze {} {
  global CATAPULT_BUILD_DIR ROOT

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
  global PE_INPUT_DATATYPE PE_WEIGHT_DATATYPE ACCUM_DATATYPE IC_DIMENSION OC_DIMENSION
  solution design set "SystolicArray<$PE_INPUT_DATATYPE, $PE_WEIGHT_DATATYPE, $ACCUM_DATATYPE, $IC_DIMENSION, $OC_DIMENSION>" -mapped
}

proc pre_libraries {} {
  solution library add {[Block] SystolicArray.v1}
}

proc pre_assembly {} {
  global full_block_name_stripped
  global PE_INPUT_DATATYPE PE_WEIGHT_DATATYPE ACCUM_DATATYPE IC_DIMENSION OC_DIMENSION
  set systolic_array_name "SystolicArray<$PE_INPUT_DATATYPE, $PE_WEIGHT_DATATYPE, $ACCUM_DATATYPE, $IC_DIMENSION, $OC_DIMENSION>"
  set systolic_array_name_stripped [string map {" " ""} $systolic_array_name]

  directive set /$full_block_name_stripped/$systolic_array_name_stripped -MAP_TO_MODULE {[Block] SystolicArray.v1}
}

proc pre_extract {} {
  global SUPPORT_MX
  if {$SUPPORT_MX == true} {
    cycle set unscaledAccumulationChannel_delayed.Push() -from unscaledAccumulationChannel.Pop() -equal 2
  } else {
    cycle set inputsToSkewer_delayed.Push() -from inputsToSkewer.Pop() -equal 2
  }
  # to prevent stuttering issues, schedule inputDin and psumIn to happen in the same cycle
  cycle set inputSkewerDin.Push() -from psumInSkewerDin.Push() -equal 0
}
