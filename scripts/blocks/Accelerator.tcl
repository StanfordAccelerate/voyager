set block "Accelerator"
set full_block_name "Accelerator"

proc pre_compile {} {
  global IO_DATATYPE ACCUM_DATATYPE VECTOR_DATATYPE IC_DIMENSION OC_DIMENSION DATATYPE ACCUM_BUFFER_DATATYPE SUPPORT_MX SCALE_DATATYPE ACCUM_BUFFER_SIZE
  foreach mapped_block [list "InputController<$IO_DATATYPE, $IC_DIMENSION>" "MatrixProcessor<$IO_DATATYPE, $IO_DATATYPE, $ACCUM_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SCALE_DATATYPE, $IC_DIMENSION, $OC_DIMENSION, $ACCUM_BUFFER_SIZE>" "VectorUnit<$VECTOR_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SCALE_DATATYPE, $OC_DIMENSION>" "WeightController<$IO_DATATYPE, $ACCUM_BUFFER_DATATYPE, $IC_DIMENSION, $OC_DIMENSION>"] {
    solution design set $mapped_block -mapped
  }
}

proc pre_libraries {} {
  solution library add {[Block] InputController.v1}
  solution library add {[Block] MatrixProcessor.v1}
  solution library add {[Block] VectorUnit.v1}
  solution library add {[Block] WeightController.v1}
}

proc pre_assembly {} {
  global IO_DATATYPE DATATYPE ACCUM_DATATYPE VECTOR_DATATYPE IC_DIMENSION OC_DIMENSION ACCUM_BUFFER_DATATYPE SCALE_DATATYPE SUPPORT_MX ACCUM_BUFFER_SIZE
  set MatrixProcessorBlock "MatrixProcessor<$IO_DATATYPE, $IO_DATATYPE, $ACCUM_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SCALE_DATATYPE, $IC_DIMENSION, $OC_DIMENSION, $ACCUM_BUFFER_SIZE>"
  set MatrixProcessorBlock_stripped [string map {" " ""} $MatrixProcessorBlock]

  set InputControllerBlock "InputController<$IO_DATATYPE, $IC_DIMENSION>"
  set InputControllerBlock_stripped [string map {" " ""} $InputControllerBlock]

  set WeightControllerBlock "WeightController<$IO_DATATYPE, $ACCUM_BUFFER_DATATYPE, $IC_DIMENSION, $OC_DIMENSION>"
  set WeightControllerBlock_stripped [string map {" " ""} $WeightControllerBlock]

  set VectorUnitBlock "VectorUnit<$VECTOR_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SCALE_DATATYPE, $OC_DIMENSION>"
  set VectorUnitBlock_stripped [string map {" " ""} $VectorUnitBlock]

  directive set /Accelerator/$MatrixProcessorBlock_stripped -MAP_TO_MODULE {[Block] MatrixProcessor.v1}
  directive set /Accelerator/$InputControllerBlock_stripped -MAP_TO_MODULE {[Block] InputController.v1}
  directive set /Accelerator/$WeightControllerBlock_stripped -MAP_TO_MODULE {[Block] WeightController.v1}
  directive set /Accelerator/$VectorUnitBlock_stripped -MAP_TO_MODULE {[Block] VectorUnit.v1}
}

proc pre_architect {} {
  global IO_DATATYPE IC_DIMENSION OC_DIMENSION C_DATA_REP_NAME IO_DATATYPE_WIDTH TECHNOLOGY memories ACCUM_BUFFER_DATATYPE ACCUM_BUFFER_SIZE ACCUM_DATATYPE_WIDTH ACC_BUF_C_DATA_REP_NAME SUPPORT_MX INPUT_BUFFER_SIZE WEIGHT_BUFFER_SIZE IC_PORT_WIDTH OC_PORT_WIDTH
  set double_buffer "DoubleBuffer<$IC_PORT_WIDTH,$INPUT_BUFFER_SIZE>"
  set double_buffer_stripped [string map {" " ""} $double_buffer]

  set memory_width [expr $IO_DATATYPE_WIDTH*$IC_DIMENSION]

  directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem0Run/mem0Run/mem0 -WORD_WIDTH $memory_width
  directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem1Run/mem1Run/mem1 -WORD_WIDTH $memory_width

  if {$TECHNOLOGY != "generic" && $TECHNOLOGY != "tsmc40"} {
    directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem0Run/mem0Run/mem0:rsc -MAP_TO_MODULE [get_memory_name 1 $INPUT_BUFFER_SIZE $memory_width]
    directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem1Run/mem1Run/mem1:rsc -MAP_TO_MODULE [get_memory_name 1 $INPUT_BUFFER_SIZE $memory_width]
  }

  # Weight double buffer
  # When this is exactly the same as the input double buffer, same directives just repeat
  set double_buffer "DoubleBuffer<$OC_PORT_WIDTH,$WEIGHT_BUFFER_SIZE>"
  set double_buffer_stripped [string map {" " ""} $double_buffer]

  set memory_width [expr $IO_DATATYPE_WIDTH*$OC_DIMENSION]

    directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem0Run/mem0Run/mem0 -WORD_WIDTH $memory_width
    directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem1Run/mem1Run/mem1 -WORD_WIDTH $memory_width

    if {$TECHNOLOGY != "generic" && $TECHNOLOGY != "tsmc40"} {
      directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem0Run/mem0Run/mem0:rsc -MAP_TO_MODULE [get_memory_name 1 $WEIGHT_BUFFER_SIZE $memory_width]
      directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem1Run/mem1Run/mem1:rsc -MAP_TO_MODULE [get_memory_name 1 $WEIGHT_BUFFER_SIZE $memory_width]
    }

  if {$SUPPORT_MX == true} {
    global SCALE_DATATYPE SCALE_DATATYPE_WIDTH

    set scale_width [expr $SCALE_DATATYPE_WIDTH*$OC_DIMENSION]

    set weight_scale_double_buffer "DoubleBuffer<$scale_width,$WEIGHT_BUFFER_SIZE>"
    directive set /Accelerator/$weight_scale_double_buffer/$weight_scale_double_buffer:mem0Run/mem0Run/mem0 -WORD_WIDTH $scale_width
    directive set /Accelerator/$weight_scale_double_buffer/$weight_scale_double_buffer:mem1Run/mem1Run/mem1 -WORD_WIDTH $scale_width
  }

  set accumulation_buffer "DualPortDoubleBuffer<Pack1D<$ACCUM_BUFFER_DATATYPE,${OC_DIMENSION}UL>,$ACCUM_BUFFER_SIZE>"
  set accumulation_buffer_stripped [string map {" " ""} $accumulation_buffer]
  set memory_width [expr $OC_DIMENSION*$ACCUM_DATATYPE_WIDTH]
  directive set /Accelerator/$accumulation_buffer_stripped/bank0_run/bank0.value.$ACC_BUF_C_DATA_REP_NAME -WORD_WIDTH $memory_width
  directive set /Accelerator/$accumulation_buffer_stripped/bank1_run/bank1.value.$ACC_BUF_C_DATA_REP_NAME -WORD_WIDTH $memory_width
}

proc pre_extract {} {
  ignore_memory_precedences -from WRITE_BANK_0* -to READ_BANK_0*
  ignore_memory_precedences -from WRITE_BANK_1* -to READ_BANK_1*
}
