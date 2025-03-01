set block "Accelerator"
set full_block_name "Accelerator"

proc pre_compile {} {
  global IO_DATATYPE ACCUM_DATATYPE VECTOR_DATATYPE IC_DIMENSION OC_DIMENSION DATATYPE ACCUM_BUFFER_DATATYPE SUPPORT_MX SCALE_DATATYPE ACCUM_BUFFER_SIZE
  set MP_IO_DATATYPE $IO_DATATYPE
    if {$DATATYPE == "HYBRID_FP8"} {
      set MP_IO_DATATYPE "F9"
    }
    foreach mapped_block [list "InputController<$IO_DATATYPE, $IC_DIMENSION>" "MatrixProcessor<$MP_IO_DATATYPE, $ACCUM_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SCALE_DATATYPE, $IC_DIMENSION, $OC_DIMENSION, $ACCUM_BUFFER_SIZE>" "VectorUnit<$IO_DATATYPE, $VECTOR_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SCALE_DATATYPE, $OC_DIMENSION>" "WeightController<$IO_DATATYPE, $ACCUM_BUFFER_DATATYPE, $IC_DIMENSION, $OC_DIMENSION>"] {
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
  set MP_IO_DATATYPE $IO_DATATYPE
  if {$DATATYPE == "HYBRID_FP8"} {
    set MP_IO_DATATYPE "F9"
  }
  set MatrixProcessorBlock "MatrixProcessor<$MP_IO_DATATYPE, $ACCUM_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SCALE_DATATYPE, $IC_DIMENSION, $OC_DIMENSION, $ACCUM_BUFFER_SIZE>"
  set MatrixProcessorBlock_stripped [string map {" " ""} $MatrixProcessorBlock]

  set InputControllerBlock "InputController<$IO_DATATYPE, $IC_DIMENSION>"
  set InputControllerBlock_stripped [string map {" " ""} $InputControllerBlock]

  set WeightControllerBlock "WeightController<$IO_DATATYPE, $ACCUM_BUFFER_DATATYPE, $IC_DIMENSION, $OC_DIMENSION>"
  set WeightControllerBlock_stripped [string map {" " ""} $WeightControllerBlock]

  set VectorUnitBlock "VectorUnit<$IO_DATATYPE, $VECTOR_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SCALE_DATATYPE, $OC_DIMENSION>"
  set VectorUnitBlock_stripped [string map {" " ""} $VectorUnitBlock]

  directive set /Accelerator/$MatrixProcessorBlock_stripped -MAP_TO_MODULE {[Block] MatrixProcessor.v1}
  directive set /Accelerator/$InputControllerBlock_stripped -MAP_TO_MODULE {[Block] InputController.v1}
  directive set /Accelerator/$WeightControllerBlock_stripped -MAP_TO_MODULE {[Block] WeightController.v1}
  directive set /Accelerator/$VectorUnitBlock_stripped -MAP_TO_MODULE {[Block] VectorUnit.v1}
}

proc pre_architect {} {
  global IO_DATATYPE IC_DIMENSION OC_DIMENSION C_DATA_REP_NAME IO_DATATYPE_WIDTH TECHNOLOGY memories ACCUM_BUFFER_DATATYPE SUPPORT_MX INPUT_BUFFER_SIZE WEIGHT_BUFFER_SIZE
  set double_buffer "DoubleBuffer<$IO_DATATYPE,$IC_DIMENSION,$INPUT_BUFFER_SIZE>"
  set double_buffer_stripped [string map {" " ""} $double_buffer]

  set memory_width [expr $IO_DATATYPE_WIDTH*$IC_DIMENSION]

  directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem0Run/mem0Run/mem0.value.$C_DATA_REP_NAME -WORD_WIDTH $memory_width
  directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem1Run/mem1Run/mem1.value.$C_DATA_REP_NAME -WORD_WIDTH $memory_width

  if {$TECHNOLOGY != "generic" && $TECHNOLOGY != "tsmc40"} {
    directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem0Run/mem0Run/mem0.value.$C_DATA_REP_NAME:rsc -MAP_TO_MODULE [get_memory_name 1 $INPUT_BUFFER_SIZE $memory_width]
    directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem1Run/mem1Run/mem1.value.$C_DATA_REP_NAME:rsc -MAP_TO_MODULE [get_memory_name 1 $INPUT_BUFFER_SIZE $memory_width]
  }

  # Weight double buffer
  # When this is exactly the same as the input double buffer, same directives just repeat
  set double_buffer "DoubleBuffer<$IO_DATATYPE,$OC_DIMENSION,$WEIGHT_BUFFER_SIZE>"
  set double_buffer_stripped [string map {" " ""} $double_buffer]

  set memory_width [expr $IO_DATATYPE_WIDTH*$OC_DIMENSION]

    directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem0Run/mem0Run/mem0.value.$C_DATA_REP_NAME -WORD_WIDTH $memory_width
    directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem1Run/mem1Run/mem1.value.$C_DATA_REP_NAME -WORD_WIDTH $memory_width

    if {$TECHNOLOGY != "generic" && $TECHNOLOGY != "tsmc40"} {
      directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem0Run/mem0Run/mem0.value.$C_DATA_REP_NAME:rsc -MAP_TO_MODULE [get_memory_name 1 $WEIGHT_BUFFER_SIZE $memory_width]
      directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem1Run/mem1Run/mem1.value.$C_DATA_REP_NAME:rsc -MAP_TO_MODULE [get_memory_name 1 $WEIGHT_BUFFER_SIZE $memory_width]
    }

  if {$SUPPORT_MX == true} {
    global SCALE_DATATYPE SCALE_C_DATA_REP_NAME SCALE_DATATYPE_WIDTH

    set weight_scale_double_buffer "DoubleBuffer<$SCALE_DATATYPE,$OC_DIMENSION,$WEIGHT_BUFFER_SIZE>"
    directive set /Accelerator/$weight_scale_double_buffer/$weight_scale_double_buffer:mem0Run/mem0Run/mem0.value.$SCALE_C_DATA_REP_NAME -WORD_WIDTH [expr $SCALE_DATATYPE_WIDTH*$OC_DIMENSION]
    directive set /Accelerator/$weight_scale_double_buffer/$weight_scale_double_buffer:mem1Run/mem1Run/mem1.value.$SCALE_C_DATA_REP_NAME -WORD_WIDTH [expr $SCALE_DATATYPE_WIDTH*$OC_DIMENSION]
  }
}
