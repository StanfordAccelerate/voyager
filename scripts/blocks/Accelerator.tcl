set block "Accelerator"
set full_block_name "Accelerator"

proc pre_compile {} {
  global INPUT_TYPE_LIST WEIGHT_TYPE_LIST SA_INPUT_TYPE SA_WEIGHT_TYPE ACCUM_DATATYPE ACCUM_BUFFER_DATATYPE VECTOR_DATATYPE SCALE_DATATYPE IC_DIMENSION OC_DIMENSION SUPPORT_MX IC_PORT_WIDTH OC_PORT_WIDTH ACCUM_BUFFER_SIZE INPUT_BUFFER_WIDTH WEIGHT_BUFFER_WIDTH
  foreach mapped_block [list \
    "InputController<InputTypeList, $IC_DIMENSION, $IC_PORT_WIDTH, $INPUT_BUFFER_WIDTH>" \
    "WeightController<WeightTypeList, $ACCUM_BUFFER_DATATYPE, $IC_DIMENSION, $OC_DIMENSION, $OC_PORT_WIDTH, $WEIGHT_BUFFER_WIDTH>" \
    "MatrixProcessor<InputTypeList, WeightTypeList, $SA_INPUT_TYPE, $SA_WEIGHT_TYPE, $ACCUM_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SCALE_DATATYPE, $IC_DIMENSION, $OC_DIMENSION, $ACCUM_BUFFER_SIZE>" \
    "VectorUnit<$VECTOR_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SCALE_DATATYPE, $OC_DIMENSION>" \
    "MatrixParamsDeserializer<[expr {$SUPPORT_MX ? 5 : 3}]>" \
  ] {
    solution design set $mapped_block -mapped
  }
}

proc pre_libraries {} {
  solution library add {[Block] InputController.v1}
  solution library add {[Block] MatrixProcessor.v1}
  solution library add {[Block] VectorUnit.v1}
  solution library add {[Block] MatrixParamsDeserializer.v1}
  solution library add {[Block] WeightController.v1}
}

proc pre_assembly {} {
  global INPUT_TYPE_LIST WEIGHT_TYPE_LIST SA_INPUT_TYPE SA_WEIGHT_TYPE ACCUM_DATATYPE ACCUM_BUFFER_DATATYPE VECTOR_DATATYPE SCALE_DATATYPE IC_DIMENSION OC_DIMENSION SUPPORT_MX IC_PORT_WIDTH OC_PORT_WIDTH ACCUM_BUFFER_SIZE INPUT_BUFFER_WIDTH WEIGHT_BUFFER_WIDTH

  set InputControllerBlock "InputController<InputTypeList, $IC_DIMENSION, $IC_PORT_WIDTH, $INPUT_BUFFER_WIDTH>"
  set InputControllerBlock_stripped [string map {" " ""} $InputControllerBlock]

  set WeightControllerBlock "WeightController<WeightTypeList, $ACCUM_BUFFER_DATATYPE, $IC_DIMENSION, $OC_DIMENSION, $OC_PORT_WIDTH, $WEIGHT_BUFFER_WIDTH>"
  set WeightControllerBlock_stripped [string map {" " ""} $WeightControllerBlock]

  set MatrixProcessorBlock "MatrixProcessor<InputTypeList, WeightTypeList, $SA_INPUT_TYPE, $SA_WEIGHT_TYPE, $ACCUM_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SCALE_DATATYPE, $IC_DIMENSION, $OC_DIMENSION, $ACCUM_BUFFER_SIZE>"
  set MatrixProcessorBlock_stripped [string map {" " ""} $MatrixProcessorBlock]

  set VectorUnitBlock "VectorUnit<$VECTOR_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SCALE_DATATYPE, $OC_DIMENSION>"
  set VectorUnitBlock_stripped [string map {" " ""} $VectorUnitBlock]

  set MatrixParamsDeserializerBlock "MatrixParamsDeserializer<[expr {$SUPPORT_MX ? 5 : 3}]>"
  set MatrixParamsDeserializerBlock_stripped [string map {" " ""} $MatrixParamsDeserializerBlock]

  directive set /Accelerator/$InputControllerBlock_stripped -MAP_TO_MODULE {[Block] InputController.v1}
  directive set /Accelerator/$WeightControllerBlock_stripped -MAP_TO_MODULE {[Block] WeightController.v1}
  directive set /Accelerator/$MatrixProcessorBlock_stripped -MAP_TO_MODULE {[Block] MatrixProcessor.v1}
  directive set /Accelerator/$VectorUnitBlock_stripped -MAP_TO_MODULE {[Block] VectorUnit.v1}
  directive set /Accelerator/$MatrixParamsDeserializerBlock_stripped -MAP_TO_MODULE {[Block] MatrixParamsDeserializer.v1}
}

proc pre_architect {} {
  global TECHNOLOGY OC_DIMENSION INPUT_BUFFER_SIZE INPUT_BUFFER_WIDTH WEIGHT_BUFFER_SIZE WEIGHT_BUFFER_WIDTH ACCUM_BUFFER_DATATYPE ACCUM_BUFFER_SIZE ACCUM_DATATYPE_WIDTH ACC_BUF_C_DATA_REP_NAME SUPPORT_MX DOUBLE_BUFFERED_ACCUM_BUFFER

  # Input double buffer
  set double_buffer "DoubleBuffer<$INPUT_BUFFER_SIZE,$INPUT_BUFFER_WIDTH>"
  set double_buffer_stripped [string map {" " ""} $double_buffer]

  directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem0Run/mem0Run/mem0 -WORD_WIDTH $INPUT_BUFFER_WIDTH
  directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem1Run/mem1Run/mem1 -WORD_WIDTH $INPUT_BUFFER_WIDTH

  if {$TECHNOLOGY != "generic" && $TECHNOLOGY != "tsmc40"} {
    directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem0Run/mem0Run/mem0:rsc -MAP_TO_MODULE [get_memory_name 1 $INPUT_BUFFER_SIZE $INPUT_BUFFER_WIDTH]
    directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem1Run/mem1Run/mem1:rsc -MAP_TO_MODULE [get_memory_name 1 $INPUT_BUFFER_SIZE $INPUT_BUFFER_WIDTH]
  }

  # Weight double buffer
  set double_buffer "DoubleBuffer<$WEIGHT_BUFFER_SIZE,$WEIGHT_BUFFER_WIDTH>"
  set double_buffer_stripped [string map {" " ""} $double_buffer]

    directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem0Run/mem0Run/mem0 -WORD_WIDTH $WEIGHT_BUFFER_WIDTH
    directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem1Run/mem1Run/mem1 -WORD_WIDTH $WEIGHT_BUFFER_WIDTH

    if {$TECHNOLOGY != "generic" && $TECHNOLOGY != "tsmc40"} {
      directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem0Run/mem0Run/mem0:rsc -MAP_TO_MODULE [get_memory_name 1 $WEIGHT_BUFFER_SIZE $WEIGHT_BUFFER_WIDTH]
      directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem1Run/mem1Run/mem1:rsc -MAP_TO_MODULE [get_memory_name 1 $WEIGHT_BUFFER_SIZE $WEIGHT_BUFFER_WIDTH]
    }

  if {$SUPPORT_MX == true} {
    global SCALE_DATATYPE SCALE_DATATYPE_WIDTH

    set scale_width [expr $SCALE_DATATYPE_WIDTH*$OC_DIMENSION]

    set weight_scale_double_buffer "DoubleBuffer<$WEIGHT_BUFFER_SIZE,$scale_width>"
    directive set /Accelerator/$weight_scale_double_buffer/$weight_scale_double_buffer:mem0Run/mem0Run/mem0 -WORD_WIDTH $scale_width
    directive set /Accelerator/$weight_scale_double_buffer/$weight_scale_double_buffer:mem1Run/mem1Run/mem1 -WORD_WIDTH $scale_width
  }

  set accumulation_buffer "DualPortBuffer<Pack1D<$ACCUM_BUFFER_DATATYPE,${OC_DIMENSION}UL>,$ACCUM_BUFFER_SIZE>"
  set accumulation_buffer_stripped [string map {" " ""} $accumulation_buffer]
  set memory_width [expr $OC_DIMENSION*$ACCUM_DATATYPE_WIDTH]
  directive set /Accelerator/$accumulation_buffer_stripped/bank0_run/bank0.value.$ACC_BUF_C_DATA_REP_NAME -WORD_WIDTH $memory_width
  if {$DOUBLE_BUFFERED_ACCUM_BUFFER == true} {
    directive set /Accelerator/$accumulation_buffer_stripped/bank1_run/bank1.value.$ACC_BUF_C_DATA_REP_NAME -WORD_WIDTH $memory_width
  }
}

proc pre_extract {} {
  global DOUBLE_BUFFERED_ACCUM_BUFFER
  ignore_memory_precedences -from WRITE_BANK_0* -to READ_BANK_0*
  if {$DOUBLE_BUFFERED_ACCUM_BUFFER == true} {
    ignore_memory_precedences -from WRITE_BANK_1* -to READ_BANK_1*
  }
}
