set block MatrixUnit
set full_block_name MatrixUnit

# Child block configuration
proc get_matrix_unit_config {} {
  global IC_DIMENSION OC_DIMENSION IC_PORT_WIDTH OC_PORT_WIDTH \
         INPUT_BUFFER_WIDTH WEIGHT_BUFFER_WIDTH ACCUM_BUFFER_DATATYPE \
         SA_INPUT_TYPE SA_WEIGHT_TYPE ACCUM_DATATYPE SCALE_DATATYPE \
         ACCUM_BUFFER_SIZE SUPPORT_MX MATRIX_BACKEND MATRIX_BACKEND_CIM
  set config [list [dict create name InputController \
    template "InputController<InputTypeList, $IC_DIMENSION, $IC_PORT_WIDTH, $INPUT_BUFFER_WIDTH>"]]
  if {$MATRIX_BACKEND == $MATRIX_BACKEND_CIM} {
    lappend config [dict create name CIMWeightController \
      template [cim_weight_controller_template]]
    lappend config [dict create name CIMProcessor template [cim_processor_template]]
  } else {
    lappend config [dict create name WeightController \
      template "WeightController<WeightTypeList, $ACCUM_BUFFER_DATATYPE, $IC_DIMENSION, $OC_DIMENSION, $OC_PORT_WIDTH, $WEIGHT_BUFFER_WIDTH>"]
    lappend config [dict create name MatrixProcessor \
      template "MatrixProcessor<InputTypeList, WeightTypeList, $SA_INPUT_TYPE, $SA_WEIGHT_TYPE, $ACCUM_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SCALE_DATATYPE, $IC_DIMENSION, $OC_DIMENSION, $ACCUM_BUFFER_SIZE>"]
  }
  lappend config [dict create name MatrixParamsDeserializer \
    template "MatrixParamsDeserializer<0, [expr {$SUPPORT_MX ? 6 : 4}]>"]
  return $config
}

proc pre_compile {} {
  foreach item [get_matrix_unit_config] {
    solution design set [dict get $item template] -mapped
  }
}

proc pre_libraries {} {
  foreach item [get_matrix_unit_config] {
    solution library add [format {[Block] %s.v1} [dict get $item name]]
  }
}

proc pre_assembly {} {
  foreach item [get_matrix_unit_config] {
    set name [string map {" " ""} [dict get $item template]]
    directive set /MatrixUnit/$name -MAP_TO_MODULE \
      [format {[Block] %s.v1} [dict get $item name]]
  }
}

# MatrixUnit SRAM configuration
proc configure_double_buffer {template_name size width technology} {
  set name [string map {" " ""} $template_name]
  set base_path "/MatrixUnit/$name/$name"
  directive set ${base_path}:mem0_run/mem0_run/mem0 -WORD_WIDTH $width
  directive set ${base_path}:mem1_run/mem1_run/mem1 -WORD_WIDTH $width
  if {$technology != "generic" && $technology != "tsmc40" && $size > 32} {
    set memory_library [get_memory_name 1 $size $width]
    directive set ${base_path}:mem0_run/mem0_run/mem0:rsc -MAP_TO_MODULE $memory_library
    directive set ${base_path}:mem1_run/mem1_run/mem1:rsc -MAP_TO_MODULE $memory_library
  }
}

proc pre_architect {} {
  global TECHNOLOGY IC_DIMENSION OC_DIMENSION INPUT_BUFFER_SIZE \
         INPUT_BUFFER_WIDTH WEIGHT_BUFFER_SIZE WEIGHT_BUFFER_WIDTH \
         ACCUM_BUFFER_DATATYPE ACCUM_BUFFER_SIZE ACCUM_DATATYPE_WIDTH \
         ACC_BUF_C_DATA_REP_NAME SUPPORT_MX DOUBLE_BUFFERED_ACCUM_BUFFER \
         SCALE_DATATYPE_WIDTH MATRIX_BACKEND MATRIX_BACKEND_CIM
  configure_double_buffer \
    "DoubleBuffer<$INPUT_BUFFER_SIZE,$INPUT_BUFFER_WIDTH>" \
    $INPUT_BUFFER_SIZE $INPUT_BUFFER_WIDTH $TECHNOLOGY

  if {$MATRIX_BACKEND != $MATRIX_BACKEND_CIM} {
    configure_double_buffer \
      "DoubleBuffer<$WEIGHT_BUFFER_SIZE,$WEIGHT_BUFFER_WIDTH>" \
      $WEIGHT_BUFFER_SIZE $WEIGHT_BUFFER_WIDTH $TECHNOLOGY
  }
  if {$SUPPORT_MX} {
    configure_double_buffer \
      "DoubleBuffer<$INPUT_BUFFER_SIZE,$SCALE_DATATYPE_WIDTH>" \
      $INPUT_BUFFER_SIZE $SCALE_DATATYPE_WIDTH $TECHNOLOGY
    set weight_scale_size [expr {$WEIGHT_BUFFER_SIZE / $IC_DIMENSION}]
    set weight_scale_width [expr {$SCALE_DATATYPE_WIDTH * $OC_DIMENSION}]
    configure_double_buffer \
      "DoubleBuffer<$weight_scale_size,$weight_scale_width>" \
      $weight_scale_size $weight_scale_width $TECHNOLOGY
  }

  set accum_template "DualPortBuffer<Pack1D<$ACCUM_BUFFER_DATATYPE,${OC_DIMENSION}UL>,$ACCUM_BUFFER_SIZE>"
  set accum_name [string map {" " ""} $accum_template]
  set accum_width [expr {$OC_DIMENSION * $ACCUM_DATATYPE_WIDTH}]
  set banks {bank0}
  if {$DOUBLE_BUFFERED_ACCUM_BUFFER} { lappend banks bank1 }
  foreach bank $banks {
    # Each bank is owned by its combined access process.
    set path "/MatrixUnit/$accum_name/${bank}_run/$bank.value.$ACC_BUF_C_DATA_REP_NAME"
    directive set $path -WORD_WIDTH $accum_width
    if {$TECHNOLOGY != "generic" && $TECHNOLOGY != "tsmc40"} {
      directive set ${path}:rsc -MAP_TO_MODULE \
        [get_memory_name 0 $ACCUM_BUFFER_SIZE $accum_width]
    }
  }
}

proc pre_extract {} {
  global DOUBLE_BUFFERED_ACCUM_BUFFER
  ignore_memory_precedences -from WRITE_BANK_0* -to READ_BANK_0*
  if {$DOUBLE_BUFFERED_ACCUM_BUFFER == true} {
    ignore_memory_precedences -from WRITE_BANK_1* -to READ_BANK_1*
  }
}
