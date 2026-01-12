set block "Accelerator"
set full_block_name "Accelerator"

# ==============================================================================
# Configuration Source
# ==============================================================================
# This proc constructs the configuration for all blocks ONCE.
# It returns a list of dictionaries, where each dict contains:
#   - name: The base name (e.g., InputController)
#   - template: The full C++ template string
# ==============================================================================
proc get_accelerator_config {} {
  global INPUT_TYPE_LIST WEIGHT_TYPE_LIST SA_INPUT_TYPE SA_WEIGHT_TYPE \
          ACCUM_DATATYPE ACCUM_BUFFER_DATATYPE VECTOR_DATATYPE SCALE_DATATYPE \
          IC_DIMENSION OC_DIMENSION VECTOR_UNIT_WIDTH SUPPORT_MX SUPPORT_MVM \
          SUPPORT_SPMM SUPPORT_DWC IC_PORT_WIDTH OC_PORT_WIDTH ACCUM_BUFFER_SIZE \
          INPUT_BUFFER_WIDTH WEIGHT_BUFFER_WIDTH MV_UNIT_WIDTH SPMM_UNIT_WIDTH \
          SPMM_META_DATATYPE DWC_DATATYPE DWC_PSUM

  set config_list [list]

  # --- Standard Blocks ---
  lappend config_list [dict create \
    name "InputController" \
    template "InputController<InputTypeList, $IC_DIMENSION, $IC_PORT_WIDTH, $INPUT_BUFFER_WIDTH>" \
  ]

  lappend config_list [dict create \
    name "WeightController" \
    template "WeightController<WeightTypeList, $ACCUM_BUFFER_DATATYPE, $IC_DIMENSION, $OC_DIMENSION, $OC_PORT_WIDTH, $WEIGHT_BUFFER_WIDTH>" \
  ]

  lappend config_list [dict create \
    name "MatrixProcessor" \
    template "MatrixProcessor<InputTypeList, WeightTypeList, $SA_INPUT_TYPE, $SA_WEIGHT_TYPE, $ACCUM_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SCALE_DATATYPE, $IC_DIMENSION, $OC_DIMENSION, $ACCUM_BUFFER_SIZE>" \
  ]

  lappend config_list [dict create \
    name "MatrixParamsDeserializer" \
    template "MatrixParamsDeserializer<0, [expr {$SUPPORT_MX ? 5 : 3}]>" \
  ]

  lappend config_list [dict create \
    name "VectorUnit" \
    template "VectorUnit<$VECTOR_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SCALE_DATATYPE, $VECTOR_UNIT_WIDTH, $OC_DIMENSION>" \
  ]

  # --- Conditional Blocks ---
  if {$SUPPORT_MVM == true} {
    lappend config_list [dict create \
      name "MatrixVectorUnit" \
      template "MatrixVectorUnit<InputTypeList, WeightTypeList, $SA_INPUT_TYPE, $SA_WEIGHT_TYPE, $ACCUM_DATATYPE, $VECTOR_DATATYPE, $SCALE_DATATYPE, $OC_PORT_WIDTH, $MV_UNIT_WIDTH, $IC_DIMENSION, $VECTOR_UNIT_WIDTH>" \
    ]
  }

  if {$SUPPORT_SPMM == true} {
    lappend config_list [dict create \
      name "SpMMUnit" \
      template "SpMMUnit<WeightTypeList, $VECTOR_DATATYPE, $SA_WEIGHT_TYPE, $SPMM_META_DATATYPE, $VECTOR_DATATYPE, $SCALE_DATATYPE, $OC_PORT_WIDTH, $SPMM_UNIT_WIDTH, $IC_DIMENSION, $VECTOR_UNIT_WIDTH>" \
    ]
  }

  if {$SUPPORT_DWC == true} {
    lappend config_list [dict create \
      name "DwCUnit" \
      template "DwCUnit<$DWC_DATATYPE, $DWC_DATATYPE, $DWC_PSUM, $ACCUM_BUFFER_DATATYPE, $OC_DIMENSION, $DWC_DATATYPE>" \
    ]
  }

  return $config_list
}

# ==============================================================================
# Pre-Compile
# ==============================================================================
proc pre_compile {} {
  foreach item [get_accelerator_config] {
    set templ [dict get $item template]
    solution design set $templ -mapped
  }
}

# ==============================================================================
# Pre-Libraries
# ==============================================================================
proc pre_libraries {} {
  foreach item [get_accelerator_config] {
    set name [dict get $item name]

    # Safe string construction for [Block] Name.v1
    set lib_name [format {[Block] %s.v1} $name]

    # Quotes ensure it is treated as a single argument
    solution library add "$lib_name"
  }
}

# ==============================================================================
# Pre-Assembly
# ==============================================================================
proc pre_assembly {} {
  foreach item [get_accelerator_config] {
    set templ [dict get $item template]
    set name [dict get $item name]

    # Strip spaces from C++ template for hierarchical path mapping
    set stripped_templ [string map {" " ""} $templ]

    # Safe string construction
    set lib_name [format {[Block] %s.v1} $name]

    directive set /Accelerator/$stripped_templ -MAP_TO_MODULE "$lib_name"
  }
}

# ==============================================================================
# Helper: Configure Double Buffer
# Handles: Space stripping, Word Width setting, and Module Mapping
# ==============================================================================
proc configure_double_buffer {template_name size width tech} {
  # 1. Clean the name (remove spaces) for the path
  set stripped_name [string map {" " ""} $template_name]
  set base_path "/Accelerator/$stripped_name/$stripped_name"

  # 2. Set Word Widths
  directive set ${base_path}:mem0_run/mem0_run/mem0 -WORD_WIDTH $width
  directive set ${base_path}:mem1_run/mem1_run/mem1 -WORD_WIDTH $width

  # 3. Map to Library (only for specific technologies)
  if {$tech != "generic" && $tech != "tsmc40" && $size > 32} {
    # 1 = Single Port Memory
    set mem_lib [get_memory_name 1 $size $width]

    directive set ${base_path}:mem0_run/mem0_run/mem0:rsc -MAP_TO_MODULE $mem_lib
    directive set ${base_path}:mem1_run/mem1_run/mem1:rsc -MAP_TO_MODULE $mem_lib
  }
}

# ==============================================================================
# Pre-Architect
# ==============================================================================
proc pre_architect {} {
  global TECHNOLOGY IC_DIMENSION OC_DIMENSION INPUT_BUFFER_SIZE \
         INPUT_BUFFER_WIDTH WEIGHT_BUFFER_SIZE WEIGHT_BUFFER_WIDTH \
         ACCUM_BUFFER_DATATYPE ACCUM_BUFFER_SIZE ACCUM_DATATYPE_WIDTH \
         ACC_BUF_C_DATA_REP_NAME SUPPORT_MX DOUBLE_BUFFERED_ACCUM_BUFFER \
         SCALE_DATATYPE_WIDTH

  # --- 1. Standard Double Buffers ---

  # Input Buffer
  configure_double_buffer \
    "DoubleBuffer<$INPUT_BUFFER_SIZE,$INPUT_BUFFER_WIDTH>" \
    $INPUT_BUFFER_SIZE $INPUT_BUFFER_WIDTH $TECHNOLOGY

  # Weight Buffer
  configure_double_buffer \
    "DoubleBuffer<$WEIGHT_BUFFER_SIZE,$WEIGHT_BUFFER_WIDTH>" \
    $WEIGHT_BUFFER_SIZE $WEIGHT_BUFFER_WIDTH $TECHNOLOGY

  # --- 2. MX Scaling Factors Buffers ---
  if {$SUPPORT_MX == true} {
    # Input Scale Buffer
    configure_double_buffer \
      "DoubleBuffer<$INPUT_BUFFER_SIZE,$SCALE_DATATYPE_WIDTH>" \
      $INPUT_BUFFER_SIZE $SCALE_DATATYPE_WIDTH $TECHNOLOGY

    # Weight Scale Buffer
    set ws_size  [expr {$WEIGHT_BUFFER_SIZE / $IC_DIMENSION}]
    set ws_width [expr $SCALE_DATATYPE_WIDTH * $OC_DIMENSION]

    configure_double_buffer \
      "DoubleBuffer<$ws_size,$ws_width>" \
      $ws_size $ws_width $TECHNOLOGY
  }

  # --- 3. Accumulation Buffer ---
  set accum_templ "DualPortBuffer<Pack1D<$ACCUM_BUFFER_DATATYPE,${OC_DIMENSION}UL>,$ACCUM_BUFFER_SIZE>"
  set accum_stripped [string map {" " ""} $accum_templ]
  set accum_width [expr $OC_DIMENSION * $ACCUM_DATATYPE_WIDTH]

  # Define the path to the memory resource
  set bank0_path "/Accelerator/$accum_stripped/bank0_run/bank0.value.$ACC_BUF_C_DATA_REP_NAME"
  set bank1_path "/Accelerator/$accum_stripped/bank1_run/bank1.value.$ACC_BUF_C_DATA_REP_NAME"

  # Configure Bank 0
  directive set $bank0_path -WORD_WIDTH $accum_width

  # Configure Bank 1 (Optional)
  if {$DOUBLE_BUFFERED_ACCUM_BUFFER == true} {
    directive set $bank1_path -WORD_WIDTH $accum_width
  }

  # Map Libraries
  if {$TECHNOLOGY != "generic" && $TECHNOLOGY != "tsmc40"} {
    set mem_lib [get_memory_name 0 $ACCUM_BUFFER_SIZE $accum_width]

    directive set ${bank0_path}:rsc -MAP_TO_MODULE $mem_lib

    if {$DOUBLE_BUFFERED_ACCUM_BUFFER == true} {
      directive set ${bank1_path}:rsc -MAP_TO_MODULE $mem_lib
    }
  }
}

# ==============================================================================
# Pre-Extract
# ==============================================================================
proc pre_extract {} {
  global DOUBLE_BUFFERED_ACCUM_BUFFER
  ignore_memory_precedences -from WRITE_BANK_0* -to READ_BANK_0*
  if {$DOUBLE_BUFFERED_ACCUM_BUFFER == true} {
    ignore_memory_precedences -from WRITE_BANK_1* -to READ_BANK_1*
  }
}
