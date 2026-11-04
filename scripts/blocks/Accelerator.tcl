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
          IC_DIMENSION OC_DIMENSION VECTOR_UNIT_WIDTH REDUCER_WIDTH \
          ACCUMULATOR_WIDTH SUPPORT_MX SUPPORT_MVM SUPPORT_SPMM SUPPORT_DWC \
          IC_PORT_WIDTH OC_PORT_WIDTH ACCUM_BUFFER_SIZE INPUT_BUFFER_WIDTH \
          WEIGHT_BUFFER_WIDTH MV_UNIT_WIDTH SPMM_UNIT_WIDTH SPMM_META_DATATYPE \
          DWC_DATATYPE DWC_PSUM

  # --- Standard Blocks ---
  set config_list [list [dict create name MatrixUnit template MatrixUnit]]

  lappend config_list [dict create \
    name "VectorUnit" \
    template "VectorUnit<$VECTOR_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SCALE_DATATYPE, $VECTOR_UNIT_WIDTH, $REDUCER_WIDTH, $ACCUMULATOR_WIDTH, $OC_DIMENSION, $OC_PORT_WIDTH>" \
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
    solution library add [format {[Block] %s.v1} [dict get $item name]]
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
