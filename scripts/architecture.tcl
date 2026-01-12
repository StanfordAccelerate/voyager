if { $DATATYPE == "P8_1" } {
  set INPUT_DATATYPE "DataTypes::posit8"
  set WEIGHT_DATATYPE "DataTypes::posit8"
  set ACCUM_DATATYPE "DataTypes::bfloat16"
  set VECTOR_DATATYPE "DataTypes::bfloat16"

  set SA_INPUT_TYPE "Posit<8, 1>::decoded"
  set SA_WEIGHT_TYPE "Posit<8, 1>::decoded"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set INPUT_DTYPE_WIDTH 8
  set WEIGHT_DTYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "E4M3" } {
  set INPUT_DATATYPE "DataTypes::e4m3"
  set WEIGHT_DATATYPE "DataTypes::e4m3"
  set ACCUM_DATATYPE "DataTypes::bfloat16"
  set VECTOR_DATATYPE "DataTypes::bfloat16"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set INPUT_DTYPE_WIDTH 8
  set WEIGHT_DTYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "E4M3_NS" } {
  set INPUT_DATATYPE "F8"
  set WEIGHT_DATATYPE "F8"
  set ACCUM_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set INPUT_DTYPE_WIDTH 8
  set WEIGHT_DTYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "E4M3_DW" } {
  set INPUT_DATATYPE "F8"
  set WEIGHT_DATATYPE "F8"
  set ACCUM_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set INPUT_DTYPE_WIDTH 8
  set WEIGHT_DTYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "E4M3_DW_NS" } {
  set INPUT_DATATYPE "F8"
  set WEIGHT_DATATYPE "F8"
  set ACCUM_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set INPUT_DTYPE_WIDTH 8
  set WEIGHT_DTYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "E5M2" } {
  set INPUT_DATATYPE "DataTypes::e5m2"
  set WEIGHT_DATATYPE "DataTypes::e5m2"
  set ACCUM_DATATYPE "DataTypes::bfloat16"
  set VECTOR_DATATYPE "DataTypes::bfloat16"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set INPUT_DTYPE_WIDTH 8
  set WEIGHT_DTYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "HYBRID_FP8" } {
  set INPUT_DATATYPE "F8"
  set WEIGHT_DATATYPE "F8"
  set ACCUM_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set INPUT_DTYPE_WIDTH 8
  set WEIGHT_DTYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "BF16" } {
  set INPUT_DATATYPE "DataTypes::bfloat16"
  set WEIGHT_DATATYPE "DataTypes::bfloat16"
  set ACCUM_DATATYPE "DataTypes::bfloat16"
  set VECTOR_DATATYPE "DataTypes::bfloat16"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 16
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "FP32" } {
  set INPUT_DATATYPE "DataTypes::float32"
  set WEIGHT_DATATYPE "DataTypes::float32"
  set ACCUM_DATATYPE "DataTypes::float32"
  set VECTOR_DATATYPE "DataTypes::float32"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set INPUT_DTYPE_WIDTH 32
  set WEIGHT_DTYPE_WIDTH 32
  set ACCUM_DATATYPE_WIDTH 32
} elseif { $DATATYPE == "INT8" } {
  set INPUT_DATATYPE "DataTypes::int8"
  set WEIGHT_DATATYPE "DataTypes::int8"
  set ACCUM_DATATYPE "DataTypes::int24"
  set VECTOR_DATATYPE "DataTypes::bfloat16"

  set ACC_BUF_C_DATA_REP_NAME "int_val"

  set INPUT_DTYPE_WIDTH 8
  set WEIGHT_DTYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 24
} elseif { $DATATYPE == "INT8_32" } {
  set INPUT_DATATYPE "DataTypes::int8"
  set WEIGHT_DATATYPE "DataTypes::int8"
  set ACCUM_DATATYPE "DataTypes::int32"
  set VECTOR_DATATYPE "DataTypes::bfloat16"

  set ACC_BUF_C_DATA_REP_NAME "int_val"

  set INPUT_DTYPE_WIDTH 8
  set WEIGHT_DTYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 32
} elseif {$DATATYPE == "MXINT8"} {
  set INPUT_DATATYPE "DataTypes::int8"
  set WEIGHT_DATATYPE "DataTypes::int8"
  set ACCUM_DATATYPE "DataTypes::int32"
  set ACCUM_BUFFER_DATATYPE "DataTypes::bfloat16"
  set VECTOR_DATATYPE "DataTypes::bfloat16"
  set SCALE_DATATYPE "DataTypes::fp8_e8m0"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set SUPPORT_MX true

  set INPUT_DTYPE_WIDTH 8
  set WEIGHT_DTYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
  set SCALE_DATATYPE_WIDTH 8
} elseif {$DATATYPE == "MXNF4"} {
  set INPUT_DATATYPE "DataTypes::uint2, DataTypes::int4, DataTypes::int6"
  set WEIGHT_DATATYPE "DataTypes::uint2, DataTypes::int4, DataTypes::int6"
  set ACCUM_DATATYPE "DataTypes::int18"
  set ACCUM_BUFFER_DATATYPE "DataTypes::bfloat16"
  set VECTOR_DATATYPE "DataTypes::bfloat16"
  set SCALE_DATATYPE "DataTypes::fp8_e5m3"
  set VU_INPUT_TYPES "$VECTOR_DATATYPE, $SCALE_DATATYPE, DataTypes::e4m3, DataTypes::int1"
  set OUTPUT_DATATYPES "$INPUT_DATATYPE, $VU_INPUT_TYPES"
  set SPMM_META_DATATYPE "DataTypes::int32"

  set SA_INPUT_TYPE "DataTypes::int6"
  set SA_WEIGHT_TYPE "DataTypes::int6"

  set SUPPORT_MX true

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set INPUT_DTYPE_WIDTH 6
  set WEIGHT_DTYPE_WIDTH 6
  set ACCUM_DATATYPE_WIDTH 16
  set SCALE_DATATYPE_WIDTH 8

  set IC_PORT_WIDTH [expr {$IC_DIMENSION * 4}]
  set OC_PORT_WIDTH [expr {$OC_DIMENSION * 4}]
  set VECTOR_UNIT_WIDTH $OC_DIMENSION
  set MV_UNIT_WIDTH [expr {$OC_DIMENSION * 2}]
  set SPMM_UNIT_WIDTH $OC_DIMENSION
} else {
  puts "Invalid DATATYPE"
  exit 1
}

if {![info exists SUPPORT_MX]} {
  set SUPPORT_MX false
}

if {![info exists SUPPORT_MVM]} {
  set SUPPORT_MVM false
}

if {![info exists SUPPORT_SPMM]} {
  set SUPPORT_SPMM false
}

if {![info exists SUPPORT_DWC]} {
  set SUPPORT_DWC false
}

if {![info exists VECTOR_UNIT_WIDTH]} {
  set VECTOR_UNIT_WIDTH $OC_DIMENSION
}

# ================================================================
# Default Datatypes
# ================================================================

if {![info exists SA_INPUT_TYPE]} {
  set SA_INPUT_TYPE $INPUT_DATATYPE
}

if {![info exists SA_WEIGHT_TYPE]} {
  set SA_WEIGHT_TYPE $WEIGHT_DATATYPE
}

if {![info exists ACCUM_BUFFER_DATATYPE]} {
  set ACCUM_BUFFER_DATATYPE $ACCUM_DATATYPE
}

if {![info exists SCALE_DATATYPE]} {
  set SCALE_DATATYPE "DataTypes::fp8_e8m0"
}

if {![info exists VU_INPUT_TYPES]} {
  if {$SUPPORT_MX} {
    set VU_INPUT_TYPES "$VECTOR_DATATYPE, $SCALE_DATATYPE"
  } else {
    set VU_INPUT_TYPES "$INPUT_DATATYPE, $VECTOR_DATATYPE"
  }
}

if {![info exists OUTPUT_DATATYPES]} {
  if {$SUPPORT_MX} {
    set OUTPUT_DATATYPES "$INPUT_DATATYPE, $VECTOR_DATATYPE, $SCALE_DATATYPE"
  } else {
    set OUTPUT_DATATYPES "$INPUT_DATATYPE, $VECTOR_DATATYPE"
  }
}

# ================================================================
# Datatype Width Configuration
# ================================================================

set INPUT_TYPE_LIST "std::tuple<$INPUT_DATATYPE>"
set WEIGHT_TYPE_LIST "std::tuple<$WEIGHT_DATATYPE>"

if {![info exists INPUT_DTYPE_WIDTH]} {
  set INPUT_DTYPE_WIDTH "${INPUT_DATATYPE}::width"
}

if {![info exists WEIGHT_DTYPE_WIDTH]} {
  set WEIGHT_DTYPE_WIDTH "${WEIGHT_DATATYPE}::width"
}

# ================================================================
# Port Width Definitions
# ================================================================

if {![info exists IC_PORT_WIDTH]} {
  set IC_PORT_WIDTH [expr {$IC_DIMENSION * $INPUT_DTYPE_WIDTH}]
}

if {![info exists OC_PORT_WIDTH]} {
  set OC_PORT_WIDTH [expr {$OC_DIMENSION * $WEIGHT_DTYPE_WIDTH}]
}

# ================================================================
# Buffer Configurations
# ================================================================

if {![info exists INPUT_BUFFER_SIZE]} {
  set INPUT_BUFFER_SIZE 1024
}

if {![info exists INPUT_BUFFER_WIDTH]} {
  set INPUT_BUFFER_WIDTH [expr {$IC_DIMENSION * $INPUT_DTYPE_WIDTH}]
}

if {![info exists WEIGHT_BUFFER_SIZE]} {
  set WEIGHT_BUFFER_SIZE 1024
}

if {![info exists WEIGHT_BUFFER_WIDTH]} {
  set WEIGHT_BUFFER_WIDTH [expr {$OC_DIMENSION * $WEIGHT_DTYPE_WIDTH}]
}

if {![info exists ACCUM_BUFFER_SIZE]} {
  set ACCUM_BUFFER_SIZE 1024
}

# ================================================================
# DwC Configurations
# ================================================================

if {$SUPPORT_DWC} {
  set DWC_WIDTH 40
  set UNROLLFACTOR $OC_DIMENSION
  set DWC_KERNEL_DIM 3
  set DWC_KERNEL_SIZE [expr $DWC_KERNEL_DIM * $DWC_KERNEL_DIM]
  set DWC_DATATYPE $INPUT_DATATYPE
  set DWC_PSUM $ACCUM_DATATYPE
}
