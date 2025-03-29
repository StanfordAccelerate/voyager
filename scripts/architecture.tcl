proc is_floating_point {datatype} {
  return [expr {[string match {F*} $datatype] || [string match {P*} $datatype]}]
}

if { $DATATYPE == "P8_1" } {
  set IO_DATATYPE "P8"
  set ACCUM_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "Posit<8, 1>::decoded"
  set PE_WEIGHT_DATATYPE "Posit<8, 1>::decoded"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "E4M3" } {
  set IO_DATATYPE "F8"
  set ACCUM_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "StdFloat<3, 4, false, true, AC_RND_CONV>::decoded"
  set PE_WEIGHT_DATATYPE "StdFloat<3, 4, false, true, AC_RND_CONV>::decoded"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "E4M3_NS" } {
  set IO_DATATYPE "F8"
  set ACCUM_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "StdFloat<3, 4, false, false, AC_RND_CONV>::decoded"
  set PE_WEIGHT_DATATYPE "StdFloat<3, 4, false, false, AC_RND_CONV>::decoded"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "E4M3_DW" } {
  set IO_DATATYPE "F8"
  set ACCUM_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "StdFloat<3, 4, true, true, AC_RND_CONV>::decoded"
  set PE_WEIGHT_DATATYPE "StdFloat<3, 4, true, true, AC_RND_CONV>::decoded"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "E4M3_DW_NS" } {
  set IO_DATATYPE "F8"
  set ACCUM_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "StdFloat<3, 4, true, false, AC_RND_CONV>::decoded"
  set PE_WEIGHT_DATATYPE "StdFloat<3, 4, true, false, AC_RND_CONV>::decoded"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "E5M2" } {
  set IO_DATATYPE "F8"
  set ACCUM_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "StdFloat<2, 5, false, true, AC_RND_CONV>::decoded"
  set PE_WEIGHT_DATATYPE "StdFloat<2, 5, false, true, AC_RND_CONV>::decoded"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "HYBRID_FP8" } {
  set IO_DATATYPE "F8"
  set ACCUM_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "StdFloat<3, 5, false, true, AC_RND_CONV>::decoded"
  set PE_WEIGHT_DATATYPE "StdFloat<3, 5, false, true, AC_RND_CONV>::decoded"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "BF16" } {
  set IO_DATATYPE "F16"
  set ACCUM_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "StdFloat<7, 8, false, true, AC_RND_CONV>::decoded"
  set PE_WEIGHT_DATATYPE "StdFloat<7, 8, false, true, AC_RND_CONV>::decoded"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 16
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "BF16_NS" } {
  set IO_DATATYPE "F16"
  set ACCUM_DATATYPE "F32"
  set VECTOR_DATATYPE "F32"
  set PE_INPUT_DATATYPE "StdFloat<7, 8, false, false, AC_RND_CONV>::decoded"
  set PE_WEIGHT_DATATYPE "StdFloat<7, 8, false, false, AC_RND_CONV>::decoded"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 16
  set ACCUM_DATATYPE_WIDTH 32
} elseif { $DATATYPE == "BF16_TRN" } {
  set IO_DATATYPE "F16"
  set ACCUM_DATATYPE "F32"
  set VECTOR_DATATYPE "F32"
  set PE_INPUT_DATATYPE "StdFloat<7, 8, false, true, AC_TRN_ZERO>::decoded"
  set PE_WEIGHT_DATATYPE "StdFloat<7, 8, false, true, AC_TRN_ZERO>::decoded"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 16
  set ACCUM_DATATYPE_WIDTH 32
} elseif { $DATATYPE == "BF16_NS_TRN" } {
  set IO_DATATYPE "F16"
  set ACCUM_DATATYPE "F32"
  set VECTOR_DATATYPE "F32"
  set PE_INPUT_DATATYPE "StdFloat<7, 8, false, false, AC_TRN_ZERO>::decoded"
  set PE_WEIGHT_DATATYPE "StdFloat<7, 8, false, false, AC_TRN_ZERO>::decoded"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 16
  set ACCUM_DATATYPE_WIDTH 32
} elseif { $DATATYPE == "BF16_ONLY" } {
  set IO_DATATYPE "F16"
  set ACCUM_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "StdFloat<7, 8, false, true, AC_RND_CONV>::decoded"
  set PE_WEIGHT_DATATYPE "StdFloat<7, 8, false, true, AC_RND_CONV>::decoded"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 16
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "BF16_ONLY_NS" } {
  set IO_DATATYPE "F16"
  set ACCUM_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "StdFloat<7, 8, false, false, AC_RND_CONV>::decoded"
  set PE_WEIGHT_DATATYPE "StdFloat<7, 8, false, false, AC_RND_CONV>::decoded"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 16
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "BF16_ONLY_TRN" } {
  set IO_DATATYPE "F16"
  set ACCUM_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "StdFloat<7, 8, false, true, AC_TRN_ZERO>::decoded"
  set PE_WEIGHT_DATATYPE "StdFloat<7, 8, false, true, AC_TRN_ZERO>::decoded"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 16
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "BF16_ONLY_NS_TRN" } {
  set IO_DATATYPE "F16"
  set ACCUM_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "StdFloat<7, 8, false, false, AC_TRN_ZERO>::decoded"
  set PE_WEIGHT_DATATYPE "StdFloat<7, 8, false, false, AC_TRN_ZERO>::decoded"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 16
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "FP32" } {
  set IO_DATATYPE "F32"
  set ACCUM_DATATYPE "F32"
  set VECTOR_DATATYPE "F32"
  set PE_INPUT_DATATYPE "StdFloat<23, 8, false, true, AC_RND_CONV>::decoded"
  set PE_WEIGHT_DATATYPE "StdFloat<23, 8, false, true, AC_RND_CONV>::decoded"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 32
  set ACCUM_DATATYPE_WIDTH 32
} elseif { $DATATYPE == "INT8" } {
  set IO_DATATYPE "DataTypes::int8"
  set ACCUM_DATATYPE "DataTypes::int24"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "Int<8, true>::decoded"
  set PE_WEIGHT_DATATYPE "Int<8, true>::decoded"

  set ACC_BUF_C_DATA_REP_NAME "int_val"

  set IO_DATATYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 24
} elseif { $DATATYPE == "INT8_32" } {
  set IO_DATATYPE "DataTypes::int8"
  set ACCUM_DATATYPE "DataTypes::int32"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "Int<8, true>::decoded"
  set PE_WEIGHT_DATATYPE "Int<8, true>::decoded"

  set ACC_BUF_C_DATA_REP_NAME "int_val"

  set IO_DATATYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 32
} elseif {$DATATYPE == "MXINT8"} {
  set IO_DATATYPE "DataTypes::int8"
  set ACCUM_DATATYPE "DataTypes::int32"
  set ACCUM_BUFFER_DATATYPE "DataTypes::bfloat16"
  set VECTOR_DATATYPE "DataTypes::bfloat16"
  set SCALE_DATATYPE "DataTypes::fp8_e8m0"

  set PE_INPUT_DATATYPE "Int<8, true>::decoded"
  set PE_WEIGHT_DATATYPE "Int<8, true>::decoded"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set SUPPORT_MX true
  set IO_DATATYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
  set SCALE_DATATYPE_WIDTH 8
} elseif {$DATATYPE == "MXNF4"} {
  set IO_DATATYPE "DataTypes::nf4"
  set ACCUM_DATATYPE "DataTypes::int32"
  set ACCUM_BUFFER_DATATYPE "DataTypes::bfloat16"
  set VECTOR_DATATYPE "DataTypes::bfloat16"
  set SCALE_DATATYPE "DataTypes::fp8_e5m3"

  set PE_INPUT_DATATYPE "NormalFloat4::decoded"
  set PE_WEIGHT_DATATYPE "NormalFloat4::decoded"

  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set SUPPORT_MX true
  set IO_DATATYPE_WIDTH 4
  set ACCUM_DATATYPE_WIDTH 16
  set SCALE_DATATYPE_WIDTH 8
} else {
    puts "Invalid DATATYPE"
    exit 1
}

if {![info exists ACCUM_BUFFER_DATATYPE]} {
  set ACCUM_BUFFER_DATATYPE $ACCUM_DATATYPE
}

if {![info exists SCALE_DATATYPE]} {
  set SCALE_DATATYPE "DataTypes::fp8_e8m0"
}

if {![info exists SUPPORT_MX]} {
  set SUPPORT_MX false
}

if {!$SUPPORT_MX} {
  set OUTPUT_DATATYPES "$IO_DATATYPE, $VECTOR_DATATYPE"
  set VECTOR_INPUT_DATATYPES "$IO_DATATYPE, $VECTOR_DATATYPE"
} else {
  set OUTPUT_DATATYPES "$IO_DATATYPE, $VECTOR_DATATYPE, $SCALE_DATATYPE"
  set VECTOR_INPUT_DATATYPES "$IO_DATATYPE, $VECTOR_DATATYPE, $SCALE_DATATYPE"
}

set IC_PORT_WIDTH [expr $IC_DIMENSION*$IO_DATATYPE_WIDTH]
set OC_PORT_WIDTH [expr $OC_DIMENSION*$IO_DATATYPE_WIDTH]
set ADDRESS_WIDTH 64
