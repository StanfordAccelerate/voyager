proc is_floating_point {datatype} {
  return [expr {[string match {F*} $datatype] || [string match {P*} $datatype]}]
}

if { $DATATYPE == "P8_1" } {
  set IO_DATATYPE "P8"
  set ACCUM_DATATYPE "F16"
  set INTERMEDIATE_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "Posit<8, 1>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "Posit<8, 1>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "StdFloat<7, 8, false, true, AC_RND_CONV>::AccumulationDatatype"
  set C_DATA_REP_NAME "bits"
  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "E4M3" } {
  set IO_DATATYPE "F8"
  set ACCUM_DATATYPE "F16"
  set INTERMEDIATE_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "StdFloat<3, 4, false, true, AC_RND_CONV>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "StdFloat<3, 4, false, true, AC_RND_CONV>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "StdFloat<7, 8, false, true, AC_RND_CONV>::AccumulationDatatype"
  set C_DATA_REP_NAME "float_val.d"
  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "E4M3_NS" } {
  set IO_DATATYPE "F8"
  set ACCUM_DATATYPE "F16"
  set INTERMEDIATE_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "StdFloat<3, 4, false, false, AC_RND_CONV>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "StdFloat<3, 4, false, false, AC_RND_CONV>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "StdFloat<7, 8, false, false, AC_RND_CONV>::AccumulationDatatype"
  set C_DATA_REP_NAME "float_val.d"
  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "E4M3_DW" } {
  set IO_DATATYPE "F8"
  set ACCUM_DATATYPE "F16"
  set INTERMEDIATE_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "StdFloat<3, 4, true, true, AC_RND_CONV>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "StdFloat<3, 4, true, true, AC_RND_CONV>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "StdFloat<7, 8, true, true, AC_RND_CONV>::AccumulationDatatype"
  set C_DATA_REP_NAME "float_val.d"
  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "E4M3_DW_NS" } {
  set IO_DATATYPE "F8"
  set ACCUM_DATATYPE "F16"
  set INTERMEDIATE_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "StdFloat<3, 4, true, false, AC_RND_CONV>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "StdFloat<3, 4, true, false, AC_RND_CONV>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "StdFloat<7, 8, true, false, AC_RND_CONV>::AccumulationDatatype"
  set C_DATA_REP_NAME "float_val.d"
  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "E5M2" } {
  set IO_DATATYPE "F8"
  set ACCUM_DATATYPE "F16"
  set INTERMEDIATE_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "StdFloat<2, 5, false, true, AC_RND_CONV>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "StdFloat<2, 5, false, true, AC_RND_CONV>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "StdFloat<7, 8, false, true, AC_RND_CONV>::AccumulationDatatype"
  set C_DATA_REP_NAME "float_val.d"
  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "HYBRID_FP8" } {
  set IO_DATATYPE "F8"
  set ACCUM_DATATYPE "F16"
  set INTERMEDIATE_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "StdFloat<3, 5, false, true, AC_RND_CONV>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "StdFloat<3, 5, false, true, AC_RND_CONV>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "StdFloat<7, 8, false, true, AC_RND_CONV>::AccumulationDatatype"
  set C_DATA_REP_NAME "float_val.d"
  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "BF16" } {
  set IO_DATATYPE "F16"
  set ACCUM_DATATYPE "F16"
  set INTERMEDIATE_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "StdFloat<7, 8, false, true, AC_RND_CONV>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "StdFloat<7, 8, false, true, AC_RND_CONV>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "StdFloat<7, 8, false, true, AC_RND_CONV>::AccumulationDatatype"
  set C_DATA_REP_NAME "float_val.d"
  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 16
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "BF16_NS" } {
  set IO_DATATYPE "F16"
  set ACCUM_DATATYPE "F32"
  set INTERMEDIATE_DATATYPE "F32"
  set VECTOR_DATATYPE "F32"
  set PE_INPUT_DATATYPE "StdFloat<7, 8, false, false, AC_RND_CONV>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "StdFloat<7, 8, false, false, AC_RND_CONV>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "StdFloat<23, 8, false, false, AC_RND_CONV>::AccumulationDatatype"
  set C_DATA_REP_NAME "float_val.d"
  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 16
  set ACCUM_DATATYPE_WIDTH 32
} elseif { $DATATYPE == "BF16_TRN" } {
  set IO_DATATYPE "F16"
  set ACCUM_DATATYPE "F32"
  set INTERMEDIATE_DATATYPE "F32"
  set VECTOR_DATATYPE "F32"
  set PE_INPUT_DATATYPE "StdFloat<7, 8, false, true, AC_TRN_ZERO>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "StdFloat<7, 8, false, true, AC_TRN_ZERO>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "StdFloat<23, 8, false, true, AC_TRN_ZERO>::AccumulationDatatype"
  set C_DATA_REP_NAME "float_val.d"
  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 16
  set ACCUM_DATATYPE_WIDTH 32
} elseif { $DATATYPE == "BF16_NS_TRN" } {
  set IO_DATATYPE "F16"
  set ACCUM_DATATYPE "F32"
  set INTERMEDIATE_DATATYPE "F32"
  set VECTOR_DATATYPE "F32"
  set PE_INPUT_DATATYPE "StdFloat<7, 8, false, false, AC_TRN_ZERO>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "StdFloat<7, 8, false, false, AC_TRN_ZERO>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "StdFloat<23, 8, false, false, AC_TRN_ZERO>::AccumulationDatatype"
  set C_DATA_REP_NAME "float_val.d"
  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 16
  set ACCUM_DATATYPE_WIDTH 32
} elseif { $DATATYPE == "BF16_ONLY" } {
  set IO_DATATYPE "F16"
  set ACCUM_DATATYPE "F16"
  set INTERMEDIATE_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "StdFloat<7, 8, false, true, AC_RND_CONV>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "StdFloat<7, 8, false, true, AC_RND_CONV>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "StdFloat<7, 8, false, true, AC_RND_CONV>::AccumulationDatatype"
  set C_DATA_REP_NAME "float_val.d"
  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 16
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "BF16_ONLY_NS" } {
  set IO_DATATYPE "F16"
  set ACCUM_DATATYPE "F16"
  set INTERMEDIATE_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "StdFloat<7, 8, false, false, AC_RND_CONV>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "StdFloat<7, 8, false, false, AC_RND_CONV>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "StdFloat<7, 8, false, false, AC_RND_CONV>::AccumulationDatatype"
  set C_DATA_REP_NAME "float_val.d"
  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 16
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "BF16_ONLY_TRN" } {
  set IO_DATATYPE "F16"
  set ACCUM_DATATYPE "F16"
  set INTERMEDIATE_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "StdFloat<7, 8, false, true, AC_TRN_ZERO>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "StdFloat<7, 8, false, true, AC_TRN_ZERO>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "StdFloat<7, 8, false, true, AC_TRN_ZERO>::AccumulationDatatype"
  set C_DATA_REP_NAME "float_val.d"
  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 16
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "BF16_ONLY_NS_TRN" } {
  set IO_DATATYPE "F16"
  set ACCUM_DATATYPE "F16"
  set INTERMEDIATE_DATATYPE "F16"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "StdFloat<7, 8, false, false, AC_TRN_ZERO>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "StdFloat<7, 8, false, false, AC_TRN_ZERO>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "StdFloat<7, 8, false, false, AC_TRN_ZERO>::AccumulationDatatype"
  set C_DATA_REP_NAME "float_val.d"
  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 16
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "FP32" } {
  set IO_DATATYPE "F32"
  set ACCUM_DATATYPE "F32"
  set INTERMEDIATE_DATATYPE "F32"
  set VECTOR_DATATYPE "F32"
  set PE_INPUT_DATATYPE "StdFloat<23, 8, false, true, AC_RND_CONV>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "StdFloat<23, 8, false, true, AC_RND_CONV>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "StdFloat<23, 8, false, true, AC_RND_CONV>::AccumulationDatatype"
  set C_DATA_REP_NAME "float_val.d"
  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 32
  set ACCUM_DATATYPE_WIDTH 32
} elseif { $DATATYPE == "INT8" } {
  set IO_DATATYPE "I8"
  set ACCUM_DATATYPE "I24"
  set INTERMEDIATE_DATATYPE "I24"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "Int<8, true>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "Int<8, true>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "Int<24, true>::AccumulationDatatype"
  set C_DATA_REP_NAME "int_val"
  set ACC_BUF_C_DATA_REP_NAME "int_val"

  set IO_DATATYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 24
} elseif { $DATATYPE == "INT8_32" } {
  set IO_DATATYPE "I8"
  set ACCUM_DATATYPE "I32"
  set INTERMEDIATE_DATATYPE "I32"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "Int<8, true>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "Int<8, true>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "Int<32, true>::AccumulationDatatype"
  set C_DATA_REP_NAME "int_val"
  set ACC_BUF_C_DATA_REP_NAME "int_val"

  set IO_DATATYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 32
} elseif {$DATATYPE == "MXINT8"} {
  set IO_DATATYPE "I8"
  set ACCUM_DATATYPE "I32"
  set VECTOR_DATATYPE "F16"
  set PE_INPUT_DATATYPE "Int<8, true>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "Int<8, true>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "Int<32, true>::AccumulationDatatype"
  set ACCUM_BUFFER_DATATYPE "F16"
  set C_DATA_REP_NAME "int_val"
  set ACC_BUF_C_DATA_REP_NAME "float_val.d"
  set MX_DATATYPE "Scale<8>"

  set SUPPORT_MX true
  set IO_DATATYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} else {
    puts "Invalid DATATYPE"
    exit 1
}


if {![info exists ACCUM_BUFFER_DATATYPE]} {
  set ACCUM_BUFFER_DATATYPE $ACCUM_DATATYPE
}

if {![info exists MX_DATATYPE]} {
  set MX_DATATYPE $IO_DATATYPE
}

# if SUPPORT_MX is not defined, set it to 0
if {![info exists SUPPORT_MX]} {
  set SUPPORT_MX false
}
