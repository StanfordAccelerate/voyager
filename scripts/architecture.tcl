if { $DATATYPE == "P8_1" } {
  set IO_DATATYPE "P8"
  set IO_DATATYPE_DEC "P8D"
  set ACCUM_DATATYPE "PositFP<8, 7>"
  set INTERMEDIATE_DATATYPE "P16D"
  set PE_INPUT_DATATYPE "Posit<8, 1>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "Posit<8, 1>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "PositFP<8, 7>::AccumulationDatatype"
  set C_DATA_REP_NAME "bits"
  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "E4M3" } {
  set IO_DATATYPE "F8"
  set ACCUM_DATATYPE "F16"
  set INTERMEDIATE_DATATYPE "F16"
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
  set PE_INPUT_DATATYPE "StdFloat<3, 5, false, true, AC_RND_CONV>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "StdFloat<3, 5, false, true, AC_RND_CONV>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "StdFloat<7, 8, false, true, AC_RND_CONV>::AccumulationDatatype"
  set C_DATA_REP_NAME "float_val.d"
  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $DATATYPE == "BF16" } {
  set IO_DATATYPE "F16"
  set ACCUM_DATATYPE "F32"
  set INTERMEDIATE_DATATYPE "F32"
  set PE_INPUT_DATATYPE "StdFloat<7, 8, false, true, AC_RND_CONV>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "StdFloat<7, 8, false, true, AC_RND_CONV>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "StdFloat<23, 8, false, true, AC_RND_CONV>::AccumulationDatatype"
  set C_DATA_REP_NAME "float_val.d"
  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 16
  set ACCUM_DATATYPE_WIDTH 32
} elseif { $DATATYPE == "BF16_NS" } {
  set IO_DATATYPE "F16"
  set ACCUM_DATATYPE "F32"
  set INTERMEDIATE_DATATYPE "F32"
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
  set PE_INPUT_DATATYPE "StdFloat<23, 8, false, true, AC_RND_CONV>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "StdFloat<23, 8, false, true, AC_RND_CONV>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "StdFloat<23, 8, false, true, AC_RND_CONV>::AccumulationDatatype"
  set C_DATA_REP_NAME "float_val.d"
  set ACC_BUF_C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 32
  set ACCUM_DATATYPE_WIDTH 32

} else {
    puts "Invalid DATATYPE"
    exit 1
}
