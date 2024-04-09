# read environment variable "DATATYPE"
global env
set datatype $env(DATATYPE)
set DIMENSION $env(DIMENSION)

if { $datatype == "P8_1" } {
  set IO_DATATYPE "P8"
  set IO_DATATYPE_DEC "P8D"
  set ACCUM_DATATYPE "PositFP<8, 7>"
  set INTERMEDIATE_DATATYPE "P16D"
  set PE_INPUT_DATATYPE "Posit<8, 1>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "Posit<8, 1>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "PositFP<8, 7>::AccumulationDatatype"
  set C_DATA_REP_NAME "bits"

  set IO_DATATYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $datatype == "E4M3" } {
  set IO_DATATYPE "F8"
  set ACCUM_DATATYPE "F16"
  set INTERMEDIATE_DATATYPE "F16"
  set PE_INPUT_DATATYPE "StdFloat<3, 4, false, true, AC_RND_CONV>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "StdFloat<3, 4, false, true, AC_RND_CONV>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "StdFloat<7, 8, false, true, AC_RND_CONV>::AccumulationDatatype"
  set C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $datatype == "E5M2" } {
  set IO_DATATYPE "F8"
  set ACCUM_DATATYPE "F16"
  set INTERMEDIATE_DATATYPE "F16"
  set PE_INPUT_DATATYPE "StdFloat<2, 5>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "StdFloat<2, 5>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "StdFloat<7, 8>::AccumulationDatatype"
  set C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $datatype == "HYBRID_FP8" } {
  set IO_DATATYPE "F8"
  set ACCUM_DATATYPE "F16"
  set INTERMEDIATE_DATATYPE "F16"
  set PE_INPUT_DATATYPE "StdFloat<3, 5>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "StdFloat<3, 5>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "StdFloat<7, 8>::AccumulationDatatype"
  set C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 8
  set ACCUM_DATATYPE_WIDTH 16
} elseif { $datatype == "BF16" } {
  set IO_DATATYPE "F16"
  set ACCUM_DATATYPE "F32"
  set INTERMEDIATE_DATATYPE "F32"
  set PE_INPUT_DATATYPE "StdFloat<7, 8>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "StdFloat<7, 8>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "StdFloat<23, 8>::AccumulationDatatype"
  set C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 16
  set ACCUM_DATATYPE_WIDTH 32
} elseif { $datatype == "BF16_ONLY" } {
  set IO_DATATYPE "F16"
  set ACCUM_DATATYPE "F16"
  set INTERMEDIATE_DATATYPE "F16"
  set PE_INPUT_DATATYPE "StdFloat<7, 8>::AccumulationDatatype"
  set PE_WEIGHT_DATATYPE "StdFloat<7, 8>::AccumulationDatatype"
  set PE_PSUM_DATATYPE "StdFloat<7, 8>::AccumulationDatatype"
  set C_DATA_REP_NAME "float_val.d"

  set IO_DATATYPE_WIDTH 16
  set ACCUM_DATATYPE_WIDTH 16
} else {
    puts "Invalid DATATYPE"
    exit 1
}
