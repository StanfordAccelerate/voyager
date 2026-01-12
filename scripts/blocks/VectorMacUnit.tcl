set block "VectorMacUnit"
set full_block_name "VectorMacUnit<$SA_INPUT_TYPE, $SA_WEIGHT_TYPE, $ACCUM_DATATYPE, $SCALE_DATATYPE, $VECTOR_DATATYPE, $MV_UNIT_WIDTH, $OC_DIMENSION>"

proc pre_extract {} {
  global full_block_name
  set vector_mac_unit_stripped [string map {" " ""} $full_block_name]
}
