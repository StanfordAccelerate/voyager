set block "SpMMUnit"
set full_block_name "SpMMUnit<WeightTypeList, $VECTOR_DATATYPE, $SA_WEIGHT_TYPE, $SPMM_META_DATATYPE, $VECTOR_DATATYPE, $SCALE_DATATYPE, $OC_PORT_WIDTH, $SPMM_UNIT_WIDTH, $IC_DIMENSION, $VECTOR_UNIT_WIDTH>"
set full_block_name_stripped [string map {" " ""} $full_block_name]

proc pre_architect {} {
  global full_block_name
  set full_block_name_stripped [string map {" " ""} $full_block_name]
}
