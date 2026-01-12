set block "MatrixVectorUnit"
set full_block_name "MatrixVectorUnit<InputTypeList, WeightTypeList, $SA_INPUT_TYPE, $SA_WEIGHT_TYPE, $ACCUM_DATATYPE, $VECTOR_DATATYPE, $SCALE_DATATYPE, $OC_PORT_WIDTH, $MV_UNIT_WIDTH, $OC_DIMENSION, $VECTOR_UNIT_WIDTH>"
set full_block_name_stripped [string map {" " ""} $full_block_name]

proc pre_libraries {} {
  solution library add {[Block] VectorMacUnit.v1}
}

proc pre_assembly {} {
  global full_block_name
  set full_block_name_stripped [string map {" " ""} $full_block_name]

  global SA_INPUT_TYPE SA_WEIGHT_TYPE ACCUM_DATATYPE SCALE_DATATYPE VECTOR_DATATYPE MV_UNIT_WIDTH OC_DIMENSION
  set vector_mac_name "VectorMacUnit<$SA_INPUT_TYPE, $SA_WEIGHT_TYPE, $ACCUM_DATATYPE, $SCALE_DATATYPE, $VECTOR_DATATYPE, $MV_UNIT_WIDTH, $OC_DIMENSION>"
  set vector_mac_name_stripped [string map {" " ""} $vector_mac_name]

  directive set /$full_block_name_stripped/$vector_mac_name_stripped -MAP_TO_MODULE {[Block] VectorMacUnit.v1}
}
