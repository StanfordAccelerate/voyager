set block "MatrixVectorUnit"
set full_block_name "MatrixVectorUnit<InputTypeList, WeightTypeList, $SA_INPUT_TYPE, $SA_WEIGHT_TYPE, $ACCUM_DATATYPE, $VECTOR_DATATYPE, $SCALE_DATATYPE, $OC_PORT_WIDTH, $MV_UNIT_WIDTH, $OC_DIMENSION>"
set full_block_name_stripped [string map {" " ""} $full_block_name]

proc pre_architect {} {
  global full_block_name_stripped
}
