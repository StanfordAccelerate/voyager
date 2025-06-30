set block "SystolicArray"
set full_block_name "SystolicArray<$SA_INPUT_TYPE, $SA_WEIGHT_TYPE, $ACCUM_DATATYPE, $IC_DIMENSION, $OC_DIMENSION>"

proc pre_compile {} {
  global SA_INPUT_TYPE SA_WEIGHT_TYPE ACCUM_DATATYPE OC_DIMENSION
  solution design set "SystolicArrayRow<$SA_INPUT_TYPE, $SA_WEIGHT_TYPE, $ACCUM_DATATYPE, $OC_DIMENSION>" -mapped
}

proc pre_libraries {} {
  solution library add {[Block] SystolicArrayRow.v1}
}

proc pre_assembly {} {
  global full_block_name
  set full_block_name_stripped [string map {" " ""} $full_block_name]

  global SA_INPUT_TYPE SA_WEIGHT_TYPE ACCUM_DATATYPE OC_DIMENSION
  set row_name "SystolicArrayRow<$SA_INPUT_TYPE,$SA_WEIGHT_TYPE,$ACCUM_DATATYPE,$OC_DIMENSION>"
  set row_name_stripped [string map {" " ""} $row_name]

  directive set /$full_block_name_stripped/$row_name_stripped -MAP_TO_MODULE {[Block] SystolicArrayRow.v1}
}
