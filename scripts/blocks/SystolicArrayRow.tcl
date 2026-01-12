set block "SystolicArrayRow"
set full_block_name "SystolicArrayRow<$SA_INPUT_TYPE, $SA_WEIGHT_TYPE, $ACCUM_DATATYPE, $OC_DIMENSION>"

proc pre_libraries {} {
  solution library add {[Block] ProcessingElement.v1}
}

proc pre_assembly {} {
  global full_block_name
  set full_block_name_stripped [string map {" " ""} $full_block_name]

  global SA_INPUT_TYPE SA_WEIGHT_TYPE ACCUM_DATATYPE
  set pe_name "ProcessingElement<$SA_INPUT_TYPE,$SA_WEIGHT_TYPE,$ACCUM_DATATYPE>"
  set pe_name_stripped [string map {" " ""} $pe_name]

  directive set /$full_block_name_stripped/$pe_name_stripped -MAP_TO_MODULE {[Block] ProcessingElement.v1}
}
