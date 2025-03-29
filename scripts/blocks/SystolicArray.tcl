set block "SystolicArray"
set full_block_name "SystolicArray<$PE_INPUT_DATATYPE, $PE_WEIGHT_DATATYPE, $ACCUM_DATATYPE, $IC_DIMENSION, $OC_DIMENSION>"

proc pre_libraries {} {
  solution library add {[Block] ProcessingElement.v1}
}

proc pre_assembly {} {
  global full_block_name
  set full_block_name_stripped [string map {" " ""} $full_block_name]

  global PE_INPUT_DATATYPE PE_WEIGHT_DATATYPE ACCUM_DATATYPE
  set pe_name "ProcessingElement<$PE_INPUT_DATATYPE,$PE_WEIGHT_DATATYPE,$ACCUM_DATATYPE>"
  set pe_name_stripped [string map {" " ""} $pe_name]

  directive set /$full_block_name_stripped/$pe_name_stripped -MAP_TO_MODULE {[Block] ProcessingElement.v1}
}