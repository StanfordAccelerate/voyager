set block "WeightController"
set full_block_name "WeightController<WeightTypeList, $ACCUM_BUFFER_DATATYPE, $IC_DIMENSION, $OC_DIMENSION, $OC_PORT_WIDTH, $WEIGHT_BUFFER_WIDTH>"

proc pre_architect {} {
  global IC_DIMENSION OC_DIMENSION

  global full_block_name
  set weight_controller_stripped [string map {" " ""} $full_block_name]

  if {$IC_DIMENSION < 64 && $OC_DIMENSION < 64} {
    directive set /$weight_controller_stripped/$weight_controller_stripped:transposer/transposer/while:if:transpose_buffer:rsc -MAP_TO_MODULE {[Register]}
  }
}
