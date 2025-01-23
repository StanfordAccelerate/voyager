set block "WeightController"
set full_block_name "WeightController<$IO_DATATYPE, $ACCUM_BUFFER_DATATYPE, $IC_DIMENSION, $OC_DIMENSION>"
set clock_multiplier 1.4

proc pre_architect {} {
  global C_DATA_REP_NAME IC_DIMENSION OC_DIMENSION

  global full_block_name
  set weight_controller_stripped [string map {" " ""} $full_block_name]

  if {$IC_DIMENSION < 64 && $OC_DIMENSION < 64} {
    directive set /$weight_controller_stripped/$weight_controller_stripped:transposer/transposer/while:if:transposeBuffer.$C_DATA_REP_NAME:rsc -MAP_TO_MODULE {[Register]}
  }
}
