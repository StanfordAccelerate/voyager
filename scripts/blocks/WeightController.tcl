set block "WeightController"
set full_block_name "WeightController<$IO_DATATYPE, $ACCUM_DATATYPE, $IC_DIMENSION, $OC_DIMENSION>"
set use_slower_clock 1

proc pre_architect {} {
  global C_DATA_REP_NAME

  global full_block_name
  set weight_controller_stripped [string map {" " ""} $full_block_name]

  directive set /$weight_controller_stripped/$weight_controller_stripped:transposer/transposer/while:if:transposeBuffer.$C_DATA_REP_NAME:rsc -MAP_TO_MODULE {[Register]}
}
