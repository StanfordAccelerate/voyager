set block "InputController"
set full_block_name "InputController<$IO_DATATYPE, $IC_DIMENSION>"
set clock_multiplier 1.1

proc pre_architect {} {
  global C_DATA_REP_NAME

  global full_block_name
  set input_controller_stripped [string map {" " ""} $full_block_name]

  directive set /$input_controller_stripped/$input_controller_stripped:transposer/transposer/while:if#2:transposeBuffer.$C_DATA_REP_NAME:rsc -MAP_TO_MODULE {[Register]}
}
