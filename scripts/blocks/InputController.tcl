set block "InputController"
set full_block_name "InputController<InputTypeList, $IC_DIMENSION, $IC_PORT_WIDTH, $INPUT_BUFFER_WIDTH>"

proc pre_architect {} {
  global full_block_name
  set input_controller_stripped [string map {" " ""} $full_block_name]

  global IC_DIMENSION
  if {$IC_DIMENSION <= 32} {
    directive set /$input_controller_stripped/$input_controller_stripped:transposer/transposer/while:if:transpose_buffer:rsc -MAP_TO_MODULE {[Register]}
  }
}
