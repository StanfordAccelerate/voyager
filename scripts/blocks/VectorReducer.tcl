set block "VectorReducer"
set full_block_name "VectorReducer<$VECTOR_DATATYPE, $OC_DIMENSION>"

proc pre_architect {} {
  global full_block_name
  set vector_reducer_stripped [string map {" " ""} $full_block_name]
}
