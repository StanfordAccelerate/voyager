set block "VectorFetchUnit"
set full_block_name "VectorFetchUnit<$VECTOR_DATATYPE, $ACCUM_BUFFER_DATATYPE, $OC_DIMENSION, $VU_INPUT_TYPES>"

proc pre_architect {} {
  global full_block_name
  set vector_fetch_unit_stripped [string map {" " ""} $full_block_name]

  directive set /$vector_fetch_unit_stripped/$vector_fetch_unit_stripped:feed_data_response_0/feed_data_response_0/while:if#3:buffer.float_val.d:rsc -MAP_TO_MODULE {[Register]}
}
