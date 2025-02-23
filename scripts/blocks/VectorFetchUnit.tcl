set block "VectorFetchUnit"
set full_block_name "VectorFetchUnit<$IO_DATATYPE, $VECTOR_DATATYPE, $SCALE_DATATYPE, $OC_DIMENSION>"

proc pre_architect {} {
  global full_block_name VECTOR_DATATYPE IO_DATATYPE ACC_BUF_C_DATA_REP_NAME
  set vector_fetch_unit_stripped [string map {" " ""} $full_block_name]

  directive set /$vector_fetch_unit_stripped/$vector_fetch_unit_stripped:feed_data_response_0/feed_data_response_0/while:if#1:buffer.$ACC_BUF_C_DATA_REP_NAME:rsc -MAP_TO_MODULE {[Register]}
}
