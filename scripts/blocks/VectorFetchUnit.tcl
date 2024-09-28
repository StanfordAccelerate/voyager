set block "VectorFetchUnit"
set full_block_name "VectorFetchUnit<$IO_DATATYPE, $VECTOR_DATATYPE, $OC_DIMENSION>"

proc pre_architect {} {
  global full_block_name VECTOR_DATATYPE IO_DATATYPE
  set vector_fetch_unit_stripped [string map {" " ""} $full_block_name]

  # only unroll the feed_data_response blocks if the VECTOR_DATATYPE and IO_datatype are both floating point
  if { [is_floating_point $VECTOR_DATATYPE] && [is_floating_point $IO_DATATYPE] } {
    directive set /$vector_fetch_unit_stripped/feed_data_response_0/UNROLL -UNROLL yes
    directive set /$vector_fetch_unit_stripped/feed_data_response_1/UNROLL -UNROLL yes
    directive set /$vector_fetch_unit_stripped/feed_data_response_2/UNROLL -UNROLL yes
  }
}