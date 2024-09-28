set block "MaxpoolUnit"
set full_block_name "MaxpoolUnit<$VECTOR_DATATYPE, $IO_DATATYPE, $OC_DIMENSION>"
set full_block_name_stripped [string map {" " ""} $full_block_name]

proc pre_architect {} {
  global full_block_name_stripped VECTOR_DATATYPE IO_DATATYPE

  # only unroll the run block if VECTOR_DATATYPE and IO_DATATYPE are both floating point
  if { [is_floating_point $VECTOR_DATATYPE] && [is_floating_point $IO_DATATYPE] } {
    directive set /$full_block_name_stripped/run/UNROLL -UNROLL yes
  }
}
