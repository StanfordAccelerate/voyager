set block "MaxpoolUnit"
set full_block_name "MaxpoolUnit<$VECTOR_DATATYPE, $SCALE_DATATYPE, $IO_DATATYPE, $OC_DIMENSION>"
set full_block_name_stripped [string map {" " ""} $full_block_name]

proc pre_architect {} {
  global full_block_name_stripped VECTOR_DATATYPE IO_DATATYPE
}
