set block "VectorFetchUnit"
set full_block_name "VectorFetchUnit<$IO_DATATYPE, $VECTOR_DATATYPE, $MX_DATATYPE, $OC_DIMENSION>"

proc pre_architect {} {
  global full_block_name VECTOR_DATATYPE IO_DATATYPE
  set vector_fetch_unit_stripped [string map {" " ""} $full_block_name]
}
