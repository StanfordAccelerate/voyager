set block "OutlierFilter"
set full_block_name "OutlierFilter<$VECTOR_DATATYPE, $SPMM_META_DATATYPE, $VECTOR_UNIT_WIDTH>"

proc pre_architect {} {
  global full_block_name
  set full_block_name_stripped [string map {" " ""} $full_block_name]
}
