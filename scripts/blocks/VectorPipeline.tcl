set block "VectorPipeline"
set full_block_name "VectorPipeline<$VECTOR_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SCALE_DATATYPE, $VECTOR_UNIT_WIDTH, $OC_DIMENSION>"

proc pre_libraries {} {
  global SUPPORT_SPMM

  if {$SUPPORT_SPMM == true} {
    solution library add {[Block] OutlierFilter.v1}
  }
}

proc pre_assembly {} {
  global full_block_name VECTOR_DATATYPE SPMM_META_DATATYPE VECTOR_UNIT_WIDTH SUPPORT_SPMM
  set full_block_name_stripped [string map {" " ""} $full_block_name]

  if {$SUPPORT_SPMM == true} {
    set outlier_filter_name "OutlierFilter<$VECTOR_DATATYPE, $SPMM_META_DATATYPE, $VECTOR_UNIT_WIDTH>"
    set outlier_filter_name_stripped [string map {" " ""} $outlier_filter_name]
    directive set /$full_block_name_stripped/$outlier_filter_name_stripped -MAP_TO_MODULE {[Block] OutlierFilter.v1}
  }
}
