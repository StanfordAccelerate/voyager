set block "VectorPipeline"
set full_block_name "VectorPipeline<$VECTOR_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SCALE_DATATYPE, $VECTOR_UNIT_WIDTH, $OC_DIMENSION>"

proc pre_architect {} {
  global full_block_name
  set vector_pipeline_stripped [string map {" " ""} $full_block_name]
}
