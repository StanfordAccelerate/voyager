set block "VectorQuantizer"
set full_block_name "VectorQuantizer<$VECTOR_DATATYPE, $SCALE_DATATYPE, $OC_DIMENSION>"

proc pre_architect {} {
  global full_block_name
  set vector_quantizer_stripped [string map {" " ""} $full_block_name]
}
