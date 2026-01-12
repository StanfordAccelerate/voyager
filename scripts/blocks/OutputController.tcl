set block "OutputController"
set full_block_name "OutputController<$VECTOR_DATATYPE, $SCALE_DATATYPE, $OC_DIMENSION, $OUTPUT_DATATYPES>"
set full_block_name_stripped [string map {" " ""} $full_block_name]

proc pre_architect {} {
  global full_block_name_stripped
}
