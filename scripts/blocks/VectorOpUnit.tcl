set block "VectorOpUnit"
set full_block_name "VectorOpUnit<$IO_DATATYPE, $VECTOR_DATATYPE, $ACCUM_DATATYPE, $OC_DIMENSION>"
set use_slower_clock 1

proc pre_architect {} {
  global full_block_name VECTOR_DATATYPE ACCUM_DATATYPE
  set vector_op_unit_stripped [string map {" " ""} $full_block_name]

  # only unroll the vectorOpRun block if VECTOR_DATATYPE and ACCUM_DATATYPE are both floating point
  if { [is_floating_point $VECTOR_DATATYPE] && [is_floating_point $ACCUM_DATATYPE] } {
    directive set /$vector_op_unit_stripped/vectorOpRun/UNROLL -UNROLL yes
  }
}