set block "VectorOpUnit"
set full_block_name "VectorOpUnit<$IO_DATATYPE, $VECTOR_DATATYPE, $ACCUM_BUFFER_DATATYPE, $OC_DIMENSION>"
set clock_multiplier 1.4

proc pre_architect {} {
  global full_block_name VECTOR_DATATYPE ACCUM_BUFFER_DATATYPE
  set vector_op_unit_stripped [string map {" " ""} $full_block_name]

  if { [is_floating_point $VECTOR_DATATYPE] && ![is_floating_point $ACCUM_BUFFER_DATATYPE] } {
    # no unrolling needed
  } elseif { $VECTOR_DATATYPE == $ACCUM_BUFFER_DATATYPE } {
    directive set /$vector_op_unit_stripped/vectorOpRun/LOOP_0_UNROLL -UNROLL yes
  } else {
    directive set /$vector_op_unit_stripped/vectorOpRun/LOOP_1_UNROLL -UNROLL yes
  }
}
