set block "VectorUnit"
set full_block_name "VectorUnit<$IO_DATATYPE, $VECTOR_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SCALE_DATATYPE, $OC_DIMENSION>"

proc pre_libraries {} {
  solution library add {[Block] VectorFetchUnit.v1}
  solution library add {[Block] VectorOpUnit.v1}
  solution library add {[Block] VectorUnitOutput.v1}
  solution library add {[Block] VectorParamsDeserializer.v1}
}

proc pre_assembly {} {
  global full_block_name
  set full_block_name_stripped [string map {" " ""} $full_block_name]

  global IO_DATATYPE ACCUM_BUFFER_DATATYPE VECTOR_DATATYPE OC_DIMENSION SCALE_DATATYPE OUTPUT_DATATYPES VECTOR_INPUT_DATATYPES
  set vector_fetch_name "VectorFetchUnit<$IO_DATATYPE,$VECTOR_DATATYPE,$SCALE_DATATYPE,$OC_DIMENSION,$VECTOR_INPUT_DATATYPES>"
  set vector_fetch_name_stripped [string map {" " ""} $vector_fetch_name]
  set vector_op_name "VectorOpUnit<$IO_DATATYPE,$VECTOR_DATATYPE,$ACCUM_BUFFER_DATATYPE,$SCALE_DATATYPE,$OC_DIMENSION>"
  set vector_op_name_stripped [string map {" " ""} $vector_op_name]
  set vector_output_name "VectorUnitOutput<$VECTOR_DATATYPE,$SCALE_DATATYPE,$IO_DATATYPE,$OC_DIMENSION,$OUTPUT_DATATYPES>"
  set vector_output_name_stripped [string map {" " ""} $vector_output_name]

  directive set /$full_block_name_stripped/$vector_fetch_name_stripped -MAP_TO_MODULE {[Block] VectorFetchUnit.v1}
  directive set /$full_block_name_stripped/$vector_op_name_stripped -MAP_TO_MODULE {[Block] VectorOpUnit.v1}
  directive set /$full_block_name_stripped/$vector_output_name_stripped -MAP_TO_MODULE {[Block] VectorUnitOutput.v1}
  directive set /$full_block_name_stripped/VectorParamsDeserializer -MAP_TO_MODULE {[Block] VectorParamsDeserializer.v1}
}