set block "VectorUnit"
set full_block_name "VectorUnit<$VECTOR_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SCALE_DATATYPE, $VECTOR_UNIT_WIDTH, $OC_DIMENSION>"

proc pre_libraries {} {
  solution library add {[Block] VectorFetchUnit.v1}
  solution library add {[Block] VectorPipeline.v1}
  solution library add {[Block] VectorReducer.v1}
  solution library add {[Block] VectorAccumulator.v1}
  solution library add {[Block] OutputController.v1}
  solution library add {[Block] VectorParamsDeserializer.v1}
}

proc pre_assembly {} {
  global full_block_name VECTOR_DATATYPE ACCUM_BUFFER_DATATYPE OC_DIMENSION VECTOR_UNIT_WIDTH VU_INPUT_TYPES SCALE_DATATYPE OUTPUT_DATATYPES
  set full_block_name_stripped [string map {" " ""} $full_block_name]

  set vector_fetch_name "VectorFetchUnit<$VECTOR_DATATYPE,$ACCUM_BUFFER_DATATYPE,$VECTOR_UNIT_WIDTH,$OC_DIMENSION,$VU_INPUT_TYPES>"
  set vector_fetch_name_stripped [string map {" " ""} $vector_fetch_name]
  directive set /$full_block_name_stripped/$vector_fetch_name_stripped -MAP_TO_MODULE {[Block] VectorFetchUnit.v1}

  set vector_pipeline_name "VectorPipeline<$VECTOR_DATATYPE,$ACCUM_BUFFER_DATATYPE,$SCALE_DATATYPE,$VECTOR_UNIT_WIDTH,$OC_DIMENSION>"
  set vector_pipeline_name_stripped [string map {" " ""} $vector_pipeline_name]
  directive set /$full_block_name_stripped/$vector_pipeline_name_stripped -MAP_TO_MODULE {[Block] VectorPipeline.v1}

  set vector_reducer_name "VectorReducer<$VECTOR_DATATYPE,$VECTOR_UNIT_WIDTH>"
  set vector_reducer_name_stripped [string map {" " ""} $vector_reducer_name]
  directive set /$full_block_name_stripped/$vector_reducer_name_stripped -MAP_TO_MODULE {[Block] VectorReducer.v1}

  set vector_accumulator_name "VectorAccumulator<$VECTOR_DATATYPE,$VECTOR_UNIT_WIDTH>"
  set vector_accumulator_name_stripped [string map {" " ""} $vector_accumulator_name]
  directive set /$full_block_name_stripped/$vector_accumulator_name_stripped -MAP_TO_MODULE {[Block] VectorAccumulator.v1}

  set output_controller_name "OutputController<$VECTOR_DATATYPE,$SCALE_DATATYPE,$OC_DIMENSION,$OUTPUT_DATATYPES>"
  set output_controller_name_stripped [string map {" " ""} $output_controller_name]
  directive set /$full_block_name_stripped/$output_controller_name_stripped -MAP_TO_MODULE {[Block] OutputController.v1}

  directive set /$full_block_name_stripped/VectorParamsDeserializer -MAP_TO_MODULE {[Block] VectorParamsDeserializer.v1}
}
