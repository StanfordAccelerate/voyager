set block "ProcessingElement"
set full_block_name "ProcessingElement<$SA_INPUT_TYPE, $SA_WEIGHT_TYPE, $ACCUM_DATATYPE>"

proc pre_extract {} {
  cycle set input_in.Pop() -from psum_in.Pop() -equal 0
}
