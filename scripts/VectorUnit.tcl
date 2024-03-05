source scripts/architecture.tcl

set block "VectorUnit"
set full_block_name "VectorUnit<$IO_DATATYPE, $ACCUM_DATATYPE, $DIMENSION>"

source scripts/common.tcl

solution library add {[Block] MaxpoolUnit.v1}
solution library add {[Block] VectorFetchUnit.v1}
solution library add {[Block] OutputAddressGenerator.v1}
solution library add {[Block] VectorOpUnit.v1}

go libraries

directive set /VectorUnit<$IO_DATATYPE,$ACCUM_DATATYPE,$DIMENSION>/MaxpoolUnit<$ACCUM_DATATYPE,$IO_DATATYPE,$DIMENSION> -MAP_TO_MODULE {[Block] MaxpoolUnit.v1}
directive set /VectorUnit<$IO_DATATYPE,$ACCUM_DATATYPE,$DIMENSION>/VectorFetchUnit<$IO_DATATYPE,$ACCUM_DATATYPE,$DIMENSION> -MAP_TO_MODULE {[Block] VectorFetchUnit.v1}
directive set /VectorUnit<$IO_DATATYPE,$ACCUM_DATATYPE,$DIMENSION>/OutputAddressGenerator<$DIMENSION> -MAP_TO_MODULE {[Block] OutputAddressGenerator.v1}
directive set /VectorUnit<$IO_DATATYPE,$ACCUM_DATATYPE,$DIMENSION>/VectorOpUnit<$IO_DATATYPE,$ACCUM_DATATYPE,$DIMENSION> -MAP_TO_MODULE {[Block] VectorOpUnit.v1}

directive set -CLOCKS $clocks

go assembly

go extract
