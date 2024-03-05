set block "DoubleBuffer"
set full_block_name "DoubleBuffer<$IO_DATATYPE, $DIMENSION, $DIMENSION>"

source scripts/common.tcl

go libraries

directive set -CLOCKS $clocks

go assembly

directive set /Accelerator/DoubleBuffer<$IO_DATATYPE,$DIMENSION,1024>/DoubleBuffer<$IO_DATATYPE,$DIMENSION,1024>:mem0Run/mem0Run/mem0.value.$C_DATA_REP_NAME -WORD_WIDTH [expr $IO_DATATYPE_WIDTH*$DIMENSION]
directive set /Accelerator/DoubleBuffer<$IO_DATATYPE,$DIMENSION,1024>/DoubleBuffer<$IO_DATATYPE,$DIMENSION,1024>:mem1Run/mem1Run/mem1.value.$C_DATA_REP_NAME -WORD_WIDTH [expr $IO_DATATYPE_WIDTH*$DIMENSION]

go extract
