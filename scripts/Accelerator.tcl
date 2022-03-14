source scripts/architecture.tcl

set block "Accelerator"
set full_block_name "Accelerator"

source scripts/common.tcl

solution library add {[Block] InputController.v1}
solution library add {[Block] MatrixProcessor.v1}
solution library add {[Block] VectorUnit.v1}

go libraries

directive set /Accelerator/MatrixProcessor<Posit<8,1>,Posit<16,1>,16,16,1024> -MAP_TO_MODULE {[Block] MatrixProcessor.v1}
directive set /Accelerator/InputController<Posit<8,1>,16> -MAP_TO_MODULE {[Block] InputController.v1}
directive set /Accelerator/VectorUnit<Posit<8,1>,Posit<16,1>,16> -MAP_TO_MODULE {[Block] VectorUnit.v1}

directive set -CLOCKS $clocks

go assembly

set double_buffer "DoubleBuffer<$IO_DATATYPE,$DIMENSION,1024>"
set weight_controller "WeightController<$IO_DATATYPE,$DIMENSION,$DIMENSION>"
set double_buffer_stripped [string map {" " ""} $double_buffer]
set weight_controller_stripped [string map {" " ""} $weight_controller]

directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem0Run/mem0Run/mem0.value.bits -WORD_WIDTH [expr $IO_DATATYPE_WIDTH*$DIMENSION]
directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem1Run/mem1Run/mem1.value.bits -WORD_WIDTH [expr $IO_DATATYPE_WIDTH*$DIMENSION]

directive set /Accelerator/$weight_controller_stripped/$weight_controller_stripped:transposer/transposer/while:if#1:transposeBuffer.bits:rsc -MAP_TO_MODULE {[Register]}

go extract
