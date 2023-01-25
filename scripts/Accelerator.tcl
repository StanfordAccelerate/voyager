source scripts/architecture.tcl

set block "Accelerator"
set full_block_name "Accelerator"

source scripts/common.tcl

solution library add {[Block] InputController.v1}
solution library add {[Block] MatrixProcessor.v1}
solution library add {[Block] VectorUnit.v1}
solution library add {[Block] WeightController.v1}

go libraries

directive set /Accelerator/MatrixProcessor<P8,P16,16,16,1024> -MAP_TO_MODULE {[Block] MatrixProcessor.v1}
directive set /Accelerator/InputController<P8,16> -MAP_TO_MODULE {[Block] InputController.v1}
directive set /Accelerator/WeightController<P8,P16,16,16> -MAP_TO_MODULE {[Block] WeightController.v1}
directive set /Accelerator/VectorUnit<P8,P16,16> -MAP_TO_MODULE {[Block] VectorUnit.v1}

directive set -CLOCKS $clocks

go assembly

set double_buffer "DoubleBuffer<$IO_DATATYPE,$DIMENSION,1024>"
set double_buffer_stripped [string map {" " ""} $double_buffer]

directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem0Run/mem0Run/mem0.value.bits -WORD_WIDTH [expr $IO_DATATYPE_WIDTH*$DIMENSION]
directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem1Run/mem1Run/mem1.value.bits -WORD_WIDTH [expr $IO_DATATYPE_WIDTH*$DIMENSION]

go extract
