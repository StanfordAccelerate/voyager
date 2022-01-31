source scripts/architecture.tcl

if { [info exists env(DEBUG)] } {
  set project_folder ./build/Catapult_debug
} else {
  set project_folder ./build/Catapult
}

if { [file exists $project_folder] } {
  file delete -force -- $project_folder
}

project new -dir $project_folder

project save

options set Input/TargetPlatform x86_64
options set /Input/CppStandard c++11
solution options set /Input/CppStandard c++11
options set Input/SearchPath ./lib
options set Output/OutputVHDL false
options set Architectural/DefaultMemMapThreshold 256
options set Architectural/DefaultRegisterThreshold 4096
options set Flows/Enable-SCVerify yes
options set Flows/VCS/SYSC_VERSION 2.3.2
options set Flows/VCS/COMP_FLAGS {-g -Wall -Wno-unknown-pragmas -I../../../lib/ -I../../../src/ -I../../../}
flow package require /SCVerify
flow package option set /SCVerify/USE_VCS true

set clocks {clk {-CLOCK_PERIOD 5 -CLOCK_EDGE rising -CLOCK_HIGH_TIME 2.5 -CLOCK_OFFSET 0.000000 -CLOCK_UNCERTAINTY 0.0 -RESET_KIND async -RESET_SYNC_NAME rst -RESET_SYNC_ACTIVE high -RESET_ASYNC_NAME arst_n -RESET_ASYNC_ACTIVE low -ENABLE_NAME {} -ENABLE_ACTIVE high}}
directive set -CLOCK_OVERHEAD 0
go new

solution file add ./src/Accelerator.cc
solution file add ./test/common/TestRunner.cc -exclude true
solution file add ./test/common/Harness.cc -exclude true
solution file add ./test/common/Utils.cc -exclude true
solution file add ./test/common/GoldModel.cc -exclude true

go analyze

directive set -DESIGN_HIERARCHY {
  {Accelerator}
}

go compile

solution options set ComponentLibs/SearchPath {/home/shared/catapult/memories /home/shared/catapult/stdcells} -append
solution library add tcbn40ulpbwp40_c170815tt1p1v25c_dc -- -rtlsyntool DesignCompiler -vendor TSMC -technology 40nm

if {[info exists env(DEBUG)]} {
  solution library add ccs_sample_mem
} else {
  solution library add ts1n40lpb1024x128m4fb_tt1p1v25c
  solution library add mem_1024x402
}


go libraries

directive set -CLOCKS $clocks


go assembly

directive set /Accelerator/DoubleBuffer<$IO_DATATYPE,$DIMENSION,1024>/DoubleBuffer<$IO_DATATYPE,$DIMENSION,1024>:mem0Run/mem0Run/mem0.value.bits -WORD_WIDTH [expr $IO_DATATYPE_WIDTH*$DIMENSION]
directive set /Accelerator/DoubleBuffer<$IO_DATATYPE,$DIMENSION,1024>/DoubleBuffer<$IO_DATATYPE,$DIMENSION,1024>:mem1Run/mem1Run/mem1.value.bits -WORD_WIDTH [expr $IO_DATATYPE_WIDTH*$DIMENSION]

# Map onto same resource
directive set /Accelerator/MatrixProcessor<$IO_DATATYPE,$INTERMEDIATE_DATATYPE,$ACCUM_DATATYPE,$DIMENSION,$DIMENSION,1024>/MatrixProcessor<$IO_DATATYPE,$INTERMEDIATE_DATATYPE,$ACCUM_DATATYPE,$DIMENSION,$DIMENSION,1024>:run/run/while:accumulation_buffer.value.bits -WORD_WIDTH [expr $ACCUM_DATATYPE_WIDTH * $DIMENSION]
directive set /Accelerator/MatrixProcessor<$IO_DATATYPE,$INTERMEDIATE_DATATYPE,$ACCUM_DATATYPE,$DIMENSION,$DIMENSION,1024>/MatrixProcessor<$IO_DATATYPE,$INTERMEDIATE_DATATYPE,$ACCUM_DATATYPE,$DIMENSION,$DIMENSION,1024>:run/run/while:accumulation_buffer.value.bits:rsc -MAP_TO_MODULE mem_1024x402.custom1024x402
# directive set /Accelerator/MatrixProcessor<Posit,Posit,PositFP,Posit,16,16,1024>/MatrixProcessor<Posit,Posit,PositFP,Posit,16,16,1024>:run/run/accumulation_buffer.value.scale -WORD_WIDTH 128
# directive set /Accelerator/MatrixProcessor<Posit,Posit,PositFP,Posit,16,16,1024>/MatrixProcessor<Posit,Posit,PositFP,Posit,16,16,1024>:run/run/accumulation_buffer.value.fraction -WORD_WIDTH 256
# directive set /Accelerator/MatrixProcessor<Posit,Posit,PositFP,Posit,16,16,1024>/MatrixProcessor<Posit,Posit,PositFP,Posit,16,16,1024>:run/run/accumulation_buffer.value.sign -WORD_WIDTH 16

# directive set /Accelerator/MatrixProcessor<Posit,Posit,PositFP,Posit,16,16,1024>/MatrixProcessor<Posit,Posit,PositFP,Posit,16,16,1024>:run/run/accumulation_buffer.value.fraction:rsc -PACKING_MODE sidebyside
# directive set /Accelerator/MatrixProcessor<Posit,Posit,PositFP,Posit,16,16,1024>/MatrixProcessor<Posit,Posit,PositFP,Posit,16,16,1024>:run/run/accumulation_buffer.value.sign -RESOURCE accumulation_buffer.value.fraction:rsc
# directive set /Accelerator/MatrixProcessor<Posit,Posit,PositFP,Posit,16,16,1024>/MatrixProcessor<Posit,Posit,PositFP,Posit,16,16,1024>:run/run/accumulation_buffer.value.scale -RESOURCE accumulation_buffer.value.fraction:rsc
# directive set /Accelerator/MatrixProcessor<Posit,Posit,PositFP,Posit,16,16,1024>/MatrixProcessor<Posit,Posit,PositFP,Posit,16,16,1024>:run/run/accumulation_buffer.value.fraction:rsc -MAP_TO_MODULE mem_1024x402.custom1024x402

# directive set /Accelerator/MatrixProcessor<$IO_DATATYPE,$IO_DATATYPE,$ACCUM_DATATYPE,$IO_DATATYPE,$DIMENSION,$DIMENSION,1024>/run/accumulation_buffer.value.bits -WORD_WIDTH [expr $ACCUM_DATATYPE_WIDTH*$DIMENSION]

if {[info exists env(DEBUG)] == 0} {
  # directive set /Accelerator/MatrixProcessor<$IO_DATATYPE,$IO_DATATYPE,$ACCUM_DATATYPE,$IO_DATATYPE,16,16,1024>/MatrixProcessor<$IO_DATATYPE,$IO_DATATYPE,$ACCUM_DATATYPE,$IO_DATATYPE,16,16,1024>:run/run/accumulation_buffer.value:rsc -MAP_TO_MODULE custom1024x128.custom1024x128
}
directive set /Accelerator/ArithmeticUnit<$ACCUM_DATATYPE,$IO_DATATYPE,$DIMENSION,$DIMENSION>/run/while:maxpool_comparator.value.bits:rsc -MAP_TO_MODULE {[Register]}

directive set /Accelerator/WeightController<$IO_DATATYPE,$DIMENSION,$DIMENSION>/WeightController<$IO_DATATYPE,$DIMENSION,$DIMENSION>:transposer/transposer/while:if#1:transposeBuffer.bits:rsc -MAP_TO_MODULE {[Register]}

go architect

ignore_memory_precedences -from WRITE_ACC_BUFFER* -to READ_ACC_BUFFER*

go allocate
go extract

