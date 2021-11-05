source scripts/architecture.tcl

project new -dir ./build/Catapult
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

go new

solution file add ./src/Accelerator.cc
solution file add ./test/simple/SimpleTest.cc -exclude true
solution file add ./test/common/Harness.cc -exclude true
solution file add ./test/common/Utils.cc -exclude true
solution file add ./test/common/GoldModel.cc -exclude true

go analyze

directive set -DESIGN_HIERARCHY {
  {Accelerator}
}

go compile

solution options set ComponentLibs/SearchPath /home/kprabhu7/catapult3_char -append
solution library add tcbn40lpbwptc_dc -- -rtlsyntool DesignCompiler -vendor TSMC -technology 40nm

solution library add ccs_sample_mem

go libraries

directive set -CLOCKS $clocks


go assembly

directive set /Accelerator/DoubleBuffer<ac_int<8,true>,$DIMENSION,1024>/DoubleBuffer<ac_int<8,true>,$DIMENSION,1024>:mem0Run/mem0Run/mem0.value -WORD_WIDTH [expr 8*$DIMENSION]
directive set /Accelerator/DoubleBuffer<ac_int<8,true>,$DIMENSION,1024>/DoubleBuffer<ac_int<8,true>,$DIMENSION,1024>:mem1Run/mem1Run/mem1.value -WORD_WIDTH [expr 8*$DIMENSION]
directive set /Accelerator/MatrixProcessor<ac_int<8,true>,ac_int<8,true>,ac_int<8,true>,$DIMENSION,$DIMENSION,1024>/run/accumulation_buffer.value -WORD_WIDTH [expr 8*$DIMENSION]

go architect

ignore_memory_precedences -from WRITE_ACC_BUFFER* -to READ_ACC_BUFFER*

go allocate
go extract

