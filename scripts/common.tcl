if { [info exists env(DEBUG)] } {
  set project_folder ./build/Catapult_debug_$block
} else {
  set project_folder ./build/Catapult_$block
}

if { [file exists $project_folder] } {
  file delete -force -- $project_folder
}

project new -dir $project_folder

project save

options set Project/SolutionName $block
#options set Output/SubBlockNamePrefix $block
logfile move ./build/${block}.log

options set Message/ErrorOverride ASSERT-1 -remove
options set Input/TargetPlatform x86_64
options set /Input/CppStandard c++11
solution options set /Input/CppStandard c++11
options set Input/SearchPath ./lib
options set Output/OutputVHDL false
options set Architectural/DefaultMemMapThreshold 256
options set Architectural/DefaultRegisterThreshold 4096
options set Flows/Enable-SCVerify yes
options set Flows/VCS/SYSC_VERSION 2.3.2
options set Flows/VCS/VLOGAN_OPTS {+v2k -timescale=1ns/10ps +notimingcheck +define+UNIT_DELAY}
options set Flows/VCS/VCSSIM_OPTS {+fsdbfile+dump.fsdb +fsdb+all=on +fsdb+dumpon+0}
options set Flows/VCS/VCS_DOFILE dump.do
options set Flows/VCS/COMP_FLAGS "-g -Wall -Wno-unknown-pragmas -I../../../lib/ -I../../../src/ -I../../../ -DNO_UNIVERSAL -DSIM_$block"
options set Flows/VCS/VCSELAB_OPTS "-timescale=1ns/1ps -sysc=blocksync -lstdc++fs"
flow package require /SCVerify
flow package option set /SCVerify/USE_VCS true

set clocks {clk {-CLOCK_PERIOD 5 -CLOCK_EDGE rising -CLOCK_HIGH_TIME 2.5 -CLOCK_OFFSET 0.000000 -CLOCK_UNCERTAINTY 0.0 -RESET_KIND async -RESET_SYNC_NAME rst -RESET_SYNC_ACTIVE high -RESET_ASYNC_NAME arst_n -RESET_ASYNC_ACTIVE low -ENABLE_NAME {} -ENABLE_ACTIVE high}}
directive set -CLOCK_OVERHEAD 0
go new

solution file add ./src/Accelerator.h -type CHEADER
solution file add ./test/common/TestRunner.cc -type C++ -exclude true
solution file add ./test/common/Harness.cc -type C++ -exclude true
solution file add ./test/common/Utils.cc -type C++ -exclude true
solution file add ./test/common/GoldModel.cc -type C++ -exclude true
solution file add ./test/common/MemoryModel.cc -type C++ -exclude true
solution file add ./test/common/SimpleMemoryModel.cc -type C++ -exclude true
solution file add ./test/common/Simulation.cc -type C++ -exclude true
solution file add ./test/toolchain/MapOperation.cc -type C++ -exclude true

# iterate through all files in ./test/toolchain/operations and add them to the solution
foreach file [glob -nocomplain -directory ./test/toolchain/operations *.cc] {
  solution file add $file -type C++ -exclude true
}

solution file add ./test/resnet/ResNet.cc -type C++ -exclude true
solution file add ./test/mobilebert/MobileBERT.cc -type C++ -exclude true


go analyze

solution design set $full_block_name -top

if {$block == "Accelerator"} {
  foreach mapped_block [list "InputController<$IO_DATATYPE, $DIMENSION>" "MatrixProcessor<$IO_DATATYPE, $ACCUM_DATATYPE, $DIMENSION, $DIMENSION, 1024>" "VectorUnit<$IO_DATATYPE, $ACCUM_DATATYPE, $DIMENSION>" "WeightController<$IO_DATATYPE, $DIMENSION, $DIMENSION>"] {
    solution design set $mapped_block -mapped
  }
}

go compile

solution options set ComponentLibs/SearchPath {/home/shared/catapult/memories /home/shared/catapult/stdcells} -append
#solution library add tcbn40ulpbwp40_c170815tt1p1v25c_dc -- -rtlsyntool DesignCompiler -vendor TSMC -technology 40nm
solution library add tcbn40ulpbwp40_c170815tt0p9v25c_dc -- -rtlsyntool DesignCompiler -vendor TSMC -technology 40nm

# Add build folder
solution options set ComponentLibs/SearchPath {./build/} -append

solution options set ComponentLibs/SearchPath /sim/kprabhu7/minotaur-accelerator/build -append

if {[info exists env(DEBUG)]} {
  solution library add ccs_sample_mem
} else {
  solution library add TS1N40LPB1024X128M4FWBA_wrapped
  solution library add mem_1024x402
}
