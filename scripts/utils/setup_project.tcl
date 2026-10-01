# Create project folder
set project_folder [file join $ROOT $CATAPULT_BUILD_DIR $BLOCK]
if { [file exists $project_folder] } {
  file delete -force -- $project_folder
}

file mkdir [file dirname $project_folder]
project new -dir $project_folder
project save

# Configure options
solution options set Project/SolutionName $BLOCK
solution options set Message/ErrorOverride ASSERT-1 -remove
solution options set Input/TargetPlatform x86_64
solution options set Input/CppStandard c++17
if {[info exists ::env(ARCHITECTURE_DEFINES)]} {
  set architecture_defines $::env(ARCHITECTURE_DEFINES)
} else {
  set architecture_defines "-D$DATATYPE"
  foreach name $ARCHITECTURE_PARAMS {
    if {[info exists $name]} {
      append architecture_defines " -D$name=[set $name]"
    }
  }
}
if {![regexp {(^|\s)-DCLOCK_PERIOD=} $architecture_defines]} {
  append architecture_defines " -DCLOCK_PERIOD=$CLOCK_PERIOD"
}
solution options set Input/CompilerFlags $architecture_defines
solution options set Input/SearchPath "$ROOT/lib"
solution options set Architectural/DefaultMemMapThreshold 1024
solution options set Architectural/DefaultRegisterThreshold 4096
solution options set Flows/Enable-SCVerify yes
solution options set Flows/VCS/SYSC_VERSION 2.3.3
solution options set Flows/VCS/VLOGAN_OPTS {+v2k -timescale=1ns/10ps +notimingcheck +define+UNIT_DELAY}
# solution options set Flows/VCS/VCSSIM_OPTS {+fsdbfile+dump.fsdb +fsdb+all=on +fsdb+dumpon+0}
solution options set Flows/VCS/VCSSIM_OPTS {+vcs+lic+wait}
solution options set Flows/VCS/VCS_DOFILE "$ROOT/utils/dump.do"
solution options set Flows/VCS/COMP_FLAGS "-O3 -Wall -Wno-unknown-pragmas -DSC_INCLUDE_DYNAMIC_PROCESSES -I$ROOT/lib/ -I$ROOT/lib/xtensor/include -I$ROOT/lib/xtl/include -I$ROOT/lib/spdlog/include -I$ROOT/src/ -I$ROOT/ -I$::env(CONDA_PREFIX)/include -DSIM_$BLOCK $architecture_defines -std=c++17"
solution options set Flows/VCS/VCSELAB_OPTS "-timescale=1ns/1ps -sysc=blocksync -lstdc++fs -L$::env(CONDA_PREFIX)/lib -LDFLAGS \"-Wl,--enable-new-dtags -Wl,-R,$::env(CONDA_PREFIX)/lib\" -labsl_hash -labsl_log_internal_check_op -labsl_log_internal_message -labsl_log_internal_nullguard -lprotobuf -lpthread"
solution options set Cache/UserCacheHome [file join $ROOT $CATAPULT_BUILD_DIR cache]
solution options set Cache/DefaultCacheHomeEnabled false

solution options set Output/OutputVHDL false
solution options set Output/MaxNameLength 256
solution options set Output/MaxNameLengthModule 256
solution options set Output/MaxNameLengthInstance 256
solution options set Output/MaxNameLengthPort 256
solution options set Output/MaxNameLengthRegWire 256

flow package require /SCVerify
flow package option set /SCVerify/USE_VCS true

go new

# Add source files
solution file add $ROOT/src/Accelerator.h -type CHEADER

# Add testbench files
solution file add $ROOT/test/common/TestRunner.cc -type C++ -exclude true
solution file add $ROOT/test/common/Harness.cc -type C++ -exclude true
solution file add $ROOT/test/common/GoldModel.cc -type C++ -exclude true
solution file add $ROOT/test/common/Simulation.cc -type C++ -exclude true
solution file add $ROOT/test/common/Checker.cc -type C++ -exclude true
solution file add $ROOT/test/common/ArrayMemory.cc -type C++ -exclude true
solution file add $ROOT/test/common/AccessCounter.cc -type C++ -exclude true
solution file add $ROOT/test/toolchain/MapOperation.cc -type C++ -exclude true

# Add the bufferized IR runtime
solution file add $ROOT/test/common/Model.cc -type C++ -exclude true
solution file add $ROOT/test/common/Interpreter.cc -type C++ -exclude true
solution file add $ROOT/test/common/GraphUtils.cc -type C++ -exclude true
solution file add $ROOT/test/common/Tiling.cc -type C++ -exclude true
solution file add $ROOT/test/compiler/proto/voyager_ir.pb.cc -type C++ -exclude true
