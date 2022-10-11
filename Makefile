.DEFAULT_GOAL := sim

# Detect OS 
OS := $(shell lsb_release -si)
VER := $(shell lsb_release -sr)

# Add default message
MSG := Compiling on $(OS) $(VER)
$(info $(MSG))

# Create build dirs automatically
BUILD_DIRS = build build/test build/test/toolchain build/test/toolchain/operations
$(info $(shell mkdir -p $(BUILD_DIRS)))

# Compilers are different on different machines
ifeq ($(OS), Ubuntu)
	CC := g++
else # CentOS or RHEL
	CC := /opt/rh/devtoolset-10/root/bin/g++
endif

INC := \
	-I/cad/mentor/2021.1/Mgc_home/shared/include/ \
	-Ilib/ \
	-Ilib/universal/include/ \
	-Isrc/ \
	-I.

# TODO(fpedd): Fix code and remove Wno-* flags step by step
BASE_FLAGS := \
	$(INC) \
	-DSC_INCLUDE_DYNAMIC_PROCESSES \
	-D_GLIBCXX_USE_CXX11_ABI=0 \
	-Wno-unknown-pragmas \
	-Wno-unused-but-set-variable \
	-Wno-unused-variable \
	-Wno-sign-compare \
	-Wno-bool-operation \
	-Wno-maybe-uninitialized \
	-Wno-class-memaccess \
	-Wall

ifeq ($(DEBUG), 1)
	BASE_FLAGS += -DDEBUG_LOG -g -ggdb
else 
	BASE_FLAGS += -O3 # TODO(fpedd): SystemC is not happy about this flag -DCONNECTIONS_FAST_SIM
endif

# We need to work with multiple C++ standards, as the SystemC lib is only 
# compatible with C++11 and the Universal Numbers Library requires C++17
C11FLAGS += $(BASE_FLAGS) -std=c++11 -Wno-deprecated-declarations
C17FLAGS += $(BASE_FLAGS) -std=c++17
LDFLAGS += -lsystemc
LDLIBS += -L/cad/mentor/2021.1/Mgc_home/shared/lib/

###########################################################
# Catapult Synthesis
###########################################################

# Main target to run HLS and build RTL (Verilog)
rtl: build/Catapult_Accelerator/Accelerator.v1/concat_rtl.v

# For debugging it might be beneficial to only build sub-components in RTL and 
# have them integrate into the SystemC code
InputController: build/Catapult_InputController/InputController.v1/concat_rtl.v
WeightController: build/Catapult_WeightController/WeightController.v1/concat_rtl.v
MatrixProcessor: build/Catapult_MatrixProcessor/MatrixProcessor.v1/concat_rtl.v
ProcessingElement: build/Catapult_ProcessingElement/ProcessingElement.v1/concat_rtl.v
VectorUnit: build/Catapult_VectorUnit/VectorUnit.v1/concat_rtl.v
MaxpoolUnit: build/Catapult_MaxpoolUnit/MaxpoolUnit.v1/concat_rtl.v
OutputAddressGenerator: build/Catapult_OutputAddressGenerator/OutputAddressGenerator.v1/concat_rtl.v
VectorFetchUnit: build/Catapult_VectorFetchUnit/VectorFetchUnit.v1/concat_rtl.v
VectorOpUnit: build/Catapult_VectorOpUnit/VectorOpUnit.v1/concat_rtl.v

build/Catapult_InputController/InputController.v1/concat_rtl.v: src/InputController.h
	catapult -shell -file scripts/InputController.tcl
build/Catapult_WeightController/WeightController.v1/concat_rtl.v: src/WeightController.h
	catapult -shell -file scripts/WeightController.tcl
build/Catapult_ProcessingElement/ProcessingElement.v1/concat_rtl.v: src/ProcessingElement.h
	catapult -shell -file scripts/ProcessingElement.tcl
build/Catapult_MatrixProcessor/MatrixProcessor.v1/concat_rtl.v: build/Catapult_ProcessingElement/ProcessingElement.v1/concat_rtl.v
	catapult -shell -file scripts/MatrixProcessor.tcl
build/Catapult_VectorUnit/VectorUnit.v1/concat_rtl.v: build/Catapult_MaxpoolUnit/MaxpoolUnit.v1/concat_rtl.v build/Catapult_OutputAddressGenerator/OutputAddressGenerator.v1/concat_rtl.v build/Catapult_VectorFetchUnit/VectorFetchUnit.v1/concat_rtl.v build/Catapult_VectorOpUnit/VectorOpUnit.v1/concat_rtl.v
	catapult -shell -file scripts/VectorUnit.tcl
build/Catapult_MaxpoolUnit/MaxpoolUnit.v1/concat_rtl.v: src/MaxpoolUnit.h
	catapult -shell -file scripts/MaxpoolUnit.tcl
build/Catapult_OutputAddressGenerator/OutputAddressGenerator.v1/concat_rtl.v: src/OutputAddressGenerator.h
	catapult -shell -file scripts/OutputAddressGenerator.tcl
build/Catapult_VectorFetchUnit/VectorFetchUnit.v1/concat_rtl.v: src/VectorFetch.h
	catapult -shell -file scripts/VectorFetchUnit.tcl
build/Catapult_VectorOpUnit/VectorOpUnit.v1/concat_rtl.v: src/VectorUnit.h
	catapult -shell -file scripts/VectorOpUnit.tcl
build/Catapult_Accelerator/Accelerator.v1/concat_rtl.v: build/Catapult_InputController/InputController.v1/concat_rtl.v build/Catapult_WeightController/WeightController.v1/concat_rtl.v build/Catapult_MatrixProcessor/MatrixProcessor.v1/concat_rtl.v build/Catapult_VectorUnit/VectorUnit.v1/concat_rtl.v
	catapult -shell -file scripts/Accelerator.tcl
	sed '/module CGHpart/,/endmodule/d;/module TSDN/,/endmodule/d;/module TS1N40LPB1024X128M4FWBA /,/endmodule/d;/^`include/d;s/module Accelerator_rtl/module Accelerator/g;s/VectorUnit_rtl/VectorUnit/g' build/Catapult_Accelerator/Accelerator.v1/concat_rtl.v > release/concat_rtl.v

.PHONY: rtl InputController WeightController MatrixProcessor ProcessingElement VectorUnit MaxpoolUnit OutputAddressGenerator VectorFetchUnit VectorOpUnit

###########################################################
# Cycle-accurate SystemC Simulations
###########################################################
build_folder := build/simv

simv_name := simv
ifeq ($(DEBUG), 1)
	simv_name := simv-debug
	build_folder := build/simv-debug
endif

sim_sysc:
	syscan -kdb -cflags "$(C11FLAGS) -g" -Mdir=$(build_folder) src/Accelerator.cc
	syscan -kdb -cflags "$(C11FLAGS) -g" -Mdir=$(build_folder) test/common/Harness.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/GoldModel.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/Utils.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/DataLoader.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/TestRunner.cc
	syscan -kdb -cflags "$(C11FLAGS) -g" -Mdir=$(build_folder) test/toolchain/MapOperation.cc
	vcs -full64 -sysc sc_main -kdb -debug_access+all -Mdir=$(build_folder) -o $(build_folder)/$(simv_name)
	./$(build_folder)/$(simv_name) -ucli -i dump_fsdb.tcl

sim_sysc_gui:
	syscan -kdb -cflags "$(C11FLAGS) -g" -Mdir=$(build_folder) src/Accelerator.cc
	syscan -kdb -cflags "$(C11FLAGS) -g" -Mdir=$(build_folder) test/common/Harness.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/GoldModel.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/Utils.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/DataLoader.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/TestRunner.cc
	syscan -kdb -cflags "$(C11FLAGS) -g" -Mdir=$(build_folder) test/toolchain/MapOperation.cc
	vcs -full64 -sysc sc_main -kdb -debug_access+all -Mdir=$(build_folder) -o $(build_folder)/$(simv_name)
	./$(build_folder)/$(simv_name) -verdi

rtl_sim: rtl
	cd build/Catapult_Accelerator/Accelerator.v1 && make -f ./scverify/Verify_concat_sim_rtl_v_vcs.mk SIMTOOL=vcs sim

rtl_sim_debug: rtl
	rm -rf build/inter.fsdb*
	cd build/Catapult_Accelerator/Accelerator.v1 && make -f ./scverify/Verify_concat_sim_rtl_v_vcs.mk SIMTOOL=vcs simgui

gui:
	catapult build/Catapult_debug

.PHONY: sim_sysc sim_sysc_gui rtl_sim rtl_sim_debug gui

###########################################################
# Standard Event-based SystemC Simulations
###########################################################

# Main target for accelerator simulations 
.PHONY: sim
sim: build/TestRunner
	./build/TestRunner

.PHONY: TestRunner
TestRunner: build/TestRunner

build/TestRunner: build/Accelerator.o build/Harness.o build/TestRunner.o build/GoldModel.o build/Utils.o build/DataLoader.o build/toolchain.a
	$(CC) -o $@ $^ $(LDLIBS) $(LDFLAGS)

# Unit tests for custom Posit implementation
.PHONY: PositTest
PositTest: build/PositTest

build/PositTest: test/common/PositTest.cc src/PositTypes.h
	$(CC) $(C17FLAGS) -fopenmp -DNO_SYSC $< -o $@

build/Accelerator.o: src/Accelerator.cc $(wildcard src/*.h)
	$(CC) $(C11FLAGS) -c -o $@ $< 

build/Harness.o: test/common/Harness.cc test/common/Harness.h $(wildcard src/*.h) 
	$(CC) $(C11FLAGS) -c -o $@ $<

build/GoldModel.o: test/common/GoldModel.cc test/common/GoldModel.h src/ArchitectureParams.h
	$(CC) $(C17FLAGS) -c -o $@ $<

build/Utils.o: test/common/Utils.cc test/common/Utils.h src/ArchitectureParams.h
	$(CC) $(C17FLAGS) -c -o $@ $<

build/DataLoader.o: test/common/DataLoader.cc test/common/DataLoader.h src/ArchitectureParams.h
	$(CC) $(C17FLAGS) -c -o $@ $<

build/TestRunner.o: test/common/TestRunner.cc test/simple/params.h test/resnet/params.h test/mobilebert/*.h
	$(CC) $(C17FLAGS) -c -o $@ $<

###########################################################
# Toolchain
###########################################################

TOOLCHAIN_SRC = test/toolchain/MapOperation.cc $(wildcard test/toolchain/operations/*.cc)
TOOLCHAIN_OBJ = $(addprefix build/,  $(TOOLCHAIN_SRC:.cc=.o) )

build/test/toolchain/%.o: test/toolchain/%.cc
	$(CC) $(C11FLAGS) -c -o $@ $<

.PHONY: toolchain
toolchain: build/toolchain.a

build/toolchain.a: $(TOOLCHAIN_OBJ)
	$(AR) rcs $@ $^

###########################################################
# Cleanup Targets
###########################################################

clean:
	rm -rf build/*.o build/TestRunner build/PositTest build/test/*.o build/test/toolchain/*.o build/test/toolchain/operations/*.o build/toolchain.a

clean-test:
	rm -rf test_outputs/*

clean-catapult:
	rm -rf build/Catapult_*

clean-rtl-sim:
	rm -rf build/Catapult_debug/Accelerator.v1/scverify/concat_sim_rtl_v_vcs

.PHONY: clean clean-test clean-catapult clean-rtl-sim