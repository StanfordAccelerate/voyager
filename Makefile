.DEFAULT_GOAL := TestRunner

# Detect OS
OS := $(shell lsb_release -si)
VER := $(shell lsb_release -sr)

# Add default message
MSG := Compiling on $(OS) $(VER)
$(info $(MSG))

# Build folder format is build/DATATYPE_DIMENSIONxDIMENSION
export PROJ_ROOT = $(shell pwd)
BUILD_DIR ?= build/$(DATATYPE)_$(IC_DIMENSION)x$(OC_DIMENSION)
CC_BUILD_DIR = $(BUILD_DIR)/cc
TOOLCHAIN_BUILD_DIR = $(BUILD_DIR)/cc/test/toolchain
TOOLCHAIN_BUILD_DIRS = $(TOOLCHAIN_BUILD_DIR) $(TOOLCHAIN_BUILD_DIR)/operations
ALL_BUILD_DIRS = $(CC_BUILD_DIR) $(TOOLCHAIN_BUILD_DIRS)
# Create build dirs automatically
$(info $(shell mkdir -p $(ALL_BUILD_DIRS)))

# Compilers are different on different machines
# CC := $(CATAPULT_ROOT)/bin/g++
CC := /cad/mentor/2024.1/Mgc_home/bin/g++

# Check if the environment variable is set
check_env_var:
ifndef DATATYPE
	$(error DATATYPE and other required environment variables are not set)
endif

INC := \
	-I/cad/mentor/2024.1/Mgc_home/shared/include/ \
	-Ilib/ \
	-Ilib/universal/include/ \
	-Isrc/ \
	-I.

# TODO(fpedd): Fix code and remove Wno-* flags step by step
override BASE_FLAGS += \
	$(INC) \
	-DSC_INCLUDE_DYNAMIC_PROCESSES \
	-Wno-unknown-pragmas \
	-Wno-unused-but-set-variable \
	-Wno-unused-variable \
	-Wno-sign-compare \
	-Wno-bool-operation \
	-Wno-maybe-uninitialized \
	-Wno-class-memaccess \
	-Wall \
	-Wno-bool-compare \
	-D$(DATATYPE) \
	-DIC_DIMENSION=$(IC_DIMENSION) \
	-DOC_DIMENSION=$(OC_DIMENSION)

ifeq ($(DEBUG), 1)
	override BASE_FLAGS += -DDEBUG_LOG -g -ggdb
else
	override BASE_FLAGS += -O3
endif

# We need to work with multiple C++ standards, as the SystemC lib is only
# compatible with C++11 and the Universal Numbers Library requires C++17
C11FLAGS += $(BASE_FLAGS) -std=c++11 -Wno-deprecated-declarations
C17FLAGS += $(BASE_FLAGS) -std=c++17
LDFLAGS += -lsystemc -lstdc++fs
LDLIBS += -L/cad/mentor/2024.1/Mgc_home/shared/lib/

###########################################################
# Catapult Synthesis
###########################################################
export CATAPULT_BUILD_DIR ?= $(BUILD_DIR)/Catapult/$(TECHNOLOGY)/clock_$(CLOCK_PERIOD)

# Main target to run HLS and build RTL (Verilog)
rtl: $(CATAPULT_BUILD_DIR)/Accelerator/Accelerator.v1/concat_rtl.v

# For debugging it might be beneficial to only build sub-components in RTL and
# have them integrate into the SystemC code
InputController: $(CATAPULT_BUILD_DIR)/InputController/InputController.v1/concat_rtl.v
WeightController: $(CATAPULT_BUILD_DIR)/WeightController/WeightController.v1/concat_rtl.v
# SystolicArrayRow: $(CATAPULT_BUILD_DIR)/SystolicArrayRow/SystolicArrayRow.v1/concat_rtl.v
# SystolicArrayChunk: $(CATAPULT_BUILD_DIR)/SystolicArrayChunk/SystolicArrayChunk.v1/concat_rtl.v
SystolicArray: $(CATAPULT_BUILD_DIR)/SystolicArray/SystolicArray.v1/concat_rtl.v
MatrixProcessor: $(CATAPULT_BUILD_DIR)/MatrixProcessor/MatrixProcessor.v1/concat_rtl.v
ProcessingElement: $(CATAPULT_BUILD_DIR)/ProcessingElement/ProcessingElement.v1/concat_rtl.v
VectorUnit: $(CATAPULT_BUILD_DIR)/VectorUnit/VectorUnit.v1/concat_rtl.v
MaxpoolUnit: $(CATAPULT_BUILD_DIR)/MaxpoolUnit/MaxpoolUnit.v1/concat_rtl.v
OutputAddressGenerator: $(CATAPULT_BUILD_DIR)/OutputAddressGenerator/OutputAddressGenerator.v1/concat_rtl.v
VectorFetchUnit: $(CATAPULT_BUILD_DIR)/VectorFetchUnit/VectorFetchUnit.v1/concat_rtl.v
VectorOpUnit: $(CATAPULT_BUILD_DIR)/VectorOpUnit/VectorOpUnit.v1/concat_rtl.v

$(CATAPULT_BUILD_DIR)/InputController/InputController.v1/concat_rtl.v: src/InputController.h
	BLOCK=InputController catapult -shell -file scripts/main.tcl
$(CATAPULT_BUILD_DIR)/WeightController/WeightController.v1/concat_rtl.v: src/WeightController.h
	BLOCK=WeightController catapult -shell -file scripts/main.tcl
$(CATAPULT_BUILD_DIR)/ProcessingElement/ProcessingElement.v1/concat_rtl.v: src/ProcessingElement.h
	BLOCK=ProcessingElement catapult -shell -file scripts/main.tcl
$(CATAPULT_BUILD_DIR)/SystolicArrayRow/SystolicArrayRow.v1/concat_rtl.v: src/SystolicArray.h $(CATAPULT_BUILD_DIR)/ProcessingElement/ProcessingElement.v1/concat_rtl.v
	BLOCK=SystolicArrayRow catapult -shell -file scripts/main.tcl
$(CATAPULT_BUILD_DIR)/SystolicArrayChunk/SystolicArrayChunk.v1/concat_rtl.v: src/SystolicArray.h $(CATAPULT_BUILD_DIR)/SystolicArrayRow/SystolicArrayRow.v1/concat_rtl.v
	BLOCK=SystolicArrayChunk catapult -shell -file scripts/main.tcl
$(CATAPULT_BUILD_DIR)/SystolicArray/SystolicArray.v1/concat_rtl.v: src/SystolicArray.h $(CATAPULT_BUILD_DIR)/ProcessingElement/ProcessingElement.v1/concat_rtl.v
	BLOCK=SystolicArray catapult -shell -file scripts/main.tcl
$(CATAPULT_BUILD_DIR)/MatrixProcessor/MatrixProcessor.v1/concat_rtl.v: src/MatrixProcessor.h src/SystolicArray.h src/Skewer.h $(CATAPULT_BUILD_DIR)/SystolicArray/SystolicArray.v1/concat_rtl.v
	BLOCK=MatrixProcessor catapult -shell -file scripts/main.tcl
$(CATAPULT_BUILD_DIR)/VectorUnit/VectorUnit.v1/concat_rtl.v: $(CATAPULT_BUILD_DIR)/MaxpoolUnit/MaxpoolUnit.v1/concat_rtl.v $(CATAPULT_BUILD_DIR)/OutputAddressGenerator/OutputAddressGenerator.v1/concat_rtl.v $(CATAPULT_BUILD_DIR)/VectorFetchUnit/VectorFetchUnit.v1/concat_rtl.v $(CATAPULT_BUILD_DIR)/VectorOpUnit/VectorOpUnit.v1/concat_rtl.v
	BLOCK=VectorUnit catapult -shell -file scripts/main.tcl
$(CATAPULT_BUILD_DIR)/MaxpoolUnit/MaxpoolUnit.v1/concat_rtl.v: src/MaxpoolUnit.h
	BLOCK=MaxpoolUnit catapult -shell -file scripts/main.tcl
$(CATAPULT_BUILD_DIR)/OutputAddressGenerator/OutputAddressGenerator.v1/concat_rtl.v: src/OutputAddressGenerator.h
	BLOCK=OutputAddressGenerator catapult -shell -file scripts/main.tcl
$(CATAPULT_BUILD_DIR)/VectorFetchUnit/VectorFetchUnit.v1/concat_rtl.v: src/VectorFetch.h
	BLOCK=VectorFetchUnit catapult -shell -file scripts/main.tcl
$(CATAPULT_BUILD_DIR)/VectorOpUnit/VectorOpUnit.v1/concat_rtl.v: src/VectorUnit.h
	BLOCK=VectorOpUnit catapult -shell -file scripts/main.tcl
$(CATAPULT_BUILD_DIR)/Accelerator/Accelerator.v1/concat_rtl.v: $(CATAPULT_BUILD_DIR)/InputController/InputController.v1/concat_rtl.v $(CATAPULT_BUILD_DIR)/WeightController/WeightController.v1/concat_rtl.v $(CATAPULT_BUILD_DIR)/MatrixProcessor/MatrixProcessor.v1/concat_rtl.v $(CATAPULT_BUILD_DIR)/VectorUnit/VectorUnit.v1/concat_rtl.v
	BLOCK=Accelerator catapult -shell -file scripts/main.tcl
	# Keeping this in case an older version of Catapult is used
	sed '/module CGHpart/,/endmodule/d;/module TSDN/,/endmodule/d;/module TS1N40LPB1024X128M4FWBA /,/endmodule/d;/module TS1N40LPB1024X64M4FW /,/endmodule/d;/^`include/d;s/module Accelerator_rtl/module Accelerator/g;s/VectorUnit_rtl/VectorUnit/g' $(CATAPULT_BUILD_DIR)/Accelerator/Accelerator.v1/concat_rtl.v > release/concat_rtl.v
	# Uncommenting memoery macro module in intel16
	awk -i inplace ' \
  /\/\*/ { comment = 1 } \
  /\*\// { comment = 1 } \
  /module intel16_1024x.*_rf_wrapper/ { within_block = 1 } \
  !within_block || within_block && !comment { print } \
  within_block && /endmodule/ { within_block = 0 } \
  comment = 0 \
' release/concat_rtl.v

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
	syscan -kdb -cflags "$(C11FLAGS) -g" -Mdir=$(build_folder) src/Accelerator.h
	syscan -kdb -cflags "$(C11FLAGS) -g" -Mdir=$(build_folder) test/common/Harness.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/GoldModel.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/Utils.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/DataLoader.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/TestRunner.cc
	syscan -kdb -cflags "$(C11FLAGS) -g" -Mdir=$(build_folder) test/toolchain/MapOperation.cc
	vcs -full64 -sysc sc_main -kdb -debug_access+all -Mdir=$(build_folder) -o $(build_folder)/$(simv_name)
	./$(build_folder)/$(simv_name) -ucli -i dump_fsdb.tcl

sim_sysc_gui:
	syscan -kdb -cflags "$(C11FLAGS) -g" -Mdir=$(build_folder) src/Accelerator.h
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
sim: $(CC_BUILD_DIR)/TestRunner
	./$(CC_BUILD_DIR)/TestRunner

.PHONY: fast-sim
fast-sim: $(CC_BUILD_DIR)/TestRunner-fast
	./$(CC_BUILD_DIR)/TestRunner-fast

.PHONY: sim-debug
sim-debug: $(CC_BUILD_DIR)/TestRunner
	gdb ./$(CC_BUILD_DIR)/TestRunner

.PHONY: TestRunner
TestRunner: check_env_var $(CC_BUILD_DIR)/TestRunner

$(CC_BUILD_DIR)/TestRunner: $(CC_BUILD_DIR)/Harness.o $(CC_BUILD_DIR)/TestRunner.o $(CC_BUILD_DIR)/GoldModel.o $(CC_BUILD_DIR)/Utils.o $(CC_BUILD_DIR)/MemoryModel.o $(CC_BUILD_DIR)/SimpleMemoryModel.o $(CC_BUILD_DIR)/Simulation.o $(CC_BUILD_DIR)/networks.a $(CC_BUILD_DIR)/toolchain.a
	$(CC) -o $@ $^ $(LDLIBS) $(LDFLAGS)

$(CC_BUILD_DIR)/TestRunner-fast: $(CC_BUILD_DIR)/Harness-fast.o $(CC_BUILD_DIR)/TestRunner.o $(CC_BUILD_DIR)/GoldModel.o $(CC_BUILD_DIR)/Utils.o $(CC_BUILD_DIR)/MemoryModel.o $(CC_BUILD_DIR)/SimpleMemoryModel.o $(CC_BUILD_DIR)/Simulation.o $(CC_BUILD_DIR)/networks.a $(CC_BUILD_DIR)/toolchain.a
	$(CC) -o $@ $^ $(LDLIBS) $(LDFLAGS)

.PHONY: MobileBERTAccuracy
MobileBERTAccuracy: $(CC_BUILD_DIR)/AccuracyTester
	./$(CC_BUILD_DIR)/AccuracyTester mobilebert models/mobilebert/binary_data/tiny_truncated_sst2/ 64

.PHONY: ResNetAccuracy
ResNetAccuracy: $(CC_BUILD_DIR)/AccuracyTester
	./$(CC_BUILD_DIR)/AccuracyTester resnet18 models/resnet/binary_data/imagenet_1000/

.PHONY: AccuracyTester
AccuracyTester: $(CC_BUILD_DIR)/AccuracyTester

$(CC_BUILD_DIR)/AccuracyTester: $(CC_BUILD_DIR)/AccuracyTester.o $(CC_BUILD_DIR)/GoldModel.o $(CC_BUILD_DIR)/Utils.o $(CC_BUILD_DIR)/MemoryModel.o $(CC_BUILD_DIR)/SimpleMemoryModel.o $(CC_BUILD_DIR)/networks.a
	$(CC) -o $@ $^ -lstdc++fs -pthread

.PHONY: MobileBERTFinetuning
MobileBERTFinetuning: $(CC_BUILD_DIR)/Finetuning
	./$(CC_BUILD_DIR)/Finetuning

$(CC_BUILD_DIR)/Finetuning: $(CC_BUILD_DIR)/Finetuning.o $(CC_BUILD_DIR)/MobileBERTParams.o $(CC_BUILD_DIR)/DatasetIterator.o $(CC_BUILD_DIR)/GoldModel.o $(CC_BUILD_DIR)/Utils.o $(CC_BUILD_DIR)/MemoryModel.o $(CC_BUILD_DIR)/SimpleMemoryModel.o $(CC_BUILD_DIR)/networks.a
	$(CC) -o $@ $^ -lstdc++fs

# Unit tests for custom Posit implementation
.PHONY: PositTest
PositTest: $(CC_BUILD_DIR)/PositTest

$(CC_BUILD_DIR)/PositTest: test/common/PositTest.cc src/PositTypes.h
	$(CC) $(C17FLAGS) -fopenmp -DNO_SYSC $< -o $@

$(CC_BUILD_DIR)/Harness.o: test/common/Harness.cc test/common/Harness.h $(wildcard src/*.h)
	$(CC) $(C11FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/Harness-fast.o: test/common/Harness.cc test/common/Harness.h $(wildcard src/*.h)
	$(CC) $(C11FLAGS) -DCONNECTIONS_FAST_SIM -c -o $@ $<

$(CC_BUILD_DIR)/GoldModel.o: test/common/GoldModel.cc test/common/GoldModel.h src/ArchitectureParams.h src/PositTypes.h src/StdFloatTypes.h
	$(CC) $(C17FLAGS) -g -c -o $@ $<

$(CC_BUILD_DIR)/Utils.o: test/common/Utils.cc test/common/Utils.h src/ArchitectureParams.h src/PositTypes.h src/StdFloatTypes.h
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/MemoryModel.o: test/common/MemoryModel.cc test/common/MemoryModel.h src/ArchitectureParams.h
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/SimpleMemoryModel.o: test/common/SimpleMemoryModel.cc test/common/SimpleMemoryModel.h src/ArchitectureParams.h src/PositTypes.h src/StdFloatTypes.h
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/Simulation.o: test/common/Simulation.cc test/common/Simulation.h src/ArchitectureParams.h src/PositTypes.h src/StdFloatTypes.h
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/TestRunner.o: test/common/TestRunner.cc
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/AccuracyTester.o: test/common/AccuracyTester.cc src/PositTypes.h src/StdFloatTypes.h
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/Finetuning.o: test/training/Finetuning.cc test/training/forward_pass.h test/training/backward_pass.h test/training/model_arch.h test/training/memory_plan.h test/training/DTYPE.h
	$(CC) $(C17FLAGS) -g -c -o $@ $<

$(CC_BUILD_DIR)/MobileBERTParams.o: test/training/MobileBERTParams.cc test/mobilebert/mobilebert_tiny2/*.h
	$(CC) $(C17FLAGS) -g -c -o $@ $<

$(CC_BUILD_DIR)/DatasetIterator.o: test/training/DatasetIterator.cc
	$(CC) $(C17FLAGS) -g -c -o $@ $<

###########################################################
# Networks
###########################################################
.PHONY: networks
networks: $(CC_BUILD_DIR)/networks.a

$(CC_BUILD_DIR)/networks.a: $(CC_BUILD_DIR)/ResNet.o $(CC_BUILD_DIR)/MobileBERT.o $(CC_BUILD_DIR)/Generic.o
	$(AR) rcs $@ $^

$(CC_BUILD_DIR)/ResNet.o: test/resnet/ResNet.cc test/resnet/*.h
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/MobileBERT.o: test/mobilebert/MobileBERT.cc test/mobilebert/*.h test/mobilebert/mobilebert_tiny2/*.h test/common/VerificationTypes.h
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/Generic.o: test/generic/Generic.cc
	$(CC) $(C17FLAGS) -c -o $@ $<

###########################################################
# Toolchain
###########################################################

TOOLCHAIN_SRC = test/toolchain/MapOperation.cc $(wildcard test/toolchain/operations/*.cc)
TOOLCHAIN_OBJ = $(addprefix $(CC_BUILD_DIR)/,  $(TOOLCHAIN_SRC:.cc=.o) )

$(CC_BUILD_DIR)/test/toolchain/%.o: test/toolchain/%.cc
	$(CC) $(C11FLAGS) -c -o $@ $<

.PHONY: toolchain
toolchain: $(CC_BUILD_DIR)/toolchain.a

$(CC_BUILD_DIR)/toolchain.a: $(TOOLCHAIN_OBJ)
	$(AR) rcs $@ $^

###########################################################
# Cleanup Targets
###########################################################

clean-all:
	rm -rf build/*

clean: check_env_var
	rm -rf $(CC_BUILD_DIR)

clean-test:
	rm -rf test_outputs/*

clean-catapult: check_env_var
	rm -rf $(CATAPULT_BUILD_DIR)

clean-rtl-sim:
	rm -rf build/Catapult_debug/Accelerator.v1/scverify/concat_sim_rtl_v_vcs

.PHONY: clean clean-test clean-catapult clean-rtl-sim
