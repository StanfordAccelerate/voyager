.DEFAULT_GOAL := TestRunner

# Build folder format is build/DATATYPE_DIMENSIONxDIMENSION
export PROJ_ROOT = $(shell pwd)
# TODO: buffer size info is not in the build dir name currently
BUILD_DIR ?= build/$(DATATYPE)_$(IC_DIMENSION)x$(OC_DIMENSION)
CC_BUILD_DIR = $(BUILD_DIR)/cc
ALL_BUILD_DIRS = $(CC_BUILD_DIR) $(TOOLCHAIN_BUILD_DIRS)
# Create build dirs automatically
$(info $(shell mkdir -p $(ALL_BUILD_DIRS)))

# Compilers are different on different machines
CC := $(CATAPULT_ROOT)/bin/g++

export CODEGEN_DIR ?= test/compiler

# Check if the environment variable is set
check_env_var:
ifndef DATATYPE
	$(error DATATYPE environment variables are not set)
endif
ifndef IC_DIMENSION
	$(error IC_DIMENSION environment variables are not set)
endif
ifndef OC_DIMENSION
	$(error OC_DIMENSION environment variables are not set)
endif

INC := \
	-I$(CATAPULT_ROOT)/shared/include/ \
	-Ilib/ \
	-Ilib/xtensor/include \
	-Ilib/xtl/include \
	-Ilib/spdlog/include \
	-Isrc/ \
	-I$(CONDA_PREFIX)/include \
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
	-DSPDLOG_COMPILED_LIB \
	-DSPDLOG_EOL=\"\" \
	-D$(DATATYPE) \
	-DIC_DIMENSION=$(IC_DIMENSION) \
	-DOC_DIMENSION=$(OC_DIMENSION)

ifndef INPUT_BUFFER_SIZE
	export INPUT_BUFFER_SIZE = 1024
else
	override BASE_FLAGS += -DINPUT_BUFFER_SIZE=$(INPUT_BUFFER_SIZE)
endif

ifndef WEIGHT_BUFFER_SIZE
	export WEIGHT_BUFFER_SIZE = 1024
else
	override BASE_FLAGS += -DWEIGHT_BUFFER_SIZE=$(WEIGHT_BUFFER_SIZE)
endif

ifndef ACCUM_BUFFER_SIZE
	export ACCUM_BUFFER_SIZE = 1024
else
	override BASE_FLAGS += -DACCUM_BUFFER_SIZE=$(ACCUM_BUFFER_SIZE)
endif

ifeq ($(DEBUG), 1)
	override BASE_FLAGS += -DDEBUG -g -ggdb
else
	override BASE_FLAGS += -O3
endif

# We need to work with multiple C++ standards, as the SystemC lib is only
# compatible with C++11 and the Universal Numbers Library requires C++17
C17FLAGS += $(BASE_FLAGS) -std=c++17 -Wno-deprecated-declarations
LDFLAGS += -lsystemc -lstdc++fs -labsl_hash -labsl_log_internal_check_op -labsl_log_internal_message -labsl_log_internal_nullguard -lprotobuf -lpthread -Wl,-rpath=$(CONDA_PREFIX)/lib
LDLIBS += -L$(CATAPULT_ROOT)/shared/lib/ -L$(CONDA_PREFIX)/lib
LDFLAGS_NO_SYSC += -lstdc++fs -labsl_hash -labsl_log_internal_check_op -labsl_log_internal_message -labsl_log_internal_nullguard -lprotobuf -lpthread -Wl,-rpath=$(CONDA_PREFIX)/lib
LDLIBS_NO_SYSC += -L$(CONDA_PREFIX)/lib

###########################################################
# spdlog
###########################################################
SPDLOG_OBJ_FILES = $(patsubst lib/spdlog/src/%.cpp,$(CC_BUILD_DIR)/spdlog_%.o,$(wildcard lib/spdlog/src/*.cpp))

$(CC_BUILD_DIR)/spdlog_%.o: lib/spdlog/src/%.cpp
	$(CC) $(C17FLAGS) -c -o $@ $<

###########################################################
# Catapult Synthesis
###########################################################
export CATAPULT_BUILD_DIR ?= $(BUILD_DIR)/Catapult/$(TECHNOLOGY)/clock_$(CLOCK_PERIOD)

# Main target to run HLS and build RTL (Verilog)
rtl: Accelerator


# For debugging it might be beneficial to only build sub-components in RTL and
# have them integrate into the SystemC code
InputController: $(CATAPULT_BUILD_DIR)/InputController/InputController.v1/concat_rtl.v
WeightController: $(CATAPULT_BUILD_DIR)/WeightController/WeightController.v1/concat_rtl.v
# SystolicArrayRow: $(CATAPULT_BUILD_DIR)/SystolicArrayRow/SystolicArrayRow.v1/concat_rtl.v
# SystolicArrayChunk: $(CATAPULT_BUILD_DIR)/SystolicArrayChunk/SystolicArrayChunk.v1/concat_rtl.v
SystolicArray: $(CATAPULT_BUILD_DIR)/SystolicArray/SystolicArray.v1/concat_rtl.v
MatrixProcessor: $(CATAPULT_BUILD_DIR)/MatrixProcessor/MatrixProcessor.v1/concat_rtl.v
ProcessingElement: $(CATAPULT_BUILD_DIR)/ProcessingElement/ProcessingElement.v1/concat_rtl.v
VectorFetchUnit: $(CATAPULT_BUILD_DIR)/VectorFetchUnit/VectorFetchUnit.v1/concat_rtl.v
VectorUnit: $(CATAPULT_BUILD_DIR)/VectorUnit/VectorUnit.v1/concat_rtl.v
VectorUnitOutput: $(CATAPULT_BUILD_DIR)/VectorUnitOutput/VectorUnitOutput.v1/concat_rtl.v
VectorOpUnit: $(CATAPULT_BUILD_DIR)/VectorOpUnit/VectorOpUnit.v1/concat_rtl.v
Accelerator: $(CATAPULT_BUILD_DIR)/Accelerator/Accelerator.v1/concat_rtl.v

$(CATAPULT_BUILD_DIR)/InputController/InputController.v1/concat_rtl.v: src/InputController.h
	BLOCK=InputController catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/InputController.log
$(CATAPULT_BUILD_DIR)/WeightController/WeightController.v1/concat_rtl.v: src/WeightController.h
	BLOCK=WeightController catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/WeightController.log
$(CATAPULT_BUILD_DIR)/ProcessingElement/ProcessingElement.v1/concat_rtl.v: src/ProcessingElement.h
	BLOCK=ProcessingElement catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/ProcessingElement.log
$(CATAPULT_BUILD_DIR)/SystolicArrayRow/SystolicArrayRow.v1/concat_rtl.v: src/SystolicArray.h $(CATAPULT_BUILD_DIR)/ProcessingElement/ProcessingElement.v1/concat_rtl.v
	BLOCK=SystolicArrayRow catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/SystolicArrayRow.log
$(CATAPULT_BUILD_DIR)/SystolicArrayChunk/SystolicArrayChunk.v1/concat_rtl.v: src/SystolicArray.h $(CATAPULT_BUILD_DIR)/SystolicArrayRow/SystolicArrayRow.v1/concat_rtl.v
	BLOCK=SystolicArrayChunk catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/SystolicArrayChunk.log
$(CATAPULT_BUILD_DIR)/SystolicArray/SystolicArray.v1/concat_rtl.v: src/SystolicArray.h $(CATAPULT_BUILD_DIR)/ProcessingElement/ProcessingElement.v1/concat_rtl.v
	BLOCK=SystolicArray catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/SystolicArray.log
$(CATAPULT_BUILD_DIR)/MatrixProcessor/MatrixProcessor.v1/concat_rtl.v: src/MatrixProcessor.h src/SystolicArray.h src/Skewer.h $(CATAPULT_BUILD_DIR)/SystolicArray/SystolicArray.v1/concat_rtl.v
	BLOCK=MatrixProcessor catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/MatrixProcessor.log
$(CATAPULT_BUILD_DIR)/VectorUnit/VectorUnit.v1/concat_rtl.v: $(CATAPULT_BUILD_DIR)/VectorFetchUnit/VectorFetchUnit.v1/concat_rtl.v $(CATAPULT_BUILD_DIR)/VectorOpUnit/VectorOpUnit.v1/concat_rtl.v $(CATAPULT_BUILD_DIR)/VectorUnitOutput/VectorUnitOutput.v1/concat_rtl.v $(CATAPULT_BUILD_DIR)/VectorParamsDeserializer/VectorParamsDeserializer.v1/concat_rtl.v
	BLOCK=VectorUnit catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/VectorUnit.log
$(CATAPULT_BUILD_DIR)/VectorFetchUnit/VectorFetchUnit.v1/concat_rtl.v: src/VectorFetch.h
	BLOCK=VectorFetchUnit catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/VectorFetchUnit.log
$(CATAPULT_BUILD_DIR)/VectorParamsDeserializer/VectorParamsDeserializer.v1/concat_rtl.v: src/VectorUnit.h
	BLOCK=VectorParamsDeserializer catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/VectorParamsDeserializer.log
$(CATAPULT_BUILD_DIR)/VectorOpUnit/VectorOpUnit.v1/concat_rtl.v: src/VectorUnit.h
	BLOCK=VectorOpUnit catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/VectorOpUnit.log
$(CATAPULT_BUILD_DIR)/VectorUnitOutput/VectorUnitOutput.v1/concat_rtl.v: src/VectorUnitOutput.h
	BLOCK=VectorUnitOutput catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/VectorUnitOutput.log
$(CATAPULT_BUILD_DIR)/Accelerator/Accelerator.v1/concat_rtl.v: $(CATAPULT_BUILD_DIR)/InputController/InputController.v1/concat_rtl.v $(CATAPULT_BUILD_DIR)/WeightController/WeightController.v1/concat_rtl.v $(CATAPULT_BUILD_DIR)/MatrixProcessor/MatrixProcessor.v1/concat_rtl.v $(CATAPULT_BUILD_DIR)/VectorUnit/VectorUnit.v1/concat_rtl.v
	BLOCK=Accelerator catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/Accelerator.log

.PHONY: rtl Accelerator InputController WeightController MatrixProcessor ProcessingElement VectorUnit VectorFetchUnit VectorOpUnit VectorUnitOutput

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
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) src/Accelerator.h
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/Harness.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/GoldModel.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/Utils.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/DataLoader.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/TestRunner.cc
	vcs -full64 -sysc sc_main -kdb -debug_access+all -Mdir=$(build_folder) -o $(build_folder)/$(simv_name)
	./$(build_folder)/$(simv_name) -ucli -i dump_fsdb.tcl

sim_sysc_gui:
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) src/Accelerator.h
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/Harness.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/GoldModel.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/Utils.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/DataLoader.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/TestRunner.cc
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
sim: $(CC_BUILD_DIR)/TestRunner network-proto
	./$(CC_BUILD_DIR)/TestRunner

.PHONY: fast-sim
fast-sim: $(CC_BUILD_DIR)/TestRunner-fast network-proto
	./$(CC_BUILD_DIR)/TestRunner-fast

.PHONY: fast-sim-check
fast-sim-check: $(CC_BUILD_DIR)/TestRunner-checker network-proto
	./$(CC_BUILD_DIR)/TestRunner-checker

.PHONY: sim-debug
sim-debug: $(CC_BUILD_DIR)/TestRunner network-proto
	gdb ./$(CC_BUILD_DIR)/TestRunner


.PHONY: fast-sim-debug
fast-sim-debug: $(CC_BUILD_DIR)/TestRunner-fast network-proto
	gdb ./$(CC_BUILD_DIR)/TestRunner-fast

.PHONY: TestRunner
TestRunner: check_env_var $(CC_BUILD_DIR)/TestRunner

.PHONY: TestRunner-fast
TestRunner-fast: check_env_var $(CC_BUILD_DIR)/TestRunner-fast

.PHONY: TestRunner-checker
TestRunner-checker: check_env_var $(CC_BUILD_DIR)/TestRunner-checker

.PHONY: AccuracyTester
AccuracyTester: ./$(CC_BUILD_DIR)/AccuracyTester

.PHONY: MobileBertAccuracy
MobileBertAccuracy: $(CC_BUILD_DIR)/AccuracyTester
	./$(CC_BUILD_DIR)/AccuracyTester mobilebert data/bert_sst2_val 64

.PHONY: ResNetAccuracy
ResNetAccuracy: $(CC_BUILD_DIR)/AccuracyTester
	./$(CC_BUILD_DIR)/AccuracyTester resnet18 data/imagenet_val 64

$(CC_BUILD_DIR)/TestRunner: $(CC_BUILD_DIR)/Harness.o $(CC_BUILD_DIR)/TestRunner.o $(CC_BUILD_DIR)/GoldModel.o $(CC_BUILD_DIR)/Utils.o $(CC_BUILD_DIR)/Simulation.o $(CC_BUILD_DIR)/ArrayMemory.o $(CC_BUILD_DIR)/DataLoader.o $(CC_BUILD_DIR)/Network.o $(CC_BUILD_DIR)/param.pb.o $(CC_BUILD_DIR)/tiling.pb.o $(CC_BUILD_DIR)/MapOperation.o $(CC_BUILD_DIR)/AccessCounter.o $(CC_BUILD_DIR)/Tiling.o  $(SPDLOG_OBJ_FILES)
	$(CC) -o $@ $^ $(LDLIBS) $(LDFLAGS)

$(CC_BUILD_DIR)/TestRunner-fast: $(CC_BUILD_DIR)/Harness-fast.o $(CC_BUILD_DIR)/TestRunner.o $(CC_BUILD_DIR)/GoldModel.o $(CC_BUILD_DIR)/Utils.o $(CC_BUILD_DIR)/Simulation.o $(CC_BUILD_DIR)/ArrayMemory.o $(CC_BUILD_DIR)/DataLoader.o $(CC_BUILD_DIR)/Network.o $(CC_BUILD_DIR)/param.pb.o $(CC_BUILD_DIR)/tiling.pb.o $(CC_BUILD_DIR)/MapOperation.o $(CC_BUILD_DIR)/AccessCounter.o $(CC_BUILD_DIR)/Tiling.o $(SPDLOG_OBJ_FILES)
	$(CC) -o $@ $^ $(LDLIBS) $(LDFLAGS)

$(CC_BUILD_DIR)/TestRunner-checker: $(CC_BUILD_DIR)/Harness-checker.o $(CC_BUILD_DIR)/TestRunner.o $(CC_BUILD_DIR)/GoldModel-checker.o $(CC_BUILD_DIR)/Utils.o $(CC_BUILD_DIR)/Simulation.o $(CC_BUILD_DIR)/ArrayMemory.o $(CC_BUILD_DIR)/DataLoader.o $(CC_BUILD_DIR)/Network.o $(CC_BUILD_DIR)/param.pb.o  $(CC_BUILD_DIR)/tiling.pb.o $(CC_BUILD_DIR)/MapOperation.o $(CC_BUILD_DIR)/PEChecker.o $(CC_BUILD_DIR)/AccessCounter.o $(CC_BUILD_DIR)/Tiling.o $(SPDLOG_OBJ_FILES)
	$(CC) -o $@ $^ $(LDLIBS) $(LDFLAGS)

$(CC_BUILD_DIR)/AccuracyTester: $(CC_BUILD_DIR)/AccuracyTester.o $(CC_BUILD_DIR)/GoldModel.o $(CC_BUILD_DIR)/Utils.o $(CC_BUILD_DIR)/ArrayMemory.o $(CC_BUILD_DIR)/DataLoader.o $(CC_BUILD_DIR)/Network.o $(CC_BUILD_DIR)/param.pb.o $(CC_BUILD_DIR)/tiling.pb.o $(CC_BUILD_DIR)/Tiling.o $(SPDLOG_OBJ_FILES)
	$(CC) -o $@ $^ $(LDLIBS_NO_SYSC) $(LDFLAGS_NO_SYSC) -pthread

$(CC_BUILD_DIR)/Harness.o: test/common/Harness.cc test/common/Harness.h test/common/VerificationTypes.h test/toolchain/MapOperation.h $(wildcard src/*.h)
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/Harness-fast.o: test/common/Harness.cc test/common/Harness.h test/common/VerificationTypes.h test/toolchain/MapOperation.h $(wildcard src/*.h)
	$(CC) $(C17FLAGS) -DCONNECTIONS_FAST_SIM -c -o $@ $<

$(CC_BUILD_DIR)/Harness-checker.o: test/common/Harness.cc test/common/Harness.h test/common/VerificationTypes.h test/toolchain/MapOperation.h $(wildcard src/*.h) test/checker/PEChecker.h
	$(CC) $(C17FLAGS) -DCONNECTIONS_FAST_SIM -DCHECK_PE -c -o $@ $<

$(CC_BUILD_DIR)/GoldModel.o: test/common/GoldModel.cc test/common/GoldModel.h test/common/VerificationTypes.h src/ArchitectureParams.h src/PositTypes.h src/StdFloatTypes.h src/IntTypes.h $(wildcard test/common/operations/*.h)
	$(CC) $(C17FLAGS) -g -c -o $@ $<

$(CC_BUILD_DIR)/GoldModel-checker.o: test/common/GoldModel.cc test/common/GoldModel.h test/common/VerificationTypes.h src/ArchitectureParams.h src/PositTypes.h src/StdFloatTypes.h src/IntTypes.h $(wildcard test/common/operations/*.h) test/checker/PEChecker.h
	$(CC) $(C17FLAGS) -DCHECK_PE -g -c -o $@ $<

$(CC_BUILD_DIR)/Utils.o: test/common/Utils.cc test/common/Utils.h src/ArchitectureParams.h src/PositTypes.h src/StdFloatTypes.h src/IntTypes.h
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/Simulation.o: test/common/Simulation.cc test/common/Simulation.h src/ArchitectureParams.h src/PositTypes.h src/StdFloatTypes.h src/IntTypes.h test/common/VerificationTypes.h
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/ArrayMemory.o: test/common/ArrayMemory.cc test/common/ArrayMemory.h
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/DataLoader.o: test/common/DataLoader.cc test/common/DataLoader.h
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/MapOperation.o: test/toolchain/MapOperation.cc $(wildcard test/toolchain/*.h)
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/TestRunner.o: test/common/TestRunner.cc
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/PEChecker.o: test/checker/PEChecker.cc test/checker/PEChecker.h src/PositTypes.h src/StdFloatTypes.h src/IntTypes.h
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/AccuracyTester.o: test/common/AccuracyTester.cc src/PositTypes.h src/StdFloatTypes.h src/IntTypes.h $(wildcard test/toolchain/*.h)
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/AccessCounter.o: test/common/AccessCounter.cc test/common/AccessCounter.h
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/Tiling.o: test/common/Tiling.cc test/common/Tiling.h
	$(CC) $(C17FLAGS) -c -o $@ $<

###########################################################
# Networks
###########################################################
$(CC_BUILD_DIR)/Network.o: test/common/Network.cc test/compiler/proto/param.pb.cc
	$(CC) $(C17FLAGS) -c -o $@ $<

test/compiler/proto/param.pb.cc: quantized-training/src/quantized_training/codegen/param.proto
	protoc -I=quantized-training/src/quantized_training/codegen --cpp_out=test/compiler/proto $<

test/compiler/proto/tiling_pb2.py: test/compiler/proto/tiling.proto
	protoc --proto_path=test/compiler/proto/ --python_out=test/compiler/proto $<

test/compiler/proto/tiling.pb.cc: test/compiler/proto/tiling.proto
	protoc -I=test/compiler/proto --cpp_out=test/compiler/proto $<

$(CC_BUILD_DIR)/param.pb.o: test/compiler/proto/param.pb.cc
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/tiling.pb.o: test/compiler/proto/tiling.pb.cc
	$(CC) $(C17FLAGS) -c -o $@ $<

.PHONY: network-proto
network-proto: $(CODEGEN_DIR)/networks/$(NETWORK)/$(DATATYPE)/model.txt test/compiler/proto/param.pb.cc test/compiler/proto/tiling_pb2.py test/compiler/proto/tiling.pb.cc $(CODEGEN_DIR)/networks/$(NETWORK)/$(DATATYPE)/$(IC_DIMENSION)x$(OC_DIMENSION)_$(INPUT_BUFFER_SIZE)x$(WEIGHT_BUFFER_SIZE)x$(ACCUM_BUFFER_SIZE)/tilings.txtpb

include codegen.mk

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

clean-rtl-sim: check_env_var
	rm -rf $(CATAPULT_BUILD_DIR)/Accelerator/Accelerator.v1/scverify/concat_sim_rtl_v_vcs

clean-protos:
	rm -rf $(CODEGEN_DIR)/networks/*
	rm -rf test/compiler/proto/*.pb.*

.PHONY: clean clean-test clean-catapult clean-rtl-sim
