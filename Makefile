.DEFAULT_GOAL := TestRunner

export PROJ_ROOT = $(shell pwd)

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

ifndef DOUBLE_BUFFERED_ACCUM_BUFFER
	export DOUBLE_BUFFERED_ACCUM_BUFFER = false
else
	override BASE_FLAGS += -DDOUBLE_BUFFERED_ACCUM_BUFFER=$(DOUBLE_BUFFERED_ACCUM_BUFFER)
endif

ifndef SUPPORT_MVM
	export SUPPORT_MVM = false
else
	override BASE_FLAGS += -DSUPPORT_MVM=$(SUPPORT_MVM)
endif

ifndef SUPPORT_SPMM
	export SUPPORT_SPMM = false
else
	override BASE_FLAGS += -DSUPPORT_SPMM=$(SUPPORT_SPMM)
endif

ifndef SUPPORT_DWC
	export SUPPORT_DWC = false
else
	override BASE_FLAGS += -DSUPPORT_DWC=$(SUPPORT_DWC)
endif

ifdef CLOCK_PERIOD
	override BASE_FLAGS += -DCLOCK_PERIOD=$(CLOCK_PERIOD)
endif

ifeq ($(DEBUG), 1)
	override BASE_FLAGS += -DDEBUG -g -O0 -ggdb
else
	override BASE_FLAGS += -O3
endif

ifdef ZERO_INIT
	override BASE_FLAGS += -DZERO_INIT
endif

# We need to work with multiple C++ standards, as the SystemC lib is only
# compatible with C++11 and the Universal Numbers Library requires C++17
C17FLAGS += $(BASE_FLAGS) -std=c++17 -Wno-deprecated-declarations
LDFLAGS += -lsystemc -lstdc++fs -labsl_hash -labsl_log_internal_check_op -labsl_log_internal_message -labsl_log_internal_nullguard -lprotobuf -lpthread -Wl,-rpath=$(CONDA_PREFIX)/lib
LDLIBS += -L$(CATAPULT_ROOT)/shared/lib/ -L$(CONDA_PREFIX)/lib
LDFLAGS_NO_SYSC += -lstdc++fs -labsl_hash -labsl_log_internal_check_op -labsl_log_internal_message -labsl_log_internal_nullguard -lprotobuf -lpthread -Wl,-rpath=$(CONDA_PREFIX)/lib
LDLIBS_NO_SYSC += -L$(CONDA_PREFIX)/lib

###########################################################
# Build Directories
###########################################################
BUILD_DIR ?= build/$(DATATYPE)_$(IC_DIMENSION)x$(OC_DIMENSION)_$(INPUT_BUFFER_SIZE)x$(WEIGHT_BUFFER_SIZE)x$(ACCUM_BUFFER_SIZE)_$(DOUBLE_BUFFERED_ACCUM_BUFFER)_$(SUPPORT_MVM)_$(SUPPORT_SPMM)
CC_BUILD_DIR = $(BUILD_DIR)/cc
ALL_BUILD_DIRS = $(CC_BUILD_DIR) $(TOOLCHAIN_BUILD_DIRS)
# Create build dirs automatically
$(info $(shell mkdir -p $(ALL_BUILD_DIRS)))

###########################################################
# spdlog
###########################################################
SPDLOG_OBJ_FILES = $(patsubst lib/spdlog/src/%.cpp,$(CC_BUILD_DIR)/spdlog_%.o,$(wildcard lib/spdlog/src/*.cpp))

$(CC_BUILD_DIR)/spdlog_%.o: lib/spdlog/src/%.cpp
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/spdlog.o: $(SPDLOG_OBJ_FILES)
	ld -r -o $@ $^

spdlog: $(CC_BUILD_DIR)/spdlog.o

###########################################################
# Catapult Synthesis
###########################################################
export CATAPULT_BUILD_DIR ?= $(BUILD_DIR)/Catapult/$(TECHNOLOGY)/clock_$(CLOCK_PERIOD)

# Main target to run HLS and build RTL (Verilog)
rtl: Accelerator

# Generating RTL requires test/compiler/proto/{param.pb.cc, tiling.pb.cc} to
# exist. But we don't want to add it as a dependency, as it would trigger a
# rebuild of the rtl target every time the proto files change. Instead we create
# a conditional dependency on the proto files, which will only create the proto
# file if it doesn't exist.
PROTOS_DEPENDENCY =
ifeq (,$(wildcard test/compiler/proto/param.pb.cc))
PROTOS_DEPENDENCY += test/compiler/proto/param.pb.cc
endif
ifeq (,$(wildcard test/compiler/proto/tiling.pb.cc))
PROTOS_DEPENDENCY += test/compiler/proto/tiling.pb.cc
endif

RTL_DEPENDENCIES =
ifeq ($(SUPPORT_MVM), true)
RTL_DEPENDENCIES += $(CATAPULT_BUILD_DIR)/MatrixVectorUnit/MatrixVectorUnit.v1/concat_rtl.v
endif
ifeq ($(SUPPORT_SPMM), true)
RTL_DEPENDENCIES += $(CATAPULT_BUILD_DIR)/SpMMUnit/SpMMUnit.v1/concat_rtl.v
endif
ifeq ($(SUPPORT_DWC), true)
RTL_DEPENDENCIES += $(CATAPULT_BUILD_DIR)/DwCUnit/DwCUnit.v1/concat_rtl.v
endif

VU_RTL_DEPENDENCIES =
ifeq ($(SUPPORT_SPMM), true)
VU_RTL_DEPENDENCIES += $(CATAPULT_BUILD_DIR)/OutlierFilter/OutlierFilter.v1/concat_rtl.v
endif

# For debugging it might be beneficial to only build sub-components in RTL and
# have them integrate into the SystemC code
InputController: $(CATAPULT_BUILD_DIR)/InputController/InputController.v1/concat_rtl.v
WeightController: $(CATAPULT_BUILD_DIR)/WeightController/WeightController.v1/concat_rtl.v
SystolicArray: $(CATAPULT_BUILD_DIR)/SystolicArray/SystolicArray.v1/concat_rtl.v
MatrixProcessor: $(CATAPULT_BUILD_DIR)/MatrixProcessor/MatrixProcessor.v1/concat_rtl.v
ProcessingElement: $(CATAPULT_BUILD_DIR)/ProcessingElement/ProcessingElement.v1/concat_rtl.v
VectorFetchUnit: $(CATAPULT_BUILD_DIR)/VectorFetchUnit/VectorFetchUnit.v1/concat_rtl.v
VectorUnit: $(CATAPULT_BUILD_DIR)/VectorUnit/VectorUnit.v1/concat_rtl.v
OutputController: $(CATAPULT_BUILD_DIR)/OutputController/OutputController.v1/concat_rtl.v
VectorPipeline: $(CATAPULT_BUILD_DIR)/VectorPipeline/VectorPipeline.v1/concat_rtl.v
MatrixVectorUnit: $(CATAPULT_BUILD_DIR)/MatrixVectorUnit/MatrixVectorUnit.v1/concat_rtl.v
SpMMUnit: $(CATAPULT_BUILD_DIR)/SpMMUnit/SpMMUnit.v1/concat_rtl.v
MulAddTree: $(CATAPULT_BUILD_DIR)/MulAddTree/MulAddTree.v1/concat_rtl.v
DwCUnit: $(CATAPULT_BUILD_DIR)/DwCUnit/DwCUnit.v1/concat_rtl.v
Accelerator: $(CATAPULT_BUILD_DIR)/Accelerator/Accelerator.v1/concat_rtl.v

$(CATAPULT_BUILD_DIR)/InputController/InputController.v1/concat_rtl.v: src/InputController.h $(PROTOS_DEPENDENCY)
	mkdir -p $(CATAPULT_BUILD_DIR)
	BLOCK=InputController catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/InputController.log

$(CATAPULT_BUILD_DIR)/WeightController/WeightController.v1/concat_rtl.v: src/WeightController.h $(PROTOS_DEPENDENCY)
	mkdir -p $(CATAPULT_BUILD_DIR)
	BLOCK=WeightController catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/WeightController.log

$(CATAPULT_BUILD_DIR)/ProcessingElement/ProcessingElement.v1/concat_rtl.v: src/ProcessingElement.h $(PROTOS_DEPENDENCY)
	mkdir -p $(CATAPULT_BUILD_DIR)
	BLOCK=ProcessingElement catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/ProcessingElement.log

$(CATAPULT_BUILD_DIR)/SystolicArrayRow/SystolicArrayRow.v1/concat_rtl.v: src/SystolicArray.h $(CATAPULT_BUILD_DIR)/ProcessingElement/ProcessingElement.v1/concat_rtl.v $(PROTOS_DEPENDENCY)
	mkdir -p $(CATAPULT_BUILD_DIR)
	BLOCK=SystolicArrayRow catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/SystolicArrayRow.log

$(CATAPULT_BUILD_DIR)/SystolicArray/SystolicArray.v1/concat_rtl.v: src/SystolicArray.h $(CATAPULT_BUILD_DIR)/SystolicArrayRow/SystolicArrayRow.v1/concat_rtl.v $(PROTOS_DEPENDENCY)
	mkdir -p $(CATAPULT_BUILD_DIR)
	BLOCK=SystolicArray catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/SystolicArray.log

$(CATAPULT_BUILD_DIR)/MatrixProcessor/MatrixProcessor.v1/concat_rtl.v: src/MatrixProcessor.h src/SystolicArray.h src/Skewer.h $(CATAPULT_BUILD_DIR)/SystolicArray/SystolicArray.v1/concat_rtl.v $(PROTOS_DEPENDENCY)
	mkdir -p $(CATAPULT_BUILD_DIR)
	BLOCK=MatrixProcessor catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/MatrixProcessor.log

$(CATAPULT_BUILD_DIR)/MatrixParamsDeserializer/MatrixParamsDeserializer.v1/concat_rtl.v: src/ParamsDeserializer.h $(PROTOS_DEPENDENCY)
	mkdir -p $(CATAPULT_BUILD_DIR)
	BLOCK=MatrixParamsDeserializer catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/MatrixParamsDeserializer.log

$(CATAPULT_BUILD_DIR)/VectorFetchUnit/VectorFetchUnit.v1/concat_rtl.v: src/vector_unit/VectorFetch.h $(PROTOS_DEPENDENCY)
	mkdir -p $(CATAPULT_BUILD_DIR)
	BLOCK=VectorFetchUnit catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/VectorFetchUnit.log

$(CATAPULT_BUILD_DIR)/VectorParamsDeserializer/VectorParamsDeserializer.v1/concat_rtl.v: src/ParamsDeserializer.h $(PROTOS_DEPENDENCY)
	mkdir -p $(CATAPULT_BUILD_DIR)
	BLOCK=VectorParamsDeserializer catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/VectorParamsDeserializer.log

$(CATAPULT_BUILD_DIR)/OutlierFilter/OutlierFilter.v1/concat_rtl.v: src/vector_unit/OutlierFilter.h $(PROTOS_DEPENDENCY)
	mkdir -p $(CATAPULT_BUILD_DIR)
	BLOCK=OutlierFilter catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/OutlierFilter.log

$(CATAPULT_BUILD_DIR)/VectorPipeline/VectorPipeline.v1/concat_rtl.v: $(VU_RTL_DEPENDENCIES) src/vector_unit/VectorPipeline.h $(PROTOS_DEPENDENCY)
	mkdir -p $(CATAPULT_BUILD_DIR)
	BLOCK=VectorPipeline catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/VectorPipeline.log

$(CATAPULT_BUILD_DIR)/VectorReducer/VectorReducer.v1/concat_rtl.v: src/vector_unit/Reducer.h $(PROTOS_DEPENDENCY)
	mkdir -p $(CATAPULT_BUILD_DIR)
	BLOCK=VectorReducer catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/VectorReducer.log

$(CATAPULT_BUILD_DIR)/VectorAccumulator/VectorAccumulator.v1/concat_rtl.v: src/vector_unit/Accumulator.h $(PROTOS_DEPENDENCY)
	mkdir -p $(CATAPULT_BUILD_DIR)
	BLOCK=VectorAccumulator catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/VectorAccumulator.log

$(CATAPULT_BUILD_DIR)/OutputController/OutputController.v1/concat_rtl.v: src/vector_unit/OutputController.h $(PROTOS_DEPENDENCY)
	mkdir -p $(CATAPULT_BUILD_DIR)
	BLOCK=OutputController catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/OutputController.log

$(CATAPULT_BUILD_DIR)/VectorUnit/VectorUnit.v1/concat_rtl.v: \
    $(CATAPULT_BUILD_DIR)/VectorFetchUnit/VectorFetchUnit.v1/concat_rtl.v \
    $(CATAPULT_BUILD_DIR)/VectorPipeline/VectorPipeline.v1/concat_rtl.v \
	$(CATAPULT_BUILD_DIR)/VectorReducer/VectorReducer.v1/concat_rtl.v \
	$(CATAPULT_BUILD_DIR)/VectorAccumulator/VectorAccumulator.v1/concat_rtl.v \
    $(CATAPULT_BUILD_DIR)/OutputController/OutputController.v1/concat_rtl.v \
    $(CATAPULT_BUILD_DIR)/VectorParamsDeserializer/VectorParamsDeserializer.v1/concat_rtl.v \
	src/vector_unit/main.h $(PROTOS_DEPENDENCY)
	mkdir -p $(CATAPULT_BUILD_DIR)
	BLOCK=VectorUnit catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/VectorUnit.log

$(CATAPULT_BUILD_DIR)/VectorMacUnit/VectorMacUnit.v1/concat_rtl.v: src/matrix_vector_unit/VectorMacUnit.h $(PROTOS_DEPENDENCY)
	mkdir -p $(CATAPULT_BUILD_DIR)
	BLOCK=VectorMacUnit catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/VectorMacUnit.log

$(CATAPULT_BUILD_DIR)/MatrixVectorUnit/MatrixVectorUnit.v1/concat_rtl.v: \
	$(CATAPULT_BUILD_DIR)/VectorMacUnit/VectorMacUnit.v1/concat_rtl.v \
	src/matrix_vector_unit/main.h $(PROTOS_DEPENDENCY)
	mkdir -p $(CATAPULT_BUILD_DIR)
	BLOCK=MatrixVectorUnit catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/MatrixVectorUnit.log

$(CATAPULT_BUILD_DIR)/SpMMUnit/SpMMUnit.v1/concat_rtl.v: src/SpMMUnit.h $(PROTOS_DEPENDENCY)
	mkdir -p $(CATAPULT_BUILD_DIR)
	BLOCK=SpMMUnit catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/SpMMUnit.log

$(CATAPULT_BUILD_DIR)/MulAddTree/MulAddTree.v1/concat_rtl.v: src/MulAddTree.h $(PROTOS_DEPENDENCY)
	mkdir -p $(CATAPULT_BUILD_DIR)
	BLOCK=MulAddTree catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/MulAddTree.log
$(CATAPULT_BUILD_DIR)/DwCUnit/DwCUnit.v1/concat_rtl.v: src/DwCUnit.h $(CATAPULT_BUILD_DIR)/MulAddTree/MulAddTree.v1/concat_rtl.v $(PROTOS_DEPENDENCY)
	mkdir -p $(CATAPULT_BUILD_DIR)
	BLOCK=DwCUnit catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/DwCUnit.log

$(CATAPULT_BUILD_DIR)/Accelerator/Accelerator.v1/concat_rtl.v: \
	src/Accelerator.h \
	src/DoubleBuffer.h \
	$(CATAPULT_BUILD_DIR)/InputController/InputController.v1/concat_rtl.v \
	$(CATAPULT_BUILD_DIR)/WeightController/WeightController.v1/concat_rtl.v \
	$(CATAPULT_BUILD_DIR)/MatrixProcessor/MatrixProcessor.v1/concat_rtl.v \
	$(CATAPULT_BUILD_DIR)/VectorUnit/VectorUnit.v1/concat_rtl.v \
	$(CATAPULT_BUILD_DIR)/MatrixParamsDeserializer/MatrixParamsDeserializer.v1/concat_rtl.v \
	$(RTL_DEPENDENCIES) \
	$(PROTOS_DEPENDENCY)
	mkdir -p $(CATAPULT_BUILD_DIR)
	BLOCK=Accelerator catapult -shell -file scripts/main.tcl -logfile $(CATAPULT_BUILD_DIR)/Accelerator.log

.PHONY: rtl Accelerator InputController WeightController MatrixProcessor ProcessingElement VectorUnit VectorParamsDeserializer VectorFetchUnit VectorPipeline VectorReducer VectorAccumulator OutputController MatrixVectorUnit MulAddTree DwCUnit

# Run RTL simulation
.PHONY: rtl-sim
rtl-sim: rtl network-proto
	cd $(CATAPULT_BUILD_DIR)/Accelerator/Accelerator.v1 && LD_PRELOAD=$(CONDA_PREFIX)/lib/libstdc++.so.6 make -f ./scverify/Verify_concat_sim_rtl_v_vcs.mk SIMTOOL=vcs sim

.PHONY: rtl-sim-debug
rtl-sim-debug: rtl network-proto
	cd $(CATAPULT_BUILD_DIR)/Accelerator/Accelerator.v1 && LD_PRELOAD=$(CONDA_PREFIX)/lib/libstdc++.so.6 SIM_DUMP_FSDB=1 make -f ./scverify/Verify_concat_sim_rtl_v_vcs.mk SIMTOOL=vcs sim

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

$(CC_BUILD_DIR)/TestRunner: $(CC_BUILD_DIR)/Harness.o $(CC_BUILD_DIR)/TestRunner.o $(CC_BUILD_DIR)/GoldModel.o $(CC_BUILD_DIR)/Simulation.o $(CC_BUILD_DIR)/ArrayMemory.o $(CC_BUILD_DIR)/DataLoader.o $(CC_BUILD_DIR)/Network.o $(CC_BUILD_DIR)/param.pb.o $(CC_BUILD_DIR)/tiling.pb.o $(CC_BUILD_DIR)/MapOperation.o $(CC_BUILD_DIR)/AccessCounter.o $(CC_BUILD_DIR)/Tiling.o  $(SPDLOG_OBJ_FILES)
	$(CC) -o $@ $^ $(LDLIBS) $(LDFLAGS)

$(CC_BUILD_DIR)/TestRunner-fast: $(CC_BUILD_DIR)/Harness-fast.o $(CC_BUILD_DIR)/TestRunner.o $(CC_BUILD_DIR)/GoldModel.o $(CC_BUILD_DIR)/Simulation.o $(CC_BUILD_DIR)/ArrayMemory.o $(CC_BUILD_DIR)/DataLoader.o $(CC_BUILD_DIR)/Network.o $(CC_BUILD_DIR)/param.pb.o $(CC_BUILD_DIR)/tiling.pb.o $(CC_BUILD_DIR)/MapOperation.o $(CC_BUILD_DIR)/AccessCounter.o $(CC_BUILD_DIR)/Tiling.o $(SPDLOG_OBJ_FILES)
	$(CC) -o $@ $^ $(LDLIBS) $(LDFLAGS)

$(CC_BUILD_DIR)/TestRunner-checker: $(CC_BUILD_DIR)/Harness-checker.o $(CC_BUILD_DIR)/TestRunner.o $(CC_BUILD_DIR)/GoldModel-checker.o $(CC_BUILD_DIR)/Simulation.o $(CC_BUILD_DIR)/ArrayMemory.o $(CC_BUILD_DIR)/DataLoader.o $(CC_BUILD_DIR)/Network.o $(CC_BUILD_DIR)/param.pb.o  $(CC_BUILD_DIR)/tiling.pb.o $(CC_BUILD_DIR)/MapOperation.o $(CC_BUILD_DIR)/PEChecker.o $(CC_BUILD_DIR)/AccessCounter.o $(CC_BUILD_DIR)/Tiling.o $(SPDLOG_OBJ_FILES)
	$(CC) -o $@ $^ $(LDLIBS) $(LDFLAGS)

$(CC_BUILD_DIR)/AccuracyTester: $(CC_BUILD_DIR)/AccuracyTester.o $(CC_BUILD_DIR)/GoldModel.o $(CC_BUILD_DIR)/ArrayMemory.o $(CC_BUILD_DIR)/DataLoader.o $(CC_BUILD_DIR)/Network.o $(CC_BUILD_DIR)/param.pb.o $(CC_BUILD_DIR)/tiling.pb.o $(CC_BUILD_DIR)/Tiling.o $(SPDLOG_OBJ_FILES)
	$(CC) -o $@ $^ $(LDLIBS_NO_SYSC) $(LDFLAGS_NO_SYSC) -pthread

$(CC_BUILD_DIR)/Harness.o: test/common/Harness.cc test/common/Harness.h test/common/Utils.h test/toolchain/MapOperation.h $(wildcard src/*.h) $(wildcard src/datatypes/*.h) $(wildcard src/vector_unit/*.h) $(wildcard src/matrix_vector_unit/*.h)
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/Harness-fast.o: test/common/Harness.cc test/common/Harness.h test/common/Utils.h test/toolchain/MapOperation.h $(wildcard src/*.h) $(wildcard src/datatypes/*.h) $(wildcard src/vector_unit/*.h) $(wildcard src/matrix_vector_unit/*.h)
	$(CC) $(C17FLAGS) -DCONNECTIONS_FAST_SIM -c -o $@ $<

$(CC_BUILD_DIR)/Harness-checker.o: test/common/Harness.cc test/common/Harness.h test/common/Utils.h test/toolchain/MapOperation.h $(wildcard src/*.h) $(wildcard src/datatypes/*.h) $(wildcard src/vector_unit/*.h) $(wildcard src/matrix_vector_unit/*.h) test/checker/PEChecker.h
	$(CC) $(C17FLAGS) -DCONNECTIONS_FAST_SIM -DCHECK_PE -c -o $@ $<

$(CC_BUILD_DIR)/GoldModel.o: test/common/GoldModel.cc test/common/GoldModel.h test/common/Utils.h src/ArchitectureParams.h src/vector_unit/ApproximationUnit.h test/toolchain/ApproximationConstants.h $(wildcard src/datatypes/*.h) $(wildcard test/common/operations/*.h)
	$(CC) $(C17FLAGS) -g -c -o $@ $<

$(CC_BUILD_DIR)/GoldModel-checker.o: test/common/GoldModel.cc test/common/GoldModel.h test/common/Utils.h src/ArchitectureParams.h $(wildcard src/datatypes/*.h) $(wildcard test/common/operations/*.h) test/checker/PEChecker.h
	$(CC) $(C17FLAGS) -DCHECK_PE -g -c -o $@ $<

$(CC_BUILD_DIR)/Simulation.o: test/common/Simulation.cc test/common/Simulation.h src/ArchitectureParams.h $(wildcard src/datatypes/*.h) test/common/Utils.h test/common/MemoryInterface.h
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/ArrayMemory.o: test/common/ArrayMemory.cc test/common/ArrayMemory.h test/common/MemoryInterface.h
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/DataLoader.o: test/common/DataLoader.cc test/common/DataLoader.h test/common/MemoryInterface.h
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/MapOperation.o: test/toolchain/MapOperation.cc $(wildcard test/toolchain/*.h)
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/TestRunner.o: test/common/TestRunner.cc
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/PEChecker.o: test/checker/PEChecker.cc test/checker/PEChecker.h $(wildcard src/datatypes/*.h)
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/AccuracyTester.o: test/common/AccuracyTester.cc $(wildcard src/datatypes/*.h) $(wildcard test/toolchain/*.h)
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/AccessCounter.o: test/common/AccessCounter.cc test/common/AccessCounter.h
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/Tiling.o: test/common/Tiling.cc test/common/Tiling.h
	$(CC) $(C17FLAGS) -c -o $@ $<

###########################################################
# Toolchain
###########################################################
toolchain: $(CC_BUILD_DIR)/MapOperation.o $(CC_BUILD_DIR)/Tiling.o $(CC_BUILD_DIR)/Network.o $(CC_BUILD_DIR)/param.pb.o $(CC_BUILD_DIR)/tiling.pb.o

###########################################################
# Networks
###########################################################
$(CC_BUILD_DIR)/Network.o: test/common/Network.cc test/compiler/proto/param.pb.cc
	$(CC) $(C17FLAGS) -c -o $@ $<

test/compiler/proto/param.pb.cc: voyager-compiler/src/voyager_compiler/codegen/param.proto
	protoc -I=voyager-compiler/src/voyager_compiler/codegen --cpp_out=test/compiler/proto $<

test/compiler/proto/tiling_pb2.py: test/compiler/proto/tiling.proto
	protoc --proto_path=test/compiler/proto/ --python_out=test/compiler/proto $<

test/compiler/proto/tiling.pb.cc: test/compiler/proto/tiling.proto
	protoc -I=test/compiler/proto --cpp_out=test/compiler/proto $<

$(CC_BUILD_DIR)/param.pb.o: test/compiler/proto/param.pb.cc
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/tiling.pb.o: test/compiler/proto/tiling.pb.cc
	$(CC) $(C17FLAGS) -c -o $@ $<

.PHONY: network-proto
network-proto: \
    $(CODEGEN_DIR)/networks/$(NETWORK)/$(DATATYPE)/model.txt \
    test/compiler/proto/param.pb.cc \
    test/compiler/proto/tiling_pb2.py \
    test/compiler/proto/tiling.pb.cc \
    $(CODEGEN_DIR)/networks/$(NETWORK)/$(DATATYPE)/$(IC_DIMENSION)x$(OC_DIMENSION)_$(INPUT_BUFFER_SIZE)x$(WEIGHT_BUFFER_SIZE)x$(ACCUM_BUFFER_SIZE)_$(DOUBLE_BUFFERED_ACCUM_BUFFER)/tilings.txtpb

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
