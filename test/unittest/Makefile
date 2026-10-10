# Standalone datapath simulation. Source the checkout's .envrc before Make.
.DEFAULT_GOAL := test-systemc
THIS_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(THIS_DIR)/../..)
PYTHON ?= python3
VERILATOR ?= verilator
SYSTEMC_CC ?= $(CATAPULT_ROOT)/bin/g++
VERILATOR_ROOT ?= $(shell $(VERILATOR) -V 2>/dev/null | tr -d '\000' | sed -n 's/.*VERILATOR_ROOT *= *//p' | head -1)

ENABLE_PERF_COUNTERS ?= 1
SYSTEMC_BUILD := $(THIS_DIR)/build/systemc$(if $(filter-out 1,$(ENABLE_PERF_COUNTERS)),_perf$(ENABLE_PERF_COUNTERS))
SYSTEMC_GENERATED := $(SYSTEMC_BUILD)/generated
CIM_VERILATED_DIR := $(SYSTEMC_BUILD)/verilated
SYSTEMC_CASES := $(THIS_DIR)/systemc/cim_element_cosim_cases.py
SYSTEMC_CASES_MK := $(SYSTEMC_GENERATED)/cim_element_cosim_cases.mk
SYSTEMC_CASES_INC := $(SYSTEMC_GENERATED)/CIMElementTbCases.inc
CIM_RTL_SRCS := $(addprefix $(REPO_ROOT)/src/cim/,cim_typedefs.svh \
    cim_macro_wrapper.sv cim_macro_model.sv cim_macro_1.sv cim_element.sv)

-include $(SYSTEMC_CASES_MK)
-include $(SYSTEMC_BUILD)/CIMElementTb.d $(SYSTEMC_BUILD)/CIMArrayTb.d \
    $(SYSTEMC_BUILD)/CIMPendingWeightTb.d

# Common headers require a datatype and dimensions; the fixtures specify their
# own CIM geometry explicitly and do not depend on these array dimensions.
SYSTEMC_CXXFLAGS := -std=c++17 -O2 -MMD -MP -DENABLE_PERF_COUNTERS=$(ENABLE_PERF_COUNTERS) \
    -DSC_INCLUDE_DYNAMIC_PROCESSES -DINT8 -DIC_DIMENSION=4 -DOC_DIMENSION=2 \
    -I$(CATAPULT_ROOT)/shared/include -I$(REPO_ROOT)/lib -I$(REPO_ROOT)/src \
    -I$(VERILATOR_ROOT)/include -I$(SYSTEMC_GENERATED) \
    $(CIM_ELEMENT_VERILATED_INCLUDE_FLAGS) \
    -Wno-unknown-pragmas -Wno-deprecated-declarations
SYSTEMC_LDFLAGS := -L$(CATAPULT_ROOT)/shared/lib \
    -Wl,-rpath,$(CATAPULT_ROOT)/shared/lib -lsystemc -pthread

SYSTEMC_PROCESSOR_CXXFLAGS := $(filter-out -DIC_DIMENSION=4,$(SYSTEMC_CXXFLAGS)) \
    -DIC_DIMENSION=8
PROCESSOR_CASES := parallel serial banked
PROCESSOR_FLAGS_parallel := -DCIM_PROCESSOR_TEST_MODE=0
PROCESSOR_FLAGS_serial := -DCIM_PROCESSOR_TEST_MODE=1
PROCESSOR_FLAGS_banked := -DCIM_PROCESSOR_TEST_MODE=0 \
    -DDOUBLE_BUFFERED_ACCUM_BUFFER=true
PROCESSOR_TESTS := $(addprefix $(SYSTEMC_BUILD)/CIMProcessorTb-,$(PROCESSOR_CASES))
-include $(addsuffix .d,$(PROCESSOR_TESTS))

MATRIX_CASES := parallel serial banked
MATRIX_FLAGS_parallel := -DCIM_MODE=0 -DDOUBLE_BUFFERED_ACCUM_BUFFER=false
MATRIX_FLAGS_serial := -DCIM_MODE=1 -DDOUBLE_BUFFERED_ACCUM_BUFFER=false
MATRIX_FLAGS_banked := -DCIM_MODE=0 -DDOUBLE_BUFFERED_ACCUM_BUFFER=true \
    -DIC_PORT_WIDTH=32 -DOC_PORT_WIDTH=8
MATRIX_CXXFLAGS := -std=c++17 -O2 -MMD -MP -DSC_INCLUDE_DYNAMIC_PROCESSES \
    -DENABLE_PERF_COUNTERS=$(ENABLE_PERF_COUNTERS) \
    -DINT8 -DIC_DIMENSION=8 -DOC_DIMENSION=2 -DMATRIX_BACKEND=MATRIX_BACKEND_CIM \
    -DCIM_MACRO_INPUT_LANES=2 -DCIM_MACRO_OUTPUT_LANES=2 -DCIM_WEIGHT_SETS=3 \
    -DCIM_BASE_A_WIDTH=4 -DCIM_BASE_B_WIDTH=4 -DCIM_BASE_C_WIDTH=12 \
    -DCIM_MACRO_WRITE_INPUT_LANES=1 -DCIM_MAC_LATENCY=1 -DCIM_SIGNED=true \
    -DCIM_TILE_INPUT_AXIS_ELEMENTS=2 -DCIM_TILE_OUTPUT_AXIS_ELEMENTS=1 \
    -DCIM_INPUT_AXIS_TILES=2 -DCIM_OUTPUT_AXIS_TILES=2 \
    -DCIM_A_PORT_TILES=2 -DCIM_B_PORT_TILES=1 -DCIM_C_PORT_TILES=2 \
    -DCIM_C_BEAT_LAYOUT=1 -DCIM_ARRAY_RESULT_SLOTS=2 -DCIM_LOCAL_ACCUM_CONTEXTS=4 \
    -DINPUT_BUFFER_SIZE=64 -DWEIGHT_BUFFER_SIZE=64 -DACCUM_BUFFER_SIZE=32 \
    -I$(CONDA_PREFIX)/include -I$(CATAPULT_ROOT)/shared/include \
    -I$(SYSTEMC_GENERATED) -I$(REPO_ROOT) -I$(REPO_ROOT)/lib -I$(REPO_ROOT)/src \
    -I$(REPO_ROOT)/lib/spdlog/include \
    -Wno-unknown-pragmas -Wno-deprecated-declarations
MATRIX_LDFLAGS := -L$(CONDA_PREFIX)/lib -Wl,-rpath,$(CONDA_PREFIX)/lib \
    $(SYSTEMC_LDFLAGS) -labsl_hash -labsl_log_internal_check_op \
    -labsl_log_internal_message -labsl_log_internal_nullguard -lprotobuf
MATRIX_TESTS := $(addprefix $(SYSTEMC_BUILD)/CIMMatrixUnitTb-,$(MATRIX_CASES))
SA_MATRIX_CASES := single banked
SA_MATRIX_FLAGS_single := -DDOUBLE_BUFFERED_ACCUM_BUFFER=false
SA_MATRIX_FLAGS_banked := -DDOUBLE_BUFFERED_ACCUM_BUFFER=true -DIC_PORT_WIDTH=32 -DOC_PORT_WIDTH=8
SA_MATRIX_TESTS := $(addprefix $(SYSTEMC_BUILD)/SystolicMatrixUnitTb-,$(SA_MATRIX_CASES))
-include $(addsuffix .d,$(SA_MATRIX_TESTS))
MATRIX_PROTO_DIR := $(SYSTEMC_GENERATED)/test/compiler/proto
MATRIX_PROTO := $(MATRIX_PROTO_DIR)/voyager_ir.pb.h
-include $(addsuffix .d,$(MATRIX_TESTS)) $(SYSTEMC_BUILD)/AccessCounter.d

.PHONY: test-systemc test-systemc-element test-systemc-array test-verilog \
    test-systemc-processor test-systemc-matrix test-systemc-scratchpad check-systemc-env clean
test-systemc: test-systemc-element test-systemc-array test-systemc-processor \
    test-systemc-matrix test-systemc-scratchpad

test-systemc-scratchpad: $(SYSTEMC_BUILD)/ScratchpadTimingTb
	$<

$(SYSTEMC_BUILD)/ScratchpadTimingTb: $(THIS_DIR)/systemc/ScratchpadTimingTb.cc \
    $(REPO_ROOT)/test/common/ScratchpadTiming.h $(MATRIX_PROTO) \
    $(SYSTEMC_BUILD)/voyager_ir.pb.o | check-systemc-env
	mkdir -p $(@D)
	$(SYSTEMC_CC) -std=c++17 -O2 -I$(SYSTEMC_GENERATED) -I$(REPO_ROOT) \
	    -I$(CONDA_PREFIX)/include -I$(CATAPULT_ROOT)/shared/include \
	    -o $@ $< $(SYSTEMC_BUILD)/voyager_ir.pb.o $(MATRIX_LDFLAGS)

$(SYSTEMC_BUILD)/voyager_ir.pb.o: $(MATRIX_PROTO)
	$(SYSTEMC_CC) -std=c++17 -O2 -I$(CONDA_PREFIX)/include \
	    -c -o $@ $(MATRIX_PROTO:.h=.cc)

test-systemc-element: $(SYSTEMC_BUILD)/CIMElementTb
	$<

test-systemc-array: $(SYSTEMC_BUILD)/CIMArrayTb $(SYSTEMC_BUILD)/CIMPendingWeightTb
	@for test in $^; do "$$test" || exit $$?; done

test-systemc-processor: $(PROCESSOR_TESTS)
	@for test in $^; do echo "Running $${test##*/}"; "$$test" || exit $$?; done

test-systemc-matrix: $(MATRIX_TESTS) $(SA_MATRIX_TESTS)
	@status=0; for test in $^; do echo "Running $${test##*/}"; \
	    "$$test" $(CIM_MATRIX_TEST_ARGS) || status=1; done; exit $$status

# The existing Verilog suites remain independently runnable.
test-verilog:
	$(PYTHON) $(THIS_DIR)/verilog/test_cim_macro.py $(TEST_ARGS)
	$(PYTHON) $(THIS_DIR)/verilog/test_cim_element.py $(TEST_ARGS)

$(SYSTEMC_CASES_MK): $(SYSTEMC_CASES) $(THIS_DIR)/Makefile
	mkdir -p $(SYSTEMC_GENERATED)
	$(PYTHON) $(SYSTEMC_CASES) --makefile $@

$(CIM_ELEMENT_VERILATED_LIBS): $(SYSTEMC_CASES) $(THIS_DIR)/Makefile | check-systemc-env

$(SYSTEMC_BUILD)/CIMElementTb.o: $(THIS_DIR)/systemc/CIMElementTb.cc \
    $(SYSTEMC_CASES_INC) $(CIM_ELEMENT_VERILATED_LIBS) $(THIS_DIR)/Makefile | check-systemc-env
	mkdir -p $(@D)
	$(SYSTEMC_CC) $(SYSTEMC_CXXFLAGS) -c -o $@ $<

$(SYSTEMC_BUILD)/CIMArrayTb.o: $(THIS_DIR)/systemc/CIMArrayTb.cc $(THIS_DIR)/Makefile | check-systemc-env
	mkdir -p $(@D)
	$(SYSTEMC_CC) $(SYSTEMC_CXXFLAGS) -c -o $@ $<

$(SYSTEMC_BUILD)/CIMElementTb: $(SYSTEMC_BUILD)/CIMElementTb.o $(CIM_ELEMENT_VERILATED_LIBS)
	$(SYSTEMC_CC) -o $@ $^ $(CIM_ELEMENT_VERILATED_RUNTIME_LIB) $(SYSTEMC_LDFLAGS)

$(SYSTEMC_BUILD)/CIMPendingWeightTb.o: $(THIS_DIR)/systemc/CIMPendingWeightTb.cc $(THIS_DIR)/Makefile | check-systemc-env
	mkdir -p $(@D)
	$(SYSTEMC_CC) $(SYSTEMC_CXXFLAGS) -c -o $@ $<

$(SYSTEMC_BUILD)/CIMPendingWeightTb: $(SYSTEMC_BUILD)/CIMPendingWeightTb.o
	$(SYSTEMC_CC) -o $@ $^ $(SYSTEMC_LDFLAGS)

$(SYSTEMC_BUILD)/CIMArrayTb: $(SYSTEMC_BUILD)/CIMArrayTb.o
	$(SYSTEMC_CC) -o $@ $^ $(SYSTEMC_LDFLAGS)

$(addsuffix .o,$(PROCESSOR_TESTS)): $(SYSTEMC_BUILD)/CIMProcessorTb-%.o: $(THIS_DIR)/systemc/CIMProcessorTb.cc $(THIS_DIR)/Makefile | check-systemc-env
	mkdir -p $(@D)
	$(SYSTEMC_CC) $(SYSTEMC_PROCESSOR_CXXFLAGS) $(PROCESSOR_FLAGS_$*) -c -o $@ $<

$(PROCESSOR_TESTS): $(SYSTEMC_BUILD)/CIMProcessorTb-%: $(SYSTEMC_BUILD)/CIMProcessorTb-%.o
	$(SYSTEMC_CC) -o $@ $^ $(SYSTEMC_LDFLAGS)

$(MATRIX_PROTO): $(REPO_ROOT)/voyager-compiler/src/voyager_compiler/codegen/voyager_ir.proto
	mkdir -p $(@D)
	protoc -I$(dir $<) --cpp_out=$(@D) $<

$(SYSTEMC_BUILD)/AccessCounter.o: $(REPO_ROOT)/test/common/AccessCounter.cc $(MATRIX_PROTO) $(THIS_DIR)/Makefile | check-systemc-env
	mkdir -p $(@D)
	$(SYSTEMC_CC) $(MATRIX_CXXFLAGS) -c -o $@ $<

$(addsuffix .o,$(MATRIX_TESTS)): $(SYSTEMC_BUILD)/CIMMatrixUnitTb-%.o: $(THIS_DIR)/systemc/MatrixUnitTb.cc $(MATRIX_PROTO) $(THIS_DIR)/Makefile | check-systemc-env
	mkdir -p $(@D)
	$(SYSTEMC_CC) $(MATRIX_CXXFLAGS) $(MATRIX_FLAGS_$*) -c -o $@ $<

$(MATRIX_TESTS): $(SYSTEMC_BUILD)/CIMMatrixUnitTb-%: $(SYSTEMC_BUILD)/CIMMatrixUnitTb-%.o $(SYSTEMC_BUILD)/AccessCounter.o
	$(SYSTEMC_CC) -o $@ $^ $(MATRIX_LDFLAGS)

$(addsuffix .o,$(SA_MATRIX_TESTS)): $(SYSTEMC_BUILD)/SystolicMatrixUnitTb-%.o: $(THIS_DIR)/systemc/MatrixUnitTb.cc $(MATRIX_PROTO) $(THIS_DIR)/Makefile | check-systemc-env
	mkdir -p $(@D)
	$(SYSTEMC_CC) $(filter-out -DMATRIX_BACKEND=MATRIX_BACKEND_CIM,$(MATRIX_CXXFLAGS)) $(SA_MATRIX_FLAGS_$*) -c -o $@ $<

$(SA_MATRIX_TESTS): $(SYSTEMC_BUILD)/SystolicMatrixUnitTb-%: $(SYSTEMC_BUILD)/SystolicMatrixUnitTb-%.o $(SYSTEMC_BUILD)/AccessCounter.o
	$(SYSTEMC_CC) -o $@ $^ $(MATRIX_LDFLAGS)

.SECONDARY: $(addsuffix .o,$(PROCESSOR_TESTS) $(MATRIX_TESTS) $(SA_MATRIX_TESTS))

check-systemc-env:
	@test -n "$(CATAPULT_ROOT)" || { echo "CATAPULT_ROOT is unset; source .envrc with the CAD mount available."; exit 2; }
	@test -f "$(CATAPULT_ROOT)/shared/include/systemc.h" || { echo "SystemC headers not found under CATAPULT_ROOT."; exit 2; }
	@command -v $(SYSTEMC_CC) >/dev/null || { echo "SystemC compiler not found: $(SYSTEMC_CC)"; exit 2; }
	@command -v $(VERILATOR) >/dev/null || { echo "Verilator not found; update the Conda environment."; exit 2; }

clean:
	rm -rf $(SYSTEMC_BUILD)
