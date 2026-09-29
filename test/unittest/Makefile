# Standalone CIM simulation. Source the checkout's .envrc before invoking Make.
.DEFAULT_GOAL := test-systemc
THIS_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(THIS_DIR)/../..)
PYTHON ?= python3
VERILATOR ?= verilator
SYSTEMC_CC ?= $(CATAPULT_ROOT)/bin/g++
VERILATOR_ROOT ?= $(shell $(VERILATOR) -V 2>/dev/null | tr -d '\000' | sed -n 's/.*VERILATOR_ROOT *= *//p' | head -1)

SYSTEMC_BUILD := $(THIS_DIR)/build/systemc
SYSTEMC_GENERATED := $(SYSTEMC_BUILD)/generated
CIM_VERILATED_DIR := $(SYSTEMC_BUILD)/verilated
SYSTEMC_CASES := $(THIS_DIR)/systemc/cim_element_cosim_cases.py
SYSTEMC_CASES_MK := $(SYSTEMC_GENERATED)/cim_element_cosim_cases.mk
SYSTEMC_CASES_INC := $(SYSTEMC_GENERATED)/CIMElementTbCases.inc
CIM_RTL_SRCS := $(addprefix $(REPO_ROOT)/src/cim/,cim_typedefs.svh \
    cim_macro_wrapper.sv cim_macro_model.sv cim_macro_1.sv cim_element.sv)

-include $(SYSTEMC_CASES_MK)
-include $(SYSTEMC_BUILD)/CIMElementTb.d $(SYSTEMC_BUILD)/CIMArrayTb.d

# Common headers require a datatype and dimensions; the fixtures specify their
# own CIM geometry explicitly and do not depend on these array dimensions.
SYSTEMC_CXXFLAGS := -std=c++17 -O2 -MMD -MP \
    -DSC_INCLUDE_DYNAMIC_PROCESSES -DINT8 -DIC_DIMENSION=4 -DOC_DIMENSION=2 \
    -I$(CATAPULT_ROOT)/shared/include -I$(REPO_ROOT)/lib -I$(REPO_ROOT)/src \
    -I$(VERILATOR_ROOT)/include -I$(SYSTEMC_GENERATED) \
    $(CIM_ELEMENT_VERILATED_INCLUDE_FLAGS) \
    -Wno-unknown-pragmas -Wno-deprecated-declarations
SYSTEMC_LDFLAGS := -L$(CATAPULT_ROOT)/shared/lib \
    -Wl,-rpath,$(CATAPULT_ROOT)/shared/lib -lsystemc -pthread

.PHONY: test-systemc test-systemc-element test-systemc-array test-verilog \
    check-systemc-env clean
test-systemc: test-systemc-element test-systemc-array

test-systemc-element: $(SYSTEMC_BUILD)/CIMElementTb
	$<

test-systemc-array: $(SYSTEMC_BUILD)/CIMArrayTb
	$<

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

$(SYSTEMC_BUILD)/CIMArrayTb: $(SYSTEMC_BUILD)/CIMArrayTb.o
	$(SYSTEMC_CC) -o $@ $^ $(SYSTEMC_LDFLAGS)

check-systemc-env:
	@test -n "$(CATAPULT_ROOT)" || { echo "CATAPULT_ROOT is unset; source .envrc with the CAD mount available."; exit 2; }
	@test -f "$(CATAPULT_ROOT)/shared/include/systemc.h" || { echo "SystemC headers not found under CATAPULT_ROOT."; exit 2; }
	@command -v $(SYSTEMC_CC) >/dev/null || { echo "SystemC compiler not found: $(SYSTEMC_CC)"; exit 2; }
	@command -v $(VERILATOR) >/dev/null || { echo "Verilator not found; update the Conda environment."; exit 2; }

clean:
	rm -rf $(SYSTEMC_BUILD)
