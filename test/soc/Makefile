# ============================================================
# Project Build Configuration
# ============================================================

ACCEL_DIR = ../..

# Build parameters (filled in from environment or defaults)
BUILD_DIR ?= $(ACCEL_DIR)/build/$(DATATYPE)_$(IC_DIMENSION)x$(OC_DIMENSION)_$(INPUT_BUFFER_SIZE)x$(WEIGHT_BUFFER_SIZE)x$(ACCUM_BUFFER_SIZE)_$(DOUBLE_BUFFERED_ACCUM_BUFFER)_$(SUPPORT_MVM)_$(SUPPORT_SPMM)
CC_BUILD_DIR = $(BUILD_DIR)/cc

export CODEGEN_DIR ?= test/compiler
export PROJECT_ROOT ?= $(ACCEL_DIR)

# ============================================================
# Toolchain
# ============================================================

CC := /cad/mentor/2024.2_2/Mgc_home/bin/g++

# ============================================================
# Includes
# ============================================================

INC := \
	-I/cad/mentor/2024.2_2/Mgc_home/shared/include/ \
	-I$(ACCEL_DIR)/lib/ \
	-I$(ACCEL_DIR)/lib/xtensor/include \
	-I$(ACCEL_DIR)/lib/xtl/include \
	-I$(ACCEL_DIR)/lib/spdlog/include \
	-I$(CONDA_PREFIX)/include \
	-I$(ACCEL_DIR)/src/ \
	-I$(ACCEL_DIR)

# ============================================================
# Base Compiler & Linker Flags
# ============================================================

EXTRA_SIM_CXXFLAGS ?=

override BASE_FLAGS += \
	$(INC) \
	-DSC_INCLUDE_DYNAMIC_PROCESSES \
	-DSPDLOG_COMPILED_LIB \
	-DSPDLOG_EOL=\"\" \
	-Wno-unknown-pragmas \
	-Wno-unused-but-set-variable \
	-Wno-unused-variable \
	-Wno-sign-compare \
	-Wno-bool-operation \
	-Wno-maybe-uninitialized \
	-Wno-class-memaccess \
	-Wall \
	-Wno-bool-compare \
	$(EXTRA_SIM_CXXFLAGS)

C17FLAGS += $(BASE_FLAGS) -std=c++17 -Wno-deprecated-declarations
LDFLAGS  += -lsystemc -lstdc++fs -labsl_hash -labsl_log_internal_check_op \
            -labsl_log_internal_message -labsl_log_internal_nullguard \
            -lrt -lpthread -lprotobuf -Wl,-rpath=$(CONDA_PREFIX)/lib
LDLIBS   += -L/cad/mentor/2024.2_2/Mgc_home/shared/lib/ -L$(CONDA_PREFIX)/lib

# ============================================================
# Debug / Release Mode Switch
# ============================================================

# Enable debug build with: make DEBUG=1
DEBUG ?= 0

ifeq ($(DEBUG),1)
    C17FLAGS += -g -O0
    LDFLAGS  += -g
else
    C17FLAGS += -O2
endif

# ============================================================
# Targets
# ============================================================

.DEFAULT_GOAL := GenerateSoCBinaries

network-proto:
	$(MAKE) -C $(ACCEL_DIR) network-proto

toolchain:
	$(MAKE) -C $(ACCEL_DIR) toolchain

$(CC_BUILD_DIR)/spdlog.o:
	$(MAKE) -C $(ACCEL_DIR) spdlog

$(CC_BUILD_DIR)/GenerateSoCBinaries.o: $(ACCEL_DIR)/test/soc/GenerateSoCBinaries.cc network-proto
	@mkdir -p $(CC_BUILD_DIR)
	$(CC) $(C17FLAGS) -c -o $@ $<

# New-IR toolchain objects. voyager_ir.pb.o / MapOperation.o / Tiling.o come
# from `make -C $(ACCEL_DIR) toolchain`; test/common and test/soc objects
# build here via the pattern rules below.
COMMON_OBJS := \
	$(CC_BUILD_DIR)/ArrayMemory.o \
	$(CC_BUILD_DIR)/Model.o \
	$(CC_BUILD_DIR)/GraphUtils.o \
	$(CC_BUILD_DIR)/Interpreter.o \
	$(CC_BUILD_DIR)/GoldModel.o \
	$(CC_BUILD_DIR)/Checker.o \
	$(CC_BUILD_DIR)/MapOperation.o \
	$(CC_BUILD_DIR)/Tiling.o \
	$(CC_BUILD_DIR)/voyager_ir.pb.o \
	$(CC_BUILD_DIR)/spdlog.o

SOC_OBJS := \
	$(CC_BUILD_DIR)/EmitC.o \
	$(CC_BUILD_DIR)/ScheduleRecorder.o

GENERATE_SOC_BINARIES_OBJS := \
	$(CC_BUILD_DIR)/GenerateSoCBinaries.o \
	$(SOC_OBJS) \
	$(COMMON_OBJS)

$(CC_BUILD_DIR)/GenerateSoCBinaries: toolchain $(GENERATE_SOC_BINARIES_OBJS)
	$(CC) -o $@ $(GENERATE_SOC_BINARIES_OBJS) $(LDLIBS) $(LDFLAGS)

GenerateSoCBinaries: $(CC_BUILD_DIR)/GenerateSoCBinaries
	$<

$(CC_BUILD_DIR)/%.o: $(ACCEL_DIR)/test/common/%.cc
	@mkdir -p $(CC_BUILD_DIR)
	$(CC) $(C17FLAGS) -c -o $@ $<

$(CC_BUILD_DIR)/%.o: $(ACCEL_DIR)/test/soc/%.cc
	@mkdir -p $(CC_BUILD_DIR)
	$(CC) $(C17FLAGS) -c -o $@ $<

# ============================================================
# Cleanup
# ============================================================

# Only the objects this Makefile builds -- $(BUILD_DIR) also holds the
# Catapult HLS results the repo-root Makefile's rtl flow depends on.
clean:
	rm -rf $(CC_BUILD_DIR)

.PHONY: network-proto toolchain clean
