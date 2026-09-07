#################################
# RISCV Toolchain
#################################

TOOLCHAIN_PREFIX := riscv64-unknown-elf-

AR       := $(TOOLCHAIN_PREFIX)ar
CC       := $(TOOLCHAIN_PREFIX)gcc
CXX      := $(TOOLCHAIN_PREFIX)g++
LD       := $(TOOLCHAIN_PREFIX)ld
OBJCOPY  := $(TOOLCHAIN_PREFIX)objcopy
OBJDUMP  := $(TOOLCHAIN_PREFIX)objdump
SIZE     := $(TOOLCHAIN_PREFIX)size

#################################
# Flags
#################################

# CPU architecture
ARCH     := rv64imafd
ABI      := lp64d
CMODEL   := medany
ARCH_FLAGS := -march=$(ARCH) -mabi=$(ABI) -mcmodel=$(CMODEL)

# spec
SPECS      := htif_nano.specs
SPEC_FLAGS := -specs=$(SPECS)

# linker script. The toolchain's own htif.ld links at the DRAM base; a config
# that runs from the scratchpad passes ../htif.ld, which links at 0x40000000.
LINKER_SCRIPT ?= htif.ld

INCLUDES := -Icommon -I../

EXTRA_CFLAGS   ?=
EXTRA_CXXFLAGS ?=
EXTRA_LDFLAGS  ?=
JTAG_SIM       ?= 0
SEMIHOSTING    ?= 0

CFLAGS   := -std=gnu99 -O2 -Wall -Wextra -fno-common -fno-builtin-printf $(INCLUDES) $(ARCH_FLAGS) $(SPEC_FLAGS) $(EXTRA_CFLAGS)
CXXFLAGS := -O2 -Wall -Wextra -fno-common $(ARCH_FLAGS) $(SPEC_FLAGS) $(EXTRA_CXXFLAGS)
LDFLAGS  := -static $(SPEC_FLAGS) -T $(LINKER_SCRIPT) $(EXTRA_LDFLAGS)

#################################
# Sources
#################################

SRC   := $(shell find -L networks -name "*.c" | grep -v "_scratchpad_data")
OBJ   := $(SRC:.c=.o)
BIN   := $(SRC:.c=.riscv)
DUMP  := $(SRC:.c=.dump)

ifeq ($(JTAG_SIM),1)
SCRATCHPAD_DATA_OBJ := $(SRC:.c=_scratchpad_data.o)
endif

COMMON_OBJS := common/run_voyager_operation.o common/traps.o common/patch_bits.o common/host_request.o

# The JTAG modes toggle -D defines through EXTRA_CFLAGS. Make cannot
# see a flag change, so stamp the flags and rebuild everything when they
# differ -- otherwise a common object built for the previous mode survives.
CFLAGS_STAMP := common/.cflags
$(shell echo '$(CFLAGS)' | cmp -s - $(CFLAGS_STAMP) || echo '$(CFLAGS)' > $(CFLAGS_STAMP))

ifeq ($(SEMIHOSTING),1)
COMMON_OBJS += common/semihost_write.o
endif

#################################
# Top-level targets
#################################

all: $(BIN)

dump: $(DUMP)

clean:
	rm -f $(OBJ) $(BIN) $(DUMP) $(COMMON_OBJS) $(SCRATCHPAD_DATA_OBJ) common/semihost_write.o $(CFLAGS_STAMP)

.PHONY: all dump clean

#################################
# Build rules
#################################

common/run_voyager_operation.o: common/run_voyager_operation.c common/run_voyager_operation.h common/voyager_address.h common/voyager_params.h $(CFLAGS_STAMP)
	$(CC) $(CFLAGS) -c $< -o $@

common/traps.o: common/traps.c common/traps.h $(CFLAGS_STAMP)
	$(CC) $(CFLAGS) -c $< -o $@

common/patch_bits.o: common/patch_bits.c common/patch_bits.h $(CFLAGS_STAMP)
	$(CC) $(CFLAGS) -c $< -o $@

common/host_request.o: common/host_request.c common/host_request.h common/run_voyager_operation.h common/voyager_address.h $(CFLAGS_STAMP)
	$(CC) $(CFLAGS) -c $< -o $@

common/semihost_write.o: common/semihost_write.c $(CFLAGS_STAMP)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ): %.o: %.c common/run_voyager_operation.h common/host_request.h common/voyager_address.h common/voyager_params.h $(CFLAGS_STAMP)
	$(CC) $(CFLAGS) -c $< -o $@

%_scratchpad_data.o: %_scratchpad_data.c
	$(CC) $(CFLAGS) -c $< -o $@

ifeq ($(JTAG_SIM),1)
$(BIN): %.riscv: %.o $(COMMON_OBJS) %_scratchpad_data.o
	$(CC) $(LDFLAGS) -T $*_scratchpad_data.ld $^ -o $@
else
$(BIN): %.riscv: %.o $(COMMON_OBJS)
	$(CC) $(LDFLAGS) $^ -o $@
endif

$(DUMP): %.dump: %.riscv
	$(OBJDUMP) -D $< > $@
