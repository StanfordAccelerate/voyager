set confirm off
set pagination off
set arch riscv:rv64
set remotetimeout 10000000

define print_runtimes
    printf "Matrix Unit Runtime     : %llu cycles\n", *(unsigned long long*)0x20000108
    printf "Vector Unit Runtime     : %llu cycles\n", *(unsigned long long*)0x20000208
    printf "MVM Unit Runtime        : %llu cycles\n", *(unsigned long long*)0x20000308
    printf "SPMM Unit Runtime       : %llu cycles\n", *(unsigned long long*)0x20000408
    printf "Accelerator Runtime     : %llu cycles\n", *(unsigned long long*)0x20000008
end

python
import os
import gdb

def getenv_int(name, default):
    value = os.environ.get(name, default)
    try:
        return int(value, 0)
    except Exception as e:
        raise gdb.GdbError(f"Invalid integer for {name}={value!r}: {e}")

# Configuration
stack_offset    = getenv_int("SOC_MEM_OFFSET", "0")
cache_size      = getenv_int("CACHE_SIZE", str(8 * 1024 * 1024))
max_tiles       = min(getenv_int("MAX_TILES", "2"), 2)
dump_scratchpad = getenv_int("JTAG_SIM", "0")
skip_load       = getenv_int("SKIP_GDB_LOAD", "0")
gdb_port        = os.environ.get("OPENOCD_GDB_PORT", "3333")
dump_path       = os.environ.get("SCRATCHPAD_DUMP_PATH", "./scratchpad_data.bin")
# Optional: a file of "0x<phys> <bytes>" lines (the emitter's
# <layer>_scratchpad_dump.txt). Only those regions are read back, one file
# per line as <dump_path>.<i>, instead of the whole scratchpad -- reading
# 2 MB over a bit-banged JTAG takes hours.
dump_list       = os.environ.get("SCRATCHPAD_DUMP_LIST", "")

scratchpad_base = 0x40000000 + stack_offset
scratchpad_size = max_tiles * cache_size
scratchpad_end  = scratchpad_base + scratchpad_size

# Print configuration
print(f"{'-'*40}\n[DEBUG] RISC-V SOC CONFIG\n{'-'*40}")
print(f"SCRATCHPAD: {scratchpad_base:#x} - {scratchpad_end:#x} ({scratchpad_size} bytes)")
print(f"GDB PORT:   {gdb_port}")

# Initialization
try:
    gdb.execute(f"target remote localhost:{gdb_port}")
    if not skip_load:
        gdb.execute("load")
    gdb.execute("set $pc = 0x40000000")
except gdb.error as e:
    print(f"Target initialization failed: {e}")
    gdb.execute("quit")
end

break _exit
continue
print_runtimes

python
if dump_scratchpad and dump_list:
    with open(dump_list) as f:
        regions = [line.split() for line in f if line.strip()]
    for i, (phys, length) in enumerate(regions):
        start = int(phys, 0)
        end = start + int(length, 0)
        gdb.execute(f"dump binary memory {dump_path}.{i} {start:#x} {end:#x}")
        print(f"[INFO] Scratchpad region {i}: {start:#x} - {end:#x} -> {dump_path}.{i}")
    print(f"[INFO] Scratchpad dump complete: {len(regions)} region(s)")
elif dump_scratchpad:
    gdb.execute(f"dump binary memory {dump_path} {scratchpad_base} {scratchpad_end}")
    print(f"[INFO] Scratchpad dump complete: {dump_path}")
end

quit
