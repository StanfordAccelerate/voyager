#include "traps.h"

#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include "mmio.h"

void enable_interrupts() {
  // Clear any stale sticky bits before unmasking -- except the semaphore
  // change flags 15:8, which run_voyager_operation.c owns: 11:8 report the
  // units' start credits being consumed, 15 flips with the testbench
  // doorbell (host_request.c) and is never read.
  reg_write16(VOYAGER_INT_STATUS, 0x00FF);
  // Enable all three interrupts in the Voyager hardware
  reg_write16(VOYAGER_INT_ENABLE, INT_BANK0 | INT_BANK1 | INT_DONE);

  // 1. Configure PLIC for Voyager
  // Set priority > 0 to enable the line
  reg_write32(PLIC_PRIORITY(VOYAGER_INT_ID), 1);
  // Enable the interrupt ID for Hart 0
  reg_write32(PLIC_ENABLE(VOYAGER_INT_ID), (1 << VOYAGER_INT_ID));
  // Set threshold to 0 to allow all interrupts with priority > 0
  reg_write32(PLIC_THRESHOLD, 0);

  // Enable RISC-V Core Interrupts
  __asm__ volatile("csrs mie, %0" ::"r"(MIE_MEIE));
  __asm__ volatile("csrs mstatus, %0" ::"r"(MSTATUS_MIE));
}

void handle_trap(void) {
  uint32_t id = reg_read32(PLIC_CLAIM);

  if (id == VOYAGER_INT_ID) {
    // Clear Voyager's internal bit. Never 15:8, the semaphore change flags:
    // a blanket W1C here would swallow a unit's start observation and hang
    // the credit that waits for it.
    uint16_t status = reg_read16(VOYAGER_INT_STATUS);
    reg_write16(VOYAGER_INT_STATUS, status & 0x00FF);  // W1C
  }

  reg_write32(PLIC_CLAIM, id);
}
