#include "traps.h"

#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include "mmio.h"

void enable_interrupts() {
  // clear any stale sticky bits before unmasking
  reg_write16(VOYAGER_INT_STATUS, 0xFFFF);
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
    // Clear Voyager's internal bit
    uint16_t status = reg_read16(VOYAGER_INT_STATUS);
    reg_write16(VOYAGER_INT_STATUS, status);  // W1C
  }

  reg_write32(PLIC_CLAIM, id);
}
