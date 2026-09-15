#ifndef TRAPS_H
#define TRAPS_H

#include <stdio.h>

#include "encoding.h"
#include "voyager_address.h"

// clang-format off
// PLIC base (Chipyard Default)
#define PLIC_BASE          0x0c000000UL
#define PLIC_PRIORITY(id)  (PLIC_BASE + ((id) * 4))
#define PLIC_ENABLE(id)    (PLIC_BASE + 0x2000)
#define PLIC_THRESHOLD     (PLIC_BASE + 0x200000)
#define PLIC_CLAIM         (PLIC_BASE + 0x200004)

// Voyager Interrupt IDs (Check your DTS, usually starts at 1)
// Ids run in device order, so a config with no UART -- which takes 1 -- moves
// Voyager down to it. Override with -DVOYAGER_INT_ID.
#ifndef VOYAGER_INT_ID
#define VOYAGER_INT_ID     2
#endif

// CSR Bits
#define MIE_MEIE           (1 << 11) // Machine External Interrupt Enable
// clang-format on

void enable_interrupts();
void handle_trap(void);

#endif
