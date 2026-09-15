#include "host_request.h"

#ifdef NO_TESTBENCH

/* Where the generated program's argument stores land when nothing services
 * requests. */
host_request_t host_dummy_request;

#else

#include "mmio.h"
#include "run_voyager_operation.h"
#include "voyager_address.h"

host_mailbox_t host_mailbox __attribute__((aligned(64)));

/* Completion cell of the synchronous requests, and how many the firmware
 * has issued: the request is done when the cell catches up. */
static volatile int64_t host_sync __attribute__((aligned(64)));
static int64_t host_sync_issued;

/* Every scratchpad store must have landed before the doorbell's MMIO store
 * is issued; the two go to different TileLink slaves. */
#define HOST_FENCE() __asm__ volatile("fence" ::: "memory")

void host_init(void) {
  volatile host_mailbox_t* mailbox = &host_mailbox;
  mailbox->head = 0;
  mailbox->tail = 0;
  host_sync = 0;
  host_sync_issued = 0;
  HOST_FENCE();
}

volatile host_request_t* host_slot(void) {
  volatile host_mailbox_t* mailbox = &host_mailbox;
  const uint64_t head = mailbox->head;
  /* The testbench takes every request off the ring the moment it is rung,
   * so a full ring only means it has not been woken yet. */
  while (head - mailbox->tail >= HOST_MAILBOX_SLOTS) {
  }
  return &mailbox->slots[head % HOST_MAILBOX_SLOTS];
}

/* The header of a slot the caller has already filled, then the doorbell. */
static void host_ring(volatile host_request_t* slot, uint64_t kind,
                      uint64_t ordinal, uint64_t nargs, uint64_t cell,
                      uint64_t amount) {
  volatile host_mailbox_t* mailbox = &host_mailbox;
  slot->kind = kind;
  slot->ordinal = ordinal;
  slot->nargs = nargs;
  slot->cell = cell;
  slot->amount = amount;
  HOST_FENCE();
  mailbox->head = mailbox->head + 1;
  HOST_FENCE();
  reg_write8(DMA_SIGNAL_ID, HOST_DOORBELL_SEMAPHORE);
  reg_write8(DMA_INC_SEM, 1); /* self-clearing pulse: +1 */
}

static void host_ring_sync(volatile host_request_t* slot, uint64_t kind,
                           uint64_t ordinal, uint64_t nargs) {
  host_sync_issued++;
  host_ring(slot, kind, ordinal, nargs, (uint64_t)(uintptr_t)&host_sync, 1);
  while (host_sync < host_sync_issued) {
  }
}

void host_copy(volatile host_request_t* slot, uint64_t ordinal, uint64_t nargs,
               volatile int64_t* cell, int64_t amount) {
  host_ring(slot, HOST_REQ_COPY, ordinal, nargs, (uint64_t)(uintptr_t)cell,
            (uint64_t)amount);
}

void host_zero(uint64_t ordinal) {
  host_ring_sync(host_slot(), HOST_REQ_ZERO, ordinal, 0);
}

void host_op(volatile host_request_t* slot, uint64_t ordinal, uint64_t nargs) {
  host_ring_sync(slot, HOST_REQ_HOST_OP, ordinal, nargs);
}

void host_post(volatile int64_t* cell, int64_t amount) {
  volatile host_request_t* slot = host_slot();
  for (int u = 0; u < HOST_NUM_UNITS; u++)
    slot->retired[u] = unit_ops_issued[u];
  host_ring(slot, HOST_REQ_POST, 0, 0, (uint64_t)(uintptr_t)cell,
            (uint64_t)amount);
}

void host_wait(volatile int64_t* posted, int64_t* balance) {
  while (*posted + *balance < 1) {
  }
  (*balance)--;
}

void host_finish(void) { host_ring_sync(host_slot(), HOST_REQ_FINISH, 0, 0); }

#endif /* NO_TESTBENCH */
