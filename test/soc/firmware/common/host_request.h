/* The firmware's interface to the SoC testbench.
 *
 * Sphinx has no DMA engine: in the testbench-driven simulation modes the
 * testbench performs the program's DRAM<->scratchpad transfers on request.
 * The firmware writes a request into a ring of slots in its own memory (the
 * mailbox), then rings a doorbell by posting one credit to hardware
 * semaphore HOST_DOORBELL_SEMAPHORE, which the Verilog collateral watches.
 * The testbench reads the slots through the VPI backdoor, performs the
 * transfer, and reports completion by adding `amount` to the counter cell
 * the request names -- a cell the firmware only ever reads.
 *
 * This header is included verbatim by the RISC-V firmware and by the
 * testbench (SoCSimulation.cc): one layout, one set of codes. Every field
 * is 64 bits so both compilers lay the structs out identically.
 *
 * Built with NO_TESTBENCH (full JTAG mode, the chip: the image is preloaded
 * and nothing services requests) the firmware calls compile to nothing. */
#ifndef HOST_REQUEST_H
#define HOST_REQUEST_H

#include <stdint.h>

/* Request kinds. COPY and POST complete asynchronously through their cell;
 * the others are serviced before the firmware continues. */
#define HOST_REQ_COPY 1    /* voyager::async_copy, by ordinal */
#define HOST_REQ_ZERO 2    /* zero fill of a buffer, by ordinal */
#define HOST_REQ_HOST_OP 3 /* a host tensor op (slice, pad, ...), by ordinal \
                            */
#define HOST_REQ_POST 4    /* add `amount` to `cell` once the units retired */
#define HOST_REQ_FINISH 5  /* the program is complete: grade the outputs */

#define HOST_REQ_MAX_ARGS 16
#define HOST_MAILBOX_SLOTS 16
#define HOST_NUM_UNITS 4 /* matrix, vector, mvm, spmm: the testbench's codes \
                          */
#define HOST_DOORBELL_SEMAPHORE 7

typedef struct {
  uint64_t kind;
  /* The prim this request names: its index in the deterministic walk of the
   * layer's selected ops that HostRequests.cc defines and the emitter and
   * the testbench both perform (COPY, ZERO, HOST_OP). */
  uint64_t ordinal;
  /* Address of the counter the testbench adds `amount` to on completion;
   * 0 for none. */
  uint64_t cell;
  uint64_t amount;
  /* POST: per unit, the number of ops the firmware had issued when the
   * program posted. The testbench completes the request once that many
   * done events, each settled, have been observed on every unit. */
  uint64_t retired[HOST_NUM_UNITS];
  /* Run-time values of the scalar SSA names the prim references, in the
   * order request_scalar_names() lists them (sorted by name). */
  uint64_t nargs;
  int64_t args[HOST_REQ_MAX_ARGS];
} host_request_t;

typedef struct {
  uint64_t head; /* requests written; the firmware's */
  uint64_t pad0_[7];
  uint64_t tail; /* requests taken; the testbench's */
  uint64_t pad1_[7];
  host_request_t slots[HOST_MAILBOX_SLOTS];
} host_mailbox_t;

#ifndef __cplusplus
_Static_assert(sizeof(host_request_t) == 200, "host_request_t layout");
_Static_assert(sizeof(host_mailbox_t) == 128 + 200 * HOST_MAILBOX_SLOTS,
               "host_mailbox_t layout");

/* --- firmware side ------------------------------------------------------
 *
 * This core has no data cache: every store outside its registers is a bus
 * transaction. A request is therefore assembled in place -- the generated
 * program takes the next free slot with host_slot(), stores the scalar
 * values straight into slot->args[], and the runtime stores the header and
 * rings the doorbell. Nothing is staged elsewhere first. */

#ifdef NO_TESTBENCH
extern host_request_t host_dummy_request;
static inline void host_init(void) {}
static inline volatile host_request_t* host_slot(void) {
  return (volatile host_request_t*)&host_dummy_request;
}
static inline void host_copy(volatile host_request_t* slot, uint64_t ordinal,
                             uint64_t nargs, volatile int64_t* cell,
                             int64_t amount) {
  (void)slot;
  (void)ordinal;
  (void)nargs;
  (void)cell;
  (void)amount;
}
static inline void host_zero(uint64_t ordinal) { (void)ordinal; }
static inline void host_op(volatile host_request_t* slot, uint64_t ordinal,
                           uint64_t nargs) {
  (void)slot;
  (void)ordinal;
  (void)nargs;
}
static inline void host_post(volatile int64_t* cell, int64_t amount) {
  (void)cell;
  (void)amount;
}
static inline void host_wait(volatile int64_t* posted, int64_t* balance) {
  (void)posted;
  (void)balance;
}
static inline void host_finish(void) {}
#else
/* The mailbox, in the firmware's .bss; run_voyager.py reads its address
 * from the ELF and hands it to the testbench as HOST_MAILBOX. */
extern host_mailbox_t host_mailbox;

void host_init(void);
/* The next free slot; the caller fills slot->args[] before ringing. */
volatile host_request_t* host_slot(void);
/* Asynchronous: the testbench adds `amount` to *cell when the copy is done. */
void host_copy(volatile host_request_t* slot, uint64_t ordinal, uint64_t nargs,
               volatile int64_t* cell, int64_t amount);
/* Serviced before returning. */
void host_zero(uint64_t ordinal);
void host_op(volatile host_request_t* slot, uint64_t ordinal, uint64_t nargs);
/* A commit's retire post: *cell += amount once every op issued so far has
 * completed on the accelerator. */
void host_post(volatile int64_t* cell, int64_t amount);
/* A program semaphore: `posted` counts the testbench's completions (its
 * cell), `balance` the firmware's own posts minus its consumptions. Spins
 * until one credit is available, then consumes it. */
void host_wait(volatile int64_t* posted, int64_t* balance);
/* Grades the outputs; returns once the testbench has. */
void host_finish(void);
#endif /* NO_TESTBENCH */
#endif /* !__cplusplus */

#endif /* HOST_REQUEST_H */
