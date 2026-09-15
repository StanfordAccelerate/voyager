#include "run_voyager_operation.h"

#include <stdio.h>
#include <string.h>

#include "mmio.h"
#include "voyager_address.h"
#include "voyager_params.h"

uint64_t unit_ops_issued[4];

void send_serialized_params(const void* params, int width, uintptr_t address) {
  const uint64_t* ptr = (const uint64_t*)params;

  // round up to multiple of 64 bits
  int padded_width = ((width + 64 - 1) / 64) * 64;

  for (int i = 0; i < padded_width / 64; i++) {
    reg_write64(address, *(ptr++));
  }
}

/* The one start ordering the harness enforces (Harness::release_starts): a
 * group's vector pass may not start before the group's compute pass has. A
 * fused op streams its matrix results through the accumulation buffer to the
 * vector unit, and a k-split reduction sends its epilogue with the last
 * k-tile's matrix params while the matrix unit is still busy with the
 * previous k-tile -- an ungated vector unit then consumes partial sums.
 *
 * Every unit waits on a start semaphore of its own, numbered like the unit,
 * that the firmware credits once per pass. A compute unit's credit is
 * granted just before the pass's last params word, after the flag the grant
 * itself raised in VOYAGER_INT_STATUS (bits 15:8 are the semaphores' change
 * flags) has been cleared; the unit cannot start before its record is
 * complete, so the next time that flag rises the pass has started. One
 * credit is outstanding per unit at a time, so two starts can never merge
 * into one observation -- counting completions from the same flags could,
 * and hung matmul_mx_default_1_fused at four tiles. The vector unit's credit
 * is posted once every compute pass of its group has started; the params
 * themselves go out immediately and the units pipeline as before. */

typedef struct {
  uint8_t sem;   /* the unit's start semaphore: its unit index */
  uint64_t flag; /* that semaphore's change flag in VOYAGER_INT_STATUS */
  int pending;   /* a credit granted whose consumption is not yet seen */
} start_gate_t;

static start_gate_t matrix_gate = {UNIT_MATRIX, 1ull << (8 + UNIT_MATRIX), 0};
static start_gate_t mvm_gate = {UNIT_MVM, 1ull << (8 + UNIT_MVM), 0};
static start_gate_t spmm_gate = {UNIT_SPMM, 1ull << (8 + UNIT_SPMM), 0};

/* Waits for the semaphore's next change and clears its flag. */
static void wait_change(uint64_t flag) {
  while ((reg_read64(VOYAGER_INT_STATUS) & flag) == 0) {
  }
  reg_write64(VOYAGER_INT_STATUS, flag); /* W1C */
}

static void post_credit(uint8_t sem) {
  reg_write8(DMA_SIGNAL_ID, sem);
  reg_write8(DMA_INC_SEM, 1); /* self-clearing pulse: +1 */
}

/* The unit's last pass has started: its credit was consumed. */
static void gate_wait_started(start_gate_t* gate) {
  if (!gate->pending) return;
  wait_change(gate->flag);
  gate->pending = 0;
}

/* A compute pass: every params word but the last, then the credit, then the
 * last word, so the credit can only be consumed once the record is in. */
static void send_compute_params(start_gate_t* gate, const void* params,
                                uintptr_t address) {
  const uint64_t* ptr = (const uint64_t*)params;
  const int words = (matrix_params_width + 63) / 64;
  for (int i = 0; i < words - 1; i++) {
    reg_write64(address, ptr[i]);
  }
  gate_wait_started(gate); /* one credit outstanding at a time */
  post_credit(gate->sem);
  wait_change(gate->flag); /* the grant's own change; no start possible yet */
  gate->pending = 1;
  reg_write64(address, ptr[words - 1]);
}

static void post_vector_credit(void) {
  gate_wait_started(&matrix_gate);
  gate_wait_started(&mvm_gate);
  gate_wait_started(&spmm_gate);
  post_credit(UNIT_VECTOR);
}

void send_matrix_unit_params(const void* matrix_params) {
  unit_ops_issued[UNIT_MATRIX]++;
  send_compute_params(&matrix_gate, matrix_params, MATRIX_UNIT_PARAMS_IN);
}

void send_matrix_vector_unit_params(const void* matrix_params) {
  unit_ops_issued[UNIT_MVM]++;
  send_compute_params(&mvm_gate, matrix_params, MVM_UNIT_PARAMS_IN);
}

void send_spmm_unit_params(const void* matrix_params) {
  unit_ops_issued[UNIT_SPMM]++;
  send_compute_params(&spmm_gate, matrix_params, SPMM_UNIT_PARAMS_IN);
}

void send_vector_params(const void* vector_params) {
  send_serialized_params(vector_params, vector_params_width,
                         VECTOR_UNIT_PARAMS_IN);
}

void send_vector_instructions(const void* vector_instructions) {
  unit_ops_issued[UNIT_VECTOR]++;
  send_serialized_params(vector_instructions, vector_instruction_config_width,
                         VECTOR_UNIT_PARAMS_IN);
  /* The vector unit holds with its params loaded until this credit lands. */
  post_vector_credit();
}

void wait_for_accelerator_done() {
  while (reg_read8(ACCELERATOR_RUNNING)) {
    __asm__ volatile("wfi");
  }
}

void wait_for_dispatch_retired(uintptr_t inflight_reg) {
  /* The firmware's Harness::drain() for one just-sent invocation group.
   * ACCELERATOR_RUNNING alone cannot distinguish "sent but not yet started"
   * from "finished", so first busy-wait for the group's closing unit to
   * actually start (its inflight count rises), then for the whole datapath
   * to drain. Valid ONLY for a synchronous dispatch: an asynchronous one
   * (inside a commit) is not waited for at all; its retirement is observed
   * by the testbench (host_post). */
  while (reg_read8(inflight_reg) == 0) {
  }
  wait_for_accelerator_done();
}

void enable_semaphore_wait() {
  /* Every unit starts on a credit the firmware posts to the unit's own
   * semaphore (send_compute_params, post_vector_credit); nothing signals.
   * Semaphore 7 is the testbench doorbell (host_request.h); 4..6 are
   * unused. */
  reg_write8(MATRIX_UNIT_WAIT_EN, 1);
  reg_write8(MATRIX_UNIT_WAIT_ID, UNIT_MATRIX);
  reg_write8(VECTOR_UNIT_WAIT_EN, 1);
  reg_write8(VECTOR_UNIT_WAIT_ID, UNIT_VECTOR);
  reg_write8(MVM_UNIT_WAIT_EN, 1);
  reg_write8(MVM_UNIT_WAIT_ID, UNIT_MVM);
  reg_write8(SPMM_UNIT_WAIT_EN, 1);
  reg_write8(SPMM_UNIT_WAIT_ID, UNIT_SPMM);
  reg_write8(MATRIX_UNIT_SIGNAL_EN, 0);
  reg_write8(VECTOR_UNIT_SIGNAL_EN, 0);
  reg_write8(MVM_UNIT_SIGNAL_EN, 0);
  reg_write8(SPMM_UNIT_SIGNAL_EN, 0);
  reg_write64(VOYAGER_INT_STATUS, 0xFF00); /* forget any earlier changes */
}

void send_voyager_params(const void** params, voyager_params_t* params_type,
                         int count) {
  for (int i = 0; i < count; i++) {
    if (params_type[i] == MATRIX_UNIT) {
      send_matrix_unit_params(params[i]);
    } else if (params_type[i] == MATRIX_VECTOR_UNIT) {
      send_matrix_vector_unit_params(params[i]);
    } else if (params_type[i] == SPMM_UNIT) {
      send_spmm_unit_params(params[i]);
    } else if (params_type[i] == VECTOR_UNIT) {
      send_vector_params(params[i]);
      send_vector_instructions(params[++i]);
    }
  }
}
