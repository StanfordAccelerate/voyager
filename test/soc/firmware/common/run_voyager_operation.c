#include "run_voyager_operation.h"

#include <stdio.h>
#include <string.h>

#include "mmio.h"
#include "voyager_address.h"
#include "voyager_params.h"

void send_serialized_params(const void* params, int width, uintptr_t address) {
  const uint64_t* ptr = (const uint64_t*)params;

  // round up to multiple of 64 bits
  int padded_width = ((width + 64 - 1) / 64) * 64;

  for (int i = 0; i < padded_width / 64; i++) {
    reg_write64(address, *(ptr++));
  }
}

void send_matrix_unit_params(const void* matrix_params) {
  send_serialized_params(matrix_params, matrix_params_width,
                         MATRIX_UNIT_PARAMS_IN);
}

void send_matrix_vector_unit_params(const void* matrix_params) {
  send_serialized_params(matrix_params, matrix_params_width,
                         MVM_UNIT_PARAMS_IN);
}

void send_spmm_unit_params(const void* matrix_params) {
  send_serialized_params(matrix_params, matrix_params_width,
                         SPMM_UNIT_PARAMS_IN);
}

void send_vector_params(const void* vector_params) {
  send_serialized_params(vector_params, vector_params_width,
                         VECTOR_UNIT_PARAMS_IN);
}

void send_vector_instructions(const void* vector_instructions) {
  send_serialized_params(vector_instructions, vector_instruction_config_width,
                         VECTOR_UNIT_PARAMS_IN);
}

void wait_for_accelerator_done() {
  while (reg_read8(ACCELERATOR_RUNNING)) {
    __asm__ volatile("wfi");
  }
}

void wait_for_dispatch_retired(uintptr_t inflight_reg) {
  /* The firmware's Harness::drain() for one just-sent invocation group.
   * ACCELERATOR_RUNNING alone cannot distinguish "granted but not yet
   * started" from "finished", so first busy-wait for the group's closing
   * unit to actually start (its inflight count rises), then for the whole
   * datapath to drain. */
  while (reg_read8(inflight_reg) == 0) {
  }
  wait_for_accelerator_done();
}

void enable_semaphore_wait() {
#ifndef DISABLE_SEMAPHORE_WAIT
  reg_write8(MATRIX_UNIT_WAIT_EN, 1);
  reg_write8(MATRIX_UNIT_WAIT_ID, 0);
  reg_write8(VECTOR_UNIT_WAIT_EN, 1);
  reg_write8(VECTOR_UNIT_WAIT_ID, 1);
  reg_write8(MVM_UNIT_WAIT_EN, 1);
  reg_write8(MVM_UNIT_WAIT_ID, 2);
  reg_write8(SPMM_UNIT_WAIT_EN, 1);
  reg_write8(SPMM_UNIT_WAIT_ID, 3);
#endif
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
