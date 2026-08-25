#ifndef RUN_VOYAGER_OPERATION_H
#define RUN_VOYAGER_OPERATION_H

#include <stdint.h>

typedef enum {
  MATRIX_UNIT,
  VECTOR_UNIT,
  MATRIX_VECTOR_UNIT,
  SPMM_UNIT,
} voyager_params_t;

void wait_for_accelerator_done();
void wait_for_dispatch_retired(uintptr_t inflight_reg);
void enable_semaphore_wait();
void send_voyager_params(const void** params, voyager_params_t* params_type,
                         int count);

// Per-unit sends the generated firmware calls directly.
void send_matrix_unit_params(const void* matrix_params);
void send_matrix_vector_unit_params(const void* matrix_params);
void send_spmm_unit_params(const void* matrix_params);
void send_vector_params(const void* vector_params);
void send_vector_instructions(const void* vector_instructions);

// Suppress printf in JTAG simulation where no frontend server is present
#ifdef SUPPRESS_PRINTF
#include <stdio.h>
#ifdef printf
#undef printf
#endif
// clang-format off
#define printf(...) do {} while (0)
// clang-format on
#endif

#endif
