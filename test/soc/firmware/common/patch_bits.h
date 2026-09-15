// clang-format off
#ifndef PATCH_BITS_H
#define PATCH_BITS_H

#include <stdint.h>

// Python/sympy floor semantics (result follows the divisor's sign), matching
// the interpreter's scalar ops -- NOT C's truncating % and /.
static inline int64_t vy_mod(int64_t a, int64_t b) {
  int64_t r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) r += b;
  return r;
}

static inline int64_t vy_fdiv(int64_t a, int64_t b) {
  int64_t q = a / b;
  int64_t r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) q -= 1;
  return q;
}

// Overwrites bit_len bits of a little-endian-serialized params blob starting
// at bit_off with the low bits of value. Byte-wise, so it never takes a
// RISC-V unaligned access trap.
void patch_bits(unsigned char *blob, uint32_t bit_off, uint32_t bit_len,
                uint64_t value);

#endif  // PATCH_BITS_H
// clang-format on
