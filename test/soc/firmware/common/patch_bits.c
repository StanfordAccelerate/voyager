#include "patch_bits.h"

void patch_bits(unsigned char *blob, uint32_t bit_off, uint32_t bit_len,
                uint64_t value) {
  for (uint32_t i = 0; i < bit_len; i++) {
    uint32_t bit = bit_off + i;
    unsigned char mask = (unsigned char)(1u << (bit % 8u));
    if ((value >> i) & 1u) {
      blob[bit / 8u] |= mask;
    } else {
      blob[bit / 8u] &= (unsigned char)~mask;
    }
  }
}
