#pragma once

#include "AccelTypes.h"

namespace pack {

// Zero a scalar or recursively zero every scalar in a nested Pack1D
template <typename T>
void clear_pack(T& value) {
  value = 0;
}

// Zero every element while keeping the assignments visible for HLS unrolling
template <typename T, size_t Width>
void clear_pack(Pack1D<T, Width>& pack) {
#pragma hls_unroll yes
  for (unsigned int i = 0; i < Width; i++) {
    clear_pack(pack[i]);
  }
}

}  // namespace pack
