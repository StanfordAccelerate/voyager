#pragma once

#include <ac_int.h>

#include "../ArchitectureParams.h"

/**
 * \brief Run the polynomial approximation on a single vector element.
 *
 * This implementation is heavily optimized for Catapult HLS. The selection
 * logic in particular is hard to follow. It should be functionally equivalent
 * to its gold model - `poly_approx`.
 */
#pragma hls_design ccore
template <typename T>
T vepoly(T x, const T maxes[NUM_MAXES], const T ranges[NUM_RANGES][NUM_COEFFS],
         const ac_int<1, false> clamp_min, const ac_int<1, false> clamp_max) {
  static_assert(NUM_RANGES == NUM_MAXES + 1, "Ranges must fencepost the maxes");
  static_assert(NUM_COEFFS == 3, "We only support quadratic polynomials");
#ifndef __SYNTHESIS__
  // We assume that the `maxes` array is sorted.
  for (int i = 1; i < NUM_MAXES; i++) {
    assert(maxes[i - 1] <= maxes[i]);
  }
#endif

  if (clamp_min && x < maxes[0]) {
    x = maxes[0];
  } else if (clamp_max && maxes[NUM_MAXES - 1] < x) {
    x = maxes[NUM_MAXES - 1];
  }

  static_assert(NUM_MAXES == 6, "This code is hard-coded for 6 maxes");
  T c0, c1, c2;
  if (x <= maxes[0]) {
    c0 = ranges[0][0];
    c1 = ranges[0][1];
    c2 = ranges[0][2];
  } else if (x <= maxes[1]) {
    c0 = ranges[1][0];
    c1 = ranges[1][1];
    c2 = ranges[1][2];
  } else if (x <= maxes[2]) {
    c0 = ranges[2][0];
    c1 = ranges[2][1];
    c2 = ranges[2][2];
  } else if (x <= maxes[3]) {
    c0 = ranges[3][0];
    c1 = ranges[3][1];
    c2 = ranges[3][2];
  } else if (x <= maxes[4]) {
    c0 = ranges[4][0];
    c1 = ranges[4][1];
    c2 = ranges[4][2];
  } else if (x <= maxes[5]) {
    c0 = ranges[5][0];
    c1 = ranges[5][1];
    c2 = ranges[5][2];
  } else {
    c0 = ranges[6][0];
    c1 = ranges[6][1];
    c2 = ranges[6][2];
  }

  return c0 + x * (c1 + x * c2);
}
