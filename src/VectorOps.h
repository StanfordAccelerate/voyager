#pragma once

#pragma hls_design ccore
template <typename ACC_DTYPE, int WIDTH>
void vadd(Pack1D<ACC_DTYPE, WIDTH>& op0, Pack1D<ACC_DTYPE, WIDTH>& op1,
          Pack1D<ACC_DTYPE, WIDTH>& res) {
#pragma hls_unroll yes
  for (int i = 0; i < WIDTH; i++) {
    res[i] = static_cast<ACC_DTYPE>(op0[i] + op1[i]);
  }
}

#pragma hls_design ccore
template <typename ACC_DTYPE, int WIDTH>
void vmult(Pack1D<ACC_DTYPE, WIDTH>& op0, Pack1D<ACC_DTYPE, WIDTH>& op1,
           Pack1D<ACC_DTYPE, WIDTH>& res) {
#pragma hls_unroll yes
  for (int i = 0; i < WIDTH; i++) {
    res[i] = static_cast<ACC_DTYPE>(op0[i] * op1[i]);
  }
}

#pragma hls_design ccore
template <typename ACC_DTYPE, int WIDTH>
void vrelu(Pack1D<ACC_DTYPE, WIDTH>& op0, Pack1D<ACC_DTYPE, WIDTH> mask,
           bool useMask, Pack1D<ACC_DTYPE, WIDTH>& res) {
  Pack1D<ACC_DTYPE, WIDTH> tmp;

#pragma hls_unroll yes
  for (int i = 0; i < WIDTH; i++) {
    tmp[i] = op0[i];
  }

  if (useMask) {
#pragma hls_unroll yes
    for (int i = 0; i < WIDTH; i++) {
      tmp[i].masked_relu(mask[i]);
    }
  } else {
#pragma hls_unroll yes
    for (int i = 0; i < WIDTH; i++) {
      tmp[i].relu();
    }
  }

#pragma hls_unroll yes
  for (int i = 0; i < WIDTH; i++) {
    res[i] = tmp[i];
  }
}

#pragma hls_design ccore
template <typename VEC_DTYPE, typename QUANTIZED_TYPE, int WIDTH>
void vquantize(Pack1D<VEC_DTYPE, WIDTH>& op0,
               Pack1D<QUANTIZED_TYPE, WIDTH>& res,
               ac_int<VEC_DTYPE::width, false> scale) {
#pragma hls_unroll yes
  for (int i = 0; i < WIDTH; i++) {
    res[i] = op0[i]
                 .template quantize<QUANTIZED_TYPE::ac_int_rep::width,
                                    QUANTIZED_TYPE::ac_int_rep::sign>(scale);
  }
}

#pragma hls_design ccore
template <typename VEC_DTYPE, typename QUANTIZED_TYPE, typename SCALE_TYPE,
          int WIDTH>
void vmxquantize(Pack1D<VEC_DTYPE, WIDTH>& op0,
                 Pack1D<QUANTIZED_TYPE, WIDTH>& res, SCALE_TYPE& scale) {
#pragma hls_unroll yes
  for (int i = 0; i < WIDTH; i++) {
    res[i] = op0[i]
                 .template quantize<QUANTIZED_TYPE::ac_int_rep::width,
                                    QUANTIZED_TYPE::ac_int_rep::sign>(scale);
  }
}

#pragma hls_design ccore
template <typename VEC_DTYPE, typename QUANTIZED_TYPE, int WIDTH>
void vdequantize(Pack1D<QUANTIZED_TYPE, WIDTH>& op0,
                 Pack1D<VEC_DTYPE, WIDTH>& res,
                 ac_int<VEC_DTYPE::width, false> scale_bits) {
  VEC_DTYPE scale;
  scale.setbits(scale_bits);

  if constexpr (std::is_same_v<VEC_DTYPE, QUANTIZED_TYPE>) {
#pragma hls_unroll yes
    for (int i = 0; i < WIDTH; i++) {
      res[i] = op0[i] * scale;
    }
  } else {
#pragma hls_unroll yes
    for (int i = 0; i < WIDTH; i++) {
      res[i] = op0[i].dequantize(scale);
    }
  }
}

#pragma hls_design ccore
template <typename ACC_DTYPE, int WIDTH>
void vexp(Pack1D<ACC_DTYPE, WIDTH>& op0, Pack1D<ACC_DTYPE, WIDTH>& res) {
  // convert to ACCUM_DATATYPE
  // Pack1D<ACCUM_DATATYPE, WIDTH> tmp;
  Pack1D<ACC_DTYPE, WIDTH> tmp;
#pragma hls_unroll yes
  for (int i = 0; i < WIDTH; i++) {
    tmp[i] = op0[i];
  }

#pragma hls_unroll yes
  for (int i = 0; i < WIDTH; i++) {
    tmp[i].exponential();
  }

// convert back to decoded format
#pragma hls_unroll yes
  for (int i = 0; i < WIDTH; i++) {
    res[i] = tmp[i];
  }
}

#pragma hls_design ccore
template <typename ACC_DTYPE, int WIDTH>
void vscaleexp(Pack1D<ACC_DTYPE, WIDTH>& op0, ac_int<8, false> expScale,
               Pack1D<ACC_DTYPE, WIDTH>& res) {
#pragma hls_unroll yes
  for (int i = 0; i < WIDTH; i++) {
    res[i] = op0[i];
  }

#pragma hls_unroll yes
  for (int i = 0; i < WIDTH; i++) {
    res[i].expScale(expScale);
    // res[i].scale += expScale;
  }
}

// #pragma hls_design ccore
// template <typename ACC_DTYPE, int WIDTH>
// void vdiv(Pack1D<ACC_DTYPE, WIDTH>& op0, Pack1D<ACC_DTYPE, WIDTH>& op1,
//           Pack1D<ACC_DTYPE, WIDTH>& res) {
//   // convert to Posit16
//   Pack1D<Posit<16, 1>, WIDTH> tmp;
// #pragma hls_unroll yes
//   for (int i = 0; i < WIDTH; i++) {
//     tmp[i] = op1[i];
//   }

// #pragma hls_unroll yes
//   for (int i = 0; i < WIDTH; i++) {
//     tmp[i].reciprocal();
//   }

// // convert back to decoded format
// #pragma hls_unroll yes
//   for (int i = 0; i < WIDTH; i++) {
//     res[i] = tmp[i];
//   }

//   vmult<ACC_DTYPE, WIDTH>(op0, res, res);
// }

#pragma hls_design ccore
template <typename ACC_DTYPE, int WIDTH>
void vmultdiv(Pack1D<ACC_DTYPE, WIDTH>& op0, Pack1D<ACC_DTYPE, WIDTH>& op1,
              Pack1D<ACC_DTYPE, WIDTH>& res, bool div, bool square) {
  Pack1D<ACC_DTYPE, WIDTH> op1_factor;
  if (div) {
#pragma hls_unroll yes
    for (int i = 0; i < WIDTH; i++) {
      // TODO Is it fine to update op1 with its reciprocal value ?
      op1[i].custom_converted_reciprocal();
      op1_factor[i] = op1[i];
    }

    // #pragma hls_unroll yes
    //     for (int i = 0; i < WIDTH; i++) {
    //       tmp[i].reciprocal();
    //     }

    // // convert back to decoded format
    // #pragma hls_unroll yes
    //     for (int i = 0; i < WIDTH; i++) {
    //       op_factor[i] = tmp[i];
    //     }

  } else if (square) {
    op1_factor = op0;
  } else {
    op1_factor = op1;
  }

#pragma hls_unroll yes
  for (int i = 0; i < WIDTH; i++) {
    res[i] = static_cast<ACC_DTYPE>(op0[i] * op1_factor[i]);
  }
}

// Need to overload treeadd and treemax for the number of dimensions that need
// to be supported

/*
 * Dimension = 64
 */
#pragma hls_design ccore
template <typename ACC_DTYPE>
ACC_DTYPE treeadd(Pack1D<ACC_DTYPE, 64>& op) {
  Pack1D<ACC_DTYPE, 32> lvl0;
#pragma hls_unroll yes
  for (int i = 0; i < 32; i++) {
    lvl0[i] = static_cast<ACC_DTYPE>(op[i * 2] + op[i * 2 + 1]);
  }

  Pack1D<ACC_DTYPE, 16> lvl1;
#pragma hls_unroll yes
  for (int i = 0; i < 16; i++) {
    lvl1[i] = static_cast<ACC_DTYPE>(lvl0[i * 2] + lvl0[i * 2 + 1]);
  }

  Pack1D<ACC_DTYPE, 8> lvl2;
#pragma hls_unroll yes
  for (int i = 0; i < 8; i++) {
    lvl2[i] = static_cast<ACC_DTYPE>(lvl1[i * 2] + lvl1[i * 2 + 1]);
  }

  Pack1D<ACC_DTYPE, 4> lvl3;
#pragma hls_unroll yes
  for (int i = 0; i < 4; i++) {
    lvl3[i] = static_cast<ACC_DTYPE>(lvl2[i * 2] + lvl2[i * 2 + 1]);
  }

  Pack1D<ACC_DTYPE, 2> lvl4;
#pragma hls_unroll yes
  for (int i = 0; i < 2; i++) {
    lvl4[i] = static_cast<ACC_DTYPE>(lvl3[i * 2] + lvl3[i * 2 + 1]);
  }

  return static_cast<ACC_DTYPE>(lvl4[0] + lvl4[1]);
}

#pragma hls_design ccore
template <typename ACC_DTYPE>
ACC_DTYPE treemax(Pack1D<ACC_DTYPE, 64>& op) {
  Pack1D<ACC_DTYPE, 32> lvl0;
#pragma hls_unroll yes
  for (int i = 0; i < 32; i++) {
    lvl0[i] = op[i * 2] < op[i * 2 + 1] ? op[i * 2 + 1] : op[i * 2];
  }

  Pack1D<ACC_DTYPE, 16> lvl1;
#pragma hls_unroll yes
  for (int i = 0; i < 16; i++) {
    lvl1[i] = lvl0[i * 2] < lvl0[i * 2 + 1] ? lvl0[i * 2 + 1] : lvl0[i * 2];
  }

  Pack1D<ACC_DTYPE, 8> lvl2;
#pragma hls_unroll yes
  for (int i = 0; i < 8; i++) {
    lvl2[i] = lvl1[i * 2] < lvl1[i * 2 + 1] ? lvl1[i * 2 + 1] : lvl1[i * 2];
  }

  Pack1D<ACC_DTYPE, 4> lvl3;
#pragma hls_unroll yes
  for (int i = 0; i < 4; i++) {
    lvl3[i] = lvl2[i * 2] < lvl2[i * 2 + 1] ? lvl2[i * 2 + 1] : lvl2[i * 2];
  }

  Pack1D<ACC_DTYPE, 2> lvl4;
#pragma hls_unroll yes
  for (int i = 0; i < 2; i++) {
    lvl4[i] = lvl3[i * 2] < lvl3[i * 2 + 1] ? lvl3[i * 2 + 1] : lvl3[i * 2];
  }

  return lvl4[0] < lvl4[1] ? lvl4[1] : lvl4[0];
}

/*
 * Dimension = 32
 */
#pragma hls_design ccore
template <typename ACC_DTYPE>
ACC_DTYPE treeadd(Pack1D<ACC_DTYPE, 32>& op) {
  Pack1D<ACC_DTYPE, 16> lvl0;
#pragma hls_unroll yes
  for (int i = 0; i < 16; i++) {
    lvl0[i] = static_cast<ACC_DTYPE>(op[i * 2] + op[i * 2 + 1]);
  }

  Pack1D<ACC_DTYPE, 8> lvl1;
#pragma hls_unroll yes
  for (int i = 0; i < 8; i++) {
    lvl1[i] = static_cast<ACC_DTYPE>(lvl0[i * 2] + lvl0[i * 2 + 1]);
  }

  Pack1D<ACC_DTYPE, 4> lvl2;
#pragma hls_unroll yes
  for (int i = 0; i < 4; i++) {
    lvl2[i] = static_cast<ACC_DTYPE>(lvl1[i * 2] + lvl1[i * 2 + 1]);
  }

  Pack1D<ACC_DTYPE, 2> lvl3;
#pragma hls_unroll yes
  for (int i = 0; i < 2; i++) {
    lvl3[i] = static_cast<ACC_DTYPE>(lvl2[i * 2] + lvl2[i * 2 + 1]);
  }

  return static_cast<ACC_DTYPE>(lvl3[0] + lvl3[1]);
}

#pragma hls_design ccore
template <typename ACC_DTYPE>
ACC_DTYPE treemax(Pack1D<ACC_DTYPE, 32>& op) {
  Pack1D<ACC_DTYPE, 16> lvl0;
#pragma hls_unroll yes
  for (int i = 0; i < 16; i++) {
    lvl0[i] = op[i * 2] < op[i * 2 + 1] ? op[i * 2 + 1] : op[i * 2];
  }

  Pack1D<ACC_DTYPE, 8> lvl1;
#pragma hls_unroll yes
  for (int i = 0; i < 8; i++) {
    lvl1[i] = lvl0[i * 2] < lvl0[i * 2 + 1] ? lvl0[i * 2 + 1] : lvl0[i * 2];
  }

  Pack1D<ACC_DTYPE, 4> lvl2;
#pragma hls_unroll yes
  for (int i = 0; i < 4; i++) {
    lvl2[i] = lvl1[i * 2] < lvl1[i * 2 + 1] ? lvl1[i * 2 + 1] : lvl1[i * 2];
  }

  Pack1D<ACC_DTYPE, 2> lvl3;
#pragma hls_unroll yes
  for (int i = 0; i < 2; i++) {
    lvl3[i] = lvl2[i * 2] < lvl2[i * 2 + 1] ? lvl2[i * 2 + 1] : lvl2[i * 2];
  }

  return lvl3[0] < lvl3[1] ? lvl3[1] : lvl3[0];
}

/*
 * Dimension = 16
 */
#pragma hls_design ccore
template <typename ACC_DTYPE>
ACC_DTYPE treeadd(Pack1D<ACC_DTYPE, 16>& op) {
  Pack1D<ACC_DTYPE, 8> lvl0;
#pragma hls_unroll yes
  for (int i = 0; i < 8; i++) {
    lvl0[i] = static_cast<ACC_DTYPE>(op[i * 2] + op[i * 2 + 1]);
  }

  Pack1D<ACC_DTYPE, 4> lvl1;
#pragma hls_unroll yes
  for (int i = 0; i < 4; i++) {
    lvl1[i] = static_cast<ACC_DTYPE>(lvl0[i * 2] + lvl0[i * 2 + 1]);
  }

  Pack1D<ACC_DTYPE, 2> lvl2;
#pragma hls_unroll yes
  for (int i = 0; i < 2; i++) {
    lvl2[i] = static_cast<ACC_DTYPE>(lvl1[i * 2] + lvl1[i * 2 + 1]);
  }

  return static_cast<ACC_DTYPE>(lvl2[0] + lvl2[1]);
}

#pragma hls_design ccore
template <typename ACC_DTYPE>
ACC_DTYPE treemax(Pack1D<ACC_DTYPE, 16>& op) {
  Pack1D<ACC_DTYPE, 8> lvl0;
#pragma hls_unroll yes
  for (int i = 0; i < 8; i++) {
    lvl0[i] = op[i * 2] < op[i * 2 + 1] ? op[i * 2 + 1] : op[i * 2];
  }

  Pack1D<ACC_DTYPE, 4> lvl1;
#pragma hls_unroll yes
  for (int i = 0; i < 4; i++) {
    lvl1[i] = lvl0[i * 2] < lvl0[i * 2 + 1] ? lvl0[i * 2 + 1] : lvl0[i * 2];
  }

  Pack1D<ACC_DTYPE, 2> lvl2;
#pragma hls_unroll yes
  for (int i = 0; i < 2; i++) {
    lvl2[i] = lvl1[i * 2] < lvl1[i * 2 + 1] ? lvl1[i * 2 + 1] : lvl1[i * 2];
  }

  return lvl2[0] < lvl2[1] ? lvl2[1] : lvl2[0];
}

/*
 * Dimension = 8
 */
#pragma hls_design ccore
template <typename ACC_DTYPE>
ACC_DTYPE treeadd(Pack1D<ACC_DTYPE, 8>& op) {
  Pack1D<ACC_DTYPE, 4> lvl0;
#pragma hls_unroll yes
  for (int i = 0; i < 4; i++) {
    lvl0[i] = static_cast<ACC_DTYPE>(op[i * 2] + op[i * 2 + 1]);
  }

  Pack1D<ACC_DTYPE, 2> lvl1;
#pragma hls_unroll yes
  for (int i = 0; i < 2; i++) {
    lvl1[i] = static_cast<ACC_DTYPE>(lvl0[i * 2] + lvl0[i * 2 + 1]);
  }

  return static_cast<ACC_DTYPE>(lvl1[0] + lvl1[1]);
}

#pragma hls_design ccore
template <typename ACC_DTYPE>
ACC_DTYPE treemax(Pack1D<ACC_DTYPE, 8>& op) {
  Pack1D<ACC_DTYPE, 4> lvl0;
#pragma hls_unroll yes
  for (int i = 0; i < 4; i++) {
    lvl0[i] = op[i * 2] < op[i * 2 + 1] ? op[i * 2 + 1] : op[i * 2];
  }

  Pack1D<ACC_DTYPE, 2> lvl1;
#pragma hls_unroll yes
  for (int i = 0; i < 2; i++) {
    lvl1[i] = lvl0[i * 2] < lvl0[i * 2 + 1] ? lvl0[i * 2 + 1] : lvl0[i * 2];
  }

  return lvl1[0] < lvl1[1] ? lvl1[1] : lvl1[0];
}

// // Compile-time template recursion
// // used to generate tree structures
// template <typename ACC_DTYPE, int WIDTH>
// struct TreeOps {
// #pragma hls_design ccore
//   ACC_DTYPE treeadd(Pack1D<ACC_DTYPE, WIDTH>& op) {
//     // split into two
//     Pack1D<ACC_DTYPE, WIDTH / 2> op_half_0;
//     Pack1D<ACC_DTYPE, WIDTH / 2> op_half_1;
// #pragma hls_unroll yes
//     for (int i = 0; i < WIDTH / 2; i++) {
//       op_half_0[i] = op[i];
//       op_half_1[i] = op[WIDTH / 2 + i];
//     }

//     Pack1D<ACC_DTYPE, 2> res;
//     res[0] = TreeOps<ACC_DTYPE, WIDTH / 2>().treeadd(op_half_0);
//     res[1] = TreeOps<ACC_DTYPE, WIDTH / 2>().treeadd(op_half_1);
//     return TreeOps<ACC_DTYPE, 2>().treeadd(res);
//   }

// #pragma hls_design ccore
//   ACC_DTYPE treemax(Pack1D<ACC_DTYPE, WIDTH>& op) {
//     // split into two
//     Pack1D<ACC_DTYPE, WIDTH / 2> op_half_0;
//     Pack1D<ACC_DTYPE, WIDTH / 2> op_half_1;
// #pragma hls_unroll yes
//     for (int i = 0; i < WIDTH / 2; i++) {
//       op_half_0[i] = op[i];
//       op_half_1[i] = op[WIDTH / 2 + i];
//     }

//     Pack1D<ACC_DTYPE, 2> res;
//     res[0] = TreeOps<ACC_DTYPE, WIDTH / 2>().treemax(op_half_0);
//     res[1] = TreeOps<ACC_DTYPE, WIDTH / 2>().treemax(op_half_1);
//     return TreeOps<ACC_DTYPE, 2>().treemax(res);
//   }
// };

// template <typename ACC_DTYPE>
// struct TreeOps<ACC_DTYPE, 2> {
// #pragma hls_design ccore
// #pragma ccore_type combinational
//   ACC_DTYPE treeadd(Pack1D<ACC_DTYPE, 2>& op) {
//     return static_cast<ACC_DTYPE>(op[0] + op[1]);
//   }
//   ACC_DTYPE treemax(Pack1D<ACC_DTYPE, 2>& op) {
//     return op[0] < op[1] ? op[1] : op[0];
//   }
// };
