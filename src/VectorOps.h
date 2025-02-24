#pragma once

#include <ac_math/ac_gelu_pwl.h>
#include <ac_math/ac_sigmoid_pwl.h>

using namespace ac_math;

#pragma hls_design ccore
template <typename Input, int Width>
void vadd(Pack1D<Input, Width>& op0, Pack1D<Input, Width>& op1,
          Pack1D<Input, Width>& res) {
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = static_cast<Input>(op0[i] + op1[i]);
  }
}

#pragma hls_design ccore
template <typename Input, int Width>
void vmult(Pack1D<Input, Width>& op0, Pack1D<Input, Width>& op1,
           Pack1D<Input, Width>& res) {
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = static_cast<Input>(op0[i] * op1[i]);
  }
}

#pragma hls_design ccore
template <typename Input, int Width>
void vrelu(Pack1D<Input, Width>& op0, Pack1D<Input, Width> mask, bool useMask,
           Pack1D<Input, Width>& res) {
  Pack1D<Input, Width> tmp;

#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    tmp[i] = op0[i];
  }

  if (useMask) {
#pragma hls_unroll yes
    for (int i = 0; i < Width; i++) {
      tmp[i].masked_relu(mask[i]);
    }
  } else {
#pragma hls_unroll yes
    for (int i = 0; i < Width; i++) {
      tmp[i].relu();
    }
  }

#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = tmp[i];
  }
}

#pragma hls_design ccore
template <typename Input, int Width>
void vgelu(Pack1D<Input, Width>& op0, Pack1D<Input, Width>& res) {
  // typedef ac_fixed<9, 4, true, AC_RND, AC_SAT> input_type;
  // typedef ac_fixed<64, 32, true, AC_RND, AC_SAT> output_type;
  typedef ac_fixed<15, 7, true, AC_RND, AC_SAT> input_type;
  typedef ac_fixed<15, 7, true, AC_RND, AC_SAT> output_type;
  Pack1D<input_type, Width> tmp;

#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    // tmp[i] = op0[i].template to_ac_fixed<9, 4, false, AC_RND, AC_SAT>();
    tmp[i] = op0[i].template to_ac_fixed<15, 7, false, AC_RND, AC_SAT>();
  }

#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = ac_gelu_pwl<output_type>(tmp[i]);
  }
}

#pragma hls_design ccore
template <typename Input, int Width>
void vsilu(Pack1D<Input, Width>& op0, Pack1D<Input, Width>& res) {
  typedef ac_fixed<15, 7, true, AC_RND, AC_SAT> input_type;
  // typedef ac_fixed<30, 3, false, AC_RND, AC_SAT> output_type;
  typedef ac_fixed<15, 7, false, AC_RND, AC_SAT> output_type;
  Pack1D<input_type, Width> tmp;

#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    tmp[i] = op0[i].template to_ac_fixed<15, 7, true, AC_RND, AC_SAT>();
  }

#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = tmp[i] * ac_sigmoid_pwl<output_type>(tmp[i]);
  }
}

#pragma hls_design ccore
template <typename Input, typename Output, typename Scale, int Width>
void vquantize(Pack1D<Input, Width>& op0, Pack1D<Output, Width>& res,
               Scale scale) {
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = op0[i] / scale;
  }
}

#pragma hls_design ccore
template <typename Input, typename Output, int Width>
void vdequantize(Pack1D<Input, Width>& op0, Pack1D<Output, Width>& res,
                 ac_int<Output::width, false> scale_bits) {
  Output scale;
  scale.set_bits(scale_bits);

#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = static_cast<Output>(op0[i]) * scale;
  }
}

#pragma hls_design ccore
template <typename Input, int Width>
void vexp(Pack1D<Input, Width>& op0, Pack1D<Input, Width>& res) {
  Pack1D<Input, Width> tmp;
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    tmp[i] = op0[i];
  }

#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    tmp[i].exponential();
  }

// convert back to decoded format
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = tmp[i];
  }
}

#pragma hls_design ccore
template <typename Input, int Width>
void vscaleexp(Pack1D<Input, Width>& op0, ac_int<8, false> expScale,
               Pack1D<Input, Width>& res) {
#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = op0[i];
  }

#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i].expScale(expScale);
    // res[i].scale += expScale;
  }
}

// #pragma hls_design ccore
// template <typename Input, int Width>
// void vdiv(Pack1D<Input, Width>& op0, Pack1D<Input, Width>& op1,
//           Pack1D<Input, Width>& res) {
//   // convert to Posit16
//   Pack1D<Posit<16, 1>, Width> tmp;
// #pragma hls_unroll yes
//   for (int i = 0; i < Width; i++) {
//     tmp[i] = op1[i];
//   }

// #pragma hls_unroll yes
//   for (int i = 0; i < Width; i++) {
//     tmp[i].reciprocal();
//   }

// // convert back to decoded format
// #pragma hls_unroll yes
//   for (int i = 0; i < Width; i++) {
//     res[i] = tmp[i];
//   }

//   vmult<Input, Width>(op0, res, res);
// }

#pragma hls_design ccore
template <typename Input, int Width>
void vmultdiv(Pack1D<Input, Width>& op0, Pack1D<Input, Width>& op1,
              Pack1D<Input, Width>& res, bool div, bool square) {
  Pack1D<Input, Width> op1_factor;
  if (div) {
#pragma hls_unroll yes
    for (int i = 0; i < Width; i++) {
      // TODO Is it fine to update op1 with its reciprocal value ?
      op1[i].custom_converted_reciprocal();
      op1_factor[i] = op1[i];
    }

    // #pragma hls_unroll yes
    //     for (int i = 0; i < Width; i++) {
    //       tmp[i].reciprocal();
    //     }

    // // convert back to decoded format
    // #pragma hls_unroll yes
    //     for (int i = 0; i < Width; i++) {
    //       op_factor[i] = tmp[i];
    //     }

  } else if (square) {
    op1_factor = op0;
  } else {
    op1_factor = op1;
  }

#pragma hls_unroll yes
  for (int i = 0; i < Width; i++) {
    res[i] = static_cast<Input>(op0[i] * op1_factor[i]);
  }
}

// Need to overload treeadd and treemax for the number of dimensions that need
// to be supported

/*
 * Dimension = 64
 */
#pragma hls_design ccore
template <typename Input>
Input treeadd(Pack1D<Input, 64>& op) {
  Pack1D<Input, 32> lvl0;
#pragma hls_unroll yes
  for (int i = 0; i < 32; i++) {
    lvl0[i] = static_cast<Input>(op[i * 2] + op[i * 2 + 1]);
  }

  Pack1D<Input, 16> lvl1;
#pragma hls_unroll yes
  for (int i = 0; i < 16; i++) {
    lvl1[i] = static_cast<Input>(lvl0[i * 2] + lvl0[i * 2 + 1]);
  }

  Pack1D<Input, 8> lvl2;
#pragma hls_unroll yes
  for (int i = 0; i < 8; i++) {
    lvl2[i] = static_cast<Input>(lvl1[i * 2] + lvl1[i * 2 + 1]);
  }

  Pack1D<Input, 4> lvl3;
#pragma hls_unroll yes
  for (int i = 0; i < 4; i++) {
    lvl3[i] = static_cast<Input>(lvl2[i * 2] + lvl2[i * 2 + 1]);
  }

  Pack1D<Input, 2> lvl4;
#pragma hls_unroll yes
  for (int i = 0; i < 2; i++) {
    lvl4[i] = static_cast<Input>(lvl3[i * 2] + lvl3[i * 2 + 1]);
  }

  return static_cast<Input>(lvl4[0] + lvl4[1]);
}

#pragma hls_design ccore
template <typename Input>
Input treemax(Pack1D<Input, 64>& op) {
  Pack1D<Input, 32> lvl0;
#pragma hls_unroll yes
  for (int i = 0; i < 32; i++) {
    lvl0[i] = op[i * 2] < op[i * 2 + 1] ? op[i * 2 + 1] : op[i * 2];
  }

  Pack1D<Input, 16> lvl1;
#pragma hls_unroll yes
  for (int i = 0; i < 16; i++) {
    lvl1[i] = lvl0[i * 2] < lvl0[i * 2 + 1] ? lvl0[i * 2 + 1] : lvl0[i * 2];
  }

  Pack1D<Input, 8> lvl2;
#pragma hls_unroll yes
  for (int i = 0; i < 8; i++) {
    lvl2[i] = lvl1[i * 2] < lvl1[i * 2 + 1] ? lvl1[i * 2 + 1] : lvl1[i * 2];
  }

  Pack1D<Input, 4> lvl3;
#pragma hls_unroll yes
  for (int i = 0; i < 4; i++) {
    lvl3[i] = lvl2[i * 2] < lvl2[i * 2 + 1] ? lvl2[i * 2 + 1] : lvl2[i * 2];
  }

  Pack1D<Input, 2> lvl4;
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
template <typename Input>
Input treeadd(Pack1D<Input, 32>& op) {
  Pack1D<Input, 16> lvl0;
#pragma hls_unroll yes
  for (int i = 0; i < 16; i++) {
    lvl0[i] = static_cast<Input>(op[i * 2] + op[i * 2 + 1]);
  }

  Pack1D<Input, 8> lvl1;
#pragma hls_unroll yes
  for (int i = 0; i < 8; i++) {
    lvl1[i] = static_cast<Input>(lvl0[i * 2] + lvl0[i * 2 + 1]);
  }

  Pack1D<Input, 4> lvl2;
#pragma hls_unroll yes
  for (int i = 0; i < 4; i++) {
    lvl2[i] = static_cast<Input>(lvl1[i * 2] + lvl1[i * 2 + 1]);
  }

  Pack1D<Input, 2> lvl3;
#pragma hls_unroll yes
  for (int i = 0; i < 2; i++) {
    lvl3[i] = static_cast<Input>(lvl2[i * 2] + lvl2[i * 2 + 1]);
  }

  return static_cast<Input>(lvl3[0] + lvl3[1]);
}

#pragma hls_design ccore
template <typename Input>
Input treemax(Pack1D<Input, 32>& op) {
  Pack1D<Input, 16> lvl0;
#pragma hls_unroll yes
  for (int i = 0; i < 16; i++) {
    lvl0[i] = op[i * 2] < op[i * 2 + 1] ? op[i * 2 + 1] : op[i * 2];
  }

  Pack1D<Input, 8> lvl1;
#pragma hls_unroll yes
  for (int i = 0; i < 8; i++) {
    lvl1[i] = lvl0[i * 2] < lvl0[i * 2 + 1] ? lvl0[i * 2 + 1] : lvl0[i * 2];
  }

  Pack1D<Input, 4> lvl2;
#pragma hls_unroll yes
  for (int i = 0; i < 4; i++) {
    lvl2[i] = lvl1[i * 2] < lvl1[i * 2 + 1] ? lvl1[i * 2 + 1] : lvl1[i * 2];
  }

  Pack1D<Input, 2> lvl3;
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
template <typename Input>
Input treeadd(Pack1D<Input, 16>& op) {
  Pack1D<Input, 8> lvl0;
#pragma hls_unroll yes
  for (int i = 0; i < 8; i++) {
    lvl0[i] = static_cast<Input>(op[i * 2] + op[i * 2 + 1]);
  }

  Pack1D<Input, 4> lvl1;
#pragma hls_unroll yes
  for (int i = 0; i < 4; i++) {
    lvl1[i] = static_cast<Input>(lvl0[i * 2] + lvl0[i * 2 + 1]);
  }

  Pack1D<Input, 2> lvl2;
#pragma hls_unroll yes
  for (int i = 0; i < 2; i++) {
    lvl2[i] = static_cast<Input>(lvl1[i * 2] + lvl1[i * 2 + 1]);
  }

  return static_cast<Input>(lvl2[0] + lvl2[1]);
}

#pragma hls_design ccore
template <typename Input>
Input treemax(Pack1D<Input, 16>& op) {
  Pack1D<Input, 8> lvl0;
#pragma hls_unroll yes
  for (int i = 0; i < 8; i++) {
    lvl0[i] = op[i * 2] < op[i * 2 + 1] ? op[i * 2 + 1] : op[i * 2];
  }

  Pack1D<Input, 4> lvl1;
#pragma hls_unroll yes
  for (int i = 0; i < 4; i++) {
    lvl1[i] = lvl0[i * 2] < lvl0[i * 2 + 1] ? lvl0[i * 2 + 1] : lvl0[i * 2];
  }

  Pack1D<Input, 2> lvl2;
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
template <typename Input>
Input treeadd(Pack1D<Input, 8>& op) {
  Pack1D<Input, 4> lvl0;
#pragma hls_unroll yes
  for (int i = 0; i < 4; i++) {
    lvl0[i] = static_cast<Input>(op[i * 2] + op[i * 2 + 1]);
  }

  Pack1D<Input, 2> lvl1;
#pragma hls_unroll yes
  for (int i = 0; i < 2; i++) {
    lvl1[i] = static_cast<Input>(lvl0[i * 2] + lvl0[i * 2 + 1]);
  }

  return static_cast<Input>(lvl1[0] + lvl1[1]);
}

#pragma hls_design ccore
template <typename Input>
Input treemax(Pack1D<Input, 8>& op) {
  Pack1D<Input, 4> lvl0;
#pragma hls_unroll yes
  for (int i = 0; i < 4; i++) {
    lvl0[i] = op[i * 2] < op[i * 2 + 1] ? op[i * 2 + 1] : op[i * 2];
  }

  Pack1D<Input, 2> lvl1;
#pragma hls_unroll yes
  for (int i = 0; i < 2; i++) {
    lvl1[i] = lvl0[i * 2] < lvl0[i * 2 + 1] ? lvl0[i * 2 + 1] : lvl0[i * 2];
  }

  return lvl1[0] < lvl1[1] ? lvl1[1] : lvl1[0];
}

// // Compile-time template recursion
// // used to generate tree structures
// template <typename Input, int Width>
// struct TreeOps {
// #pragma hls_design ccore
//   Input treeadd(Pack1D<Input, Width>& op) {
//     // split into two
//     Pack1D<Input, Width / 2> op_half_0;
//     Pack1D<Input, Width / 2> op_half_1;
// #pragma hls_unroll yes
//     for (int i = 0; i < Width / 2; i++) {
//       op_half_0[i] = op[i];
//       op_half_1[i] = op[Width / 2 + i];
//     }

//     Pack1D<Input, 2> res;
//     res[0] = TreeOps<Input, Width / 2>().treeadd(op_half_0);
//     res[1] = TreeOps<Input, Width / 2>().treeadd(op_half_1);
//     return TreeOps<Input, 2>().treeadd(res);
//   }

// #pragma hls_design ccore
//   Input treemax(Pack1D<Input, Width>& op) {
//     // split into two
//     Pack1D<Input, Width / 2> op_half_0;
//     Pack1D<Input, Width / 2> op_half_1;
// #pragma hls_unroll yes
//     for (int i = 0; i < Width / 2; i++) {
//       op_half_0[i] = op[i];
//       op_half_1[i] = op[Width / 2 + i];
//     }

//     Pack1D<Input, 2> res;
//     res[0] = TreeOps<Input, Width / 2>().treemax(op_half_0);
//     res[1] = TreeOps<Input, Width / 2>().treemax(op_half_1);
//     return TreeOps<Input, 2>().treemax(res);
//   }
// };

// template <typename Input>
// struct TreeOps<Input, 2> {
// #pragma hls_design ccore
// #pragma ccore_type combinational
//   Input treeadd(Pack1D<Input, 2>& op) {
//     return static_cast<Input>(op[0] + op[1]);
//   }
//   Input treemax(Pack1D<Input, 2>& op) {
//     return op[0] < op[1] ? op[1] : op[0];
//   }
// };
