#pragma once

#pragma hls_design ccore
#pragma ccore_type combinational
template <typename ACC_DTYPE, int WIDTH>
void vadd(Pack1D<ACC_DTYPE, WIDTH>& op0, Pack1D<ACC_DTYPE, WIDTH>& op1,
          Pack1D<ACC_DTYPE, WIDTH>& res) {
#pragma hls_unroll yes
  for (int i = 0; i < WIDTH; i++) {
    res[i] = op0[i] + op1[i];
  }
}

#pragma hls_design ccore
#pragma ccore_type combinational
template <typename ACC_DTYPE, int WIDTH>
void vmult(Pack1D<ACC_DTYPE, WIDTH>& op0, Pack1D<ACC_DTYPE, WIDTH>& op1,
           Pack1D<ACC_DTYPE, WIDTH>& res) {
#pragma hls_unroll yes
  for (int i = 0; i < WIDTH; i++) {
    res[i] = op0[i] * op1[i];
  }
}

#pragma hls_design ccore
#pragma ccore_type combinational
template <typename ACC_DTYPE, int WIDTH>
void vrelu(Pack1D<ACC_DTYPE, WIDTH>& op0, Pack1D<ACC_DTYPE, WIDTH>& res) {
  Pack1D<ACC_DTYPE, WIDTH> tmp;

#pragma hls_unroll yes
  for (int i = 0; i < WIDTH; i++) {
    tmp[i] = op0[i];
  }

#pragma hls_unroll yes
  for (int i = 0; i < WIDTH; i++) {
    tmp[i].relu();
  }

#pragma hls_unroll yes
  for (int i = 0; i < WIDTH; i++) {
    res[i] = tmp[i];
  }
}

#pragma hls_design ccore
#pragma ccore_type combinational
template <typename ACC_DTYPE, int WIDTH>
void vexp(Pack1D<ACC_DTYPE, WIDTH>& op0, Pack1D<ACC_DTYPE, WIDTH>& res) {
  // convert to Posit16
  Pack1D<Posit<16, 0, 8, 16>, WIDTH> tmp;
#pragma hls_unroll yes
  for (int i = 0; i < WIDTH; i++) {
    tmp[i] = op0[i];
  }

#pragma hls_unroll yes
  for (int i = 0; i < WIDTH; i++) {
    tmp[i].exp();
  }

// convert back to decoded format
#pragma hls_unroll yes
  for (int i = 0; i < WIDTH; i++) {
    res[i] = tmp[i];
  }
}

#pragma hls_design ccore
#pragma ccore_type combinational
template <typename ACC_DTYPE, int WIDTH>
void vdiv(Pack1D<ACC_DTYPE, WIDTH>& op0, Pack1D<ACC_DTYPE, WIDTH>& op1,
          Pack1D<ACC_DTYPE, WIDTH>& res) {
  // convert to Posit16
  Pack1D<Posit<16, 0, 8, 16>, WIDTH> tmp;
#pragma hls_unroll yes
  for (int i = 0; i < WIDTH; i++) {
    tmp[i] = op0[i];
  }

#pragma hls_unroll yes
  for (int i = 0; i < WIDTH; i++) {
    tmp[i].reciprocal();
  }

// convert back to decoded format
#pragma hls_unroll yes
  for (int i = 0; i < WIDTH; i++) {
    res[i] = tmp[i];
  }

  vmult<ACC_DTYPE, WIDTH>(op0, res, res);
}

// Compile-time template recursion
// used to generate tree structures
template <typename ACC_DTYPE, int WIDTH>
struct TreeOps {
#pragma hls_design ccore
  ACC_DTYPE treeadd(Pack1D<ACC_DTYPE, WIDTH>& op) {
    // split into two
    Pack1D<ACC_DTYPE, WIDTH / 2> op_half_0;
    Pack1D<ACC_DTYPE, WIDTH / 2> op_half_1;
#pragma hls_unroll yes
    for (int i = 0; i < WIDTH / 2; i++) {
      op_half_0[i] = op[i];
      op_half_1[i] = op[WIDTH / 2 + i];
    }

    Pack1D<ACC_DTYPE, 2> res;
    res[0] = TreeOps<ACC_DTYPE, WIDTH / 2>().treeadd(op_half_0);
    res[1] = TreeOps<ACC_DTYPE, WIDTH / 2>().treeadd(op_half_1);
    return TreeOps<ACC_DTYPE, 2>().treeadd(res);
  }

#pragma hls_design ccore
  ACC_DTYPE treemax(Pack1D<ACC_DTYPE, WIDTH>& op) {
    // split into two
    Pack1D<ACC_DTYPE, WIDTH / 2> op_half_0;
    Pack1D<ACC_DTYPE, WIDTH / 2> op_half_1;
#pragma hls_unroll yes
    for (int i = 0; i < WIDTH / 2; i++) {
      op_half_0[i] = op[i];
      op_half_1[i] = op[WIDTH / 2 + i];
    }

    Pack1D<ACC_DTYPE, 2> res;
    res[0] = TreeOps<ACC_DTYPE, WIDTH / 2>().treemax(op_half_0);
    res[1] = TreeOps<ACC_DTYPE, WIDTH / 2>().treemax(op_half_1);
    return TreeOps<ACC_DTYPE, 2>().treemax(res);
  }
};

template <typename ACC_DTYPE>
struct TreeOps<ACC_DTYPE, 2> {
#pragma hls_design ccore
#pragma ccore_type combinational
  ACC_DTYPE treeadd(Pack1D<ACC_DTYPE, 2>& op) { return op[0] + op[1]; }
  ACC_DTYPE treemax(Pack1D<ACC_DTYPE, 2>& op) {
    return op[0] < op[1] ? op[1] : op[0];
  }
};
