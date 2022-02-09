#pragma once

struct VInst {
  ac_int<1, false> src0;
  ac_int<1, false> src1;
  ac_int<1, false> src2;
  ac_int<1, false> src3;
  static const unsigned int reg = 0;
  static const unsigned int interface = 1;

  ac_int<3, false> vOp0;  // add, mult
  ac_int<3, false> vOp1;  // add
  ac_int<3, false> vOp2;  // exp, relu
  ac_int<3, false> vOp3;  // add, div
  static const unsigned int nop = 0;
  static const unsigned int vadd = 1;
  static const unsigned int vmult = 2;
  static const unsigned int vexp = 3;
  static const unsigned int vdiv = 4;
  static const unsigned int vrelu = 5;
  static const unsigned int vnegate = 6;

  ac_int<2, false> sOp;
  // static const unsigned int nop = 0;
  static const unsigned int sreduce = 1;
  static const unsigned int smax = 2;
  static const unsigned int smin = 3;

  static const unsigned int width = (1 * 4) + (3 * 4) + 2;

  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    m &src0;
    m &src1;
    m &src2;
    m &src3;
    m &vOp0;
    m &vOp1;
    m &vOp2;
    m &vOp3;
    m &sOp;
  }
};
