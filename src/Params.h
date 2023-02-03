#pragma once

#ifndef NO_SYSC
#include "TypeToBits.h"
#endif

// Base params struct
struct BaseParams {
  // empty, only purpose is to serve as a base class for polymorphism
  // We need this or any other virtual member to make Base polymorphic
  virtual ~BaseParams() {}
};

struct MatrixParams : BaseParams {
  int INPUT_OFFSET;
  int WEIGHT_OFFSET;

  // systolic array loop
  int loops[2][6];
  int inputXLoopIndex[2];
  int inputYLoopIndex[2];
  int reductionLoopIndex[2];
  int weightLoopIndex[2];
  int fxIndex;
  int fyIndex;
  int weightReuseIndex[2];

  // weight address generator loop
  int weightAddressGenLoops[2][5];
  // in the inner loop, there are actually 2 reduction loops: the
  // standard reduction loop and the reduction that is parallelized in
  // the systolic array
  int weightAddressGenReductionLoopIndex[2];
  int weightAddressGenWeightLoopIndex[2];
  int weightAddressGenFxIndex;
  int weightAddressGenFyIndex;
  int weightAddressGenInputXLoopIndex;  // only care about outer X and Y loops
  int weightAddressGenInputYLoopIndex;

  int STRIDE;
  int HEAD_SIZE_LG2;

  bool WEIGHT_TRANSPOSE;
  bool REPLICATION;

  bool STORE_IN_ACC;
  bool ACC_FROM_ACC;
  bool CONCAT_INPUT;
  bool CONCAT_HEAD_WEIGHTS;
  bool TRANPOSE_INPUTS;

  int GRAD_OFFSET;
  bool COMBINE_GRADS;
  ac_int<8, false> learningRate;

  static const unsigned int width =
      13 * 32 + 12 * 32 + 10 * 32 + 7 * 1 + 11 * 32 + 32 + 1 + 8;

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m& INPUT_OFFSET;
    m& WEIGHT_OFFSET;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 6; j++) {
        m& loops[i][j];
      }
    }
    for (int i = 0; i < 2; i++) {
      m& inputXLoopIndex[i];
    }
    for (int i = 0; i < 2; i++) {
      m& inputYLoopIndex[i];
    }
    for (int i = 0; i < 2; i++) {
      m& reductionLoopIndex[i];
    }
    for (int i = 0; i < 2; i++) {
      m& weightLoopIndex[i];
    }
    m& fxIndex;
    m& fyIndex;
    for (int i = 0; i < 2; i++) {
      m& weightReuseIndex[i];
    }
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 5; j++) {
        m& weightAddressGenLoops[i][j];
      }
    }
    for (int i = 0; i < 2; i++) {
      m& weightAddressGenReductionLoopIndex[i];
    }
    for (int i = 0; i < 2; i++) {
      m& weightAddressGenWeightLoopIndex[i];
    }
    m& weightAddressGenFxIndex;
    m& weightAddressGenFyIndex;
    m& weightAddressGenInputXLoopIndex;
    m& weightAddressGenInputYLoopIndex;
    m& STRIDE;
    m& HEAD_SIZE_LG2;
    m& WEIGHT_TRANSPOSE;
    m& REPLICATION;
    m& STORE_IN_ACC;
    m& ACC_FROM_ACC;
    m& CONCAT_INPUT;
    m& CONCAT_HEAD_WEIGHTS;
    m& TRANPOSE_INPUTS;
    m& GRAD_OFFSET;
    m& COMBINE_GRADS;
    m& learningRate;
  }

  inline friend void sc_trace(sc_trace_file* tf, const MatrixParams& params,
                              const std::string& name) {
    // TODO
  }
#endif

  inline friend std::ostream& operator<<(ostream& os,
                                         const MatrixParams& params) {
    os << "INPUT_OFFSET: " << params.INPUT_OFFSET << std::endl;
    os << "WEIGHT_OFFSET: " << params.WEIGHT_OFFSET << std::endl;
    os << "WEIGHT_TRANSPOSE: " << params.WEIGHT_TRANSPOSE << std::endl;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 6; j++) {
        os << "loops[" << i << "][" << j << "]: " << params.loops[i][j]
           << std::endl;
      }
    }
    for (int i = 0; i < 2; i++) {
      os << "inputXLoopIndex[" << i << "]: " << params.inputXLoopIndex[i]
         << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "inputYLoopIndex[" << i << "]: " << params.inputYLoopIndex[i]
         << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "reductionLoopIndex[" << i << "]: " << params.reductionLoopIndex[i]
         << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "weightLoopIndex[" << i << "]: " << params.weightLoopIndex[i]
         << std::endl;
    }
    os << "fxIndex: " << params.fxIndex << std::endl;
    os << "fyIndex: " << params.fyIndex << std::endl;
    for (int i = 0; i < 2; i++) {
      os << "weightReuseIndex[" << i << "]: " << params.weightReuseIndex[i]
         << std::endl;
    }
    os << "STRIDE: " << params.STRIDE << std::endl;
    os << "REPLICATION: " << params.REPLICATION << std::endl;

    return os;
  }
};

struct VectorInstructions {
  /*
   * Vector Instruction
   * instType determines if it's for vector or reduction unit
   *
   * Vector Unit Pipeline Configuration
   * Reduce Unit Configuration
   */

  ac_int<2, false> instType;
  static const unsigned int vector = 0;
  static const unsigned int reduction = 1;
  static const unsigned int accumulation = 2;

  ac_int<3, false> vInput;
  static const unsigned int readFromSystolicArray = 1;
  static const unsigned int readFromVectorFetch = 2;
  static const unsigned int readFromAccumulation = 3;
  static const unsigned int readFromReduce = 4;

  // src0 refers to lhs and src1 refers to rhs

  // Stage 0: add, mult
  ac_int<3, false> vOp0Src1;
  static const unsigned int readInterface = 1;
  static const unsigned int op0immediate0 = 2;
  static const unsigned int op0immediate1 = 3;
  // static const unsigned int readFromReduce = 4;

  ac_int<2, false> vOp0;  // add, sub, mult
  static const unsigned int nop = 0;
  static const unsigned int vadd = 1;
  static const unsigned int vmult = 2;
  static const unsigned int vsub = 3;

  // Stage 1: exp
  ac_int<1, false> vOp1;
  static const unsigned int vexp = 1;

  // Stage 2: send to reduce unit
  ac_int<1, false> vOp2;
  static const unsigned int toReduce = 1;

  // Stage 3: add, div
  ac_int<3, false> vOp3Src1;  // don't read, read from reduce interface 2, or
                              // normal interface
  static const unsigned int readReduceInterface = 1;
  static const unsigned int readNormalInterface = 2;
  static const unsigned int op3immediate0 = 3;
  static const unsigned int op3immediate1 = 4;
  ac_int<3, false> vOp3;  // add, div
  // static const unsigned int vadd = 1;
  // static const unsigned int vmult = 2;
  static const unsigned int vdiv = 3;
  static const unsigned int vsquare = 4;
  static const unsigned int vscaleexp = 5;

  // Stage 4: relu
  ac_int<2, false> vOp4;
  static const unsigned int vrelu = 1;
  static const unsigned int vrelumask = 2;

  ac_int<1, false> vAccumulatePush;

  // Vector Unit write out
  ac_int<1, false> vDest;
  static const unsigned int vWriteOut = 1;

  ac_int<10, false> rCount;
  ac_int<2, false> rOp;
  static const unsigned int radd = 1;
  static const unsigned int rmax = 2;

  ac_int<1, false> rInvSqrt;
  ac_int<1, false> rDuplicate;

  ac_int<3, false> rDest;
  static const unsigned int toVectorOp0Src0 = 1;
  static const unsigned int toVectorOp0Src1 = 2;
  static const unsigned int toVectorOp3Src1 = 3;
  static const unsigned int sWriteOut = 4;

  // broadcast count comes from {immediate1,immediate0}
  ac_int<1, false> rBroadcast;

  ac_int<8, false> immediate0;
  ac_int<8, false> immediate1;

  static const unsigned int width = 56;
  VectorInstructions() {}

#ifndef NO_SYSC
  VectorInstructions(const int a) {
    ac_int<width, false> val = a;
    sc_lv<width> valLV;
    type_to_vector(val, width, valLV);
    *this = BitsToType<VectorInstructions>(valLV);
  }

  explicit operator int() const {
    ac_int<32, false> val;
    vector_to_type(TypeToBits<VectorInstructions>(*this), false, &val);

    int a = val;
    return a;
  }

  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m& instType;
    m& vInput;
    m& vOp0Src1;
    m& vOp0;
    m& vOp1;
    m& vOp2;
    m& vOp3Src1;
    m& vOp3;
    m& vOp4;
    m& vAccumulatePush;
    m& vDest;
    m& rCount;
    m& rOp;
    m& rInvSqrt;
    m& rDuplicate;
    m& rDest;
    m& rBroadcast;
    m& immediate0;
    m& immediate1;
  }

  inline friend void sc_trace(sc_trace_file* tf,
                              const VectorInstructions& params,
                              const std::string& name) {
    // TODO
  }
#endif

  inline friend std::ostream& operator<<(ostream& os,
                                         const VectorInstructions& params) {
    os << "instType: " << params.instType << std::endl;
    os << "vInput: " << params.vInput << std::endl;
    os << "vOp0Src1: " << params.vOp0Src1 << std::endl;
    os << "vOp0: " << params.vOp0 << std::endl;
    os << "vOp1: " << params.vOp1 << std::endl;
    os << "vOp2: " << params.vOp2 << std::endl;
    os << "vOp3Src1: " << params.vOp3Src1 << std::endl;
    os << "vOp3: " << params.vOp3 << std::endl;
    os << "vOp4: " << params.vOp4 << std::endl;
    os << "vDest: " << params.vDest << std::endl;
    os << "rCount: " << params.rCount << std::endl;
    os << "rOp: " << params.rOp << std::endl;
    os << "rDest: " << params.rDest << std::endl;
    return os;
  }
};

struct VectorParams : BaseParams {
  // 3 address generators:
  // - Vector Input
  // - Residual/Op0Src1
  // - Bias/Op3Src1

  // Address Gen 0 (vector input)
  int VECTOR_OFFSET;
  int addressGen0Loop[2][3];  // tiled 2d tensor
  bool DP_VEC0;

  // Address Gen 1 (residual/op0src1)
  int ADDRESS_GEN1_OFFSET;
  int addressGen1Loops[2][3];
  int addressGen1InputXLoopIndex[2];
  int addressGen1InputYLoopIndex[2];
  int addressGen1WeightLoopIndex[2];
  bool DP_VEC1;

  // Address Gen 2 (bias/op3src1)
  int ADDRESS_GEN2_OFFSET;
  int addressGen2Loops[2][3];
  int addressGen2InputXLoopIndex[2];
  int addressGen2InputYLoopIndex[2];
  int addressGen2WeightLoopIndex[2];
  bool DP_VEC2;

  int VECTOR_OUTPUT_OFFSET;
  int SCALAR_OUTPUT_OFFSET;

  int scalarOutputCount;

  int outputLoops[2][3];
  int outputXLoopIndex[2];
  int outputYLoopIndex[2];
  int outputWeightLoopIndex[2];
  int FULL_HEAD_SIZE;
  bool SPLIT_OUTPUT;

  bool DP_OUTPUT;

  bool addressGen0Enable;
  bool addressGen0Broadcast;
  int addressGen0BroadcastCount;
  ac_int<2, false> addressGen1Mode;  // 1- residual, 2- 2dtensor
  ac_int<2, false> addressGen2Mode;  // 1- bias, 2- 2dtensor
  bool MAXPOOL;
  bool AVGPOOL;

  static const unsigned int width =
      13 * 32 + 1 + 1 + 2 + 2 + 1 + 1 + 37 * 32 + 2 + 1 + 1 + 1;

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m& VECTOR_OFFSET;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        m& addressGen0Loop[i][j];
      }
    }
    m& DP_VEC0;
    m& ADDRESS_GEN1_OFFSET;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        m& addressGen1Loops[i][j];
      }
    }
    for (int i = 0; i < 2; i++) {
      m& addressGen1InputXLoopIndex[i];
    }
    for (int i = 0; i < 2; i++) {
      m& addressGen1InputYLoopIndex[i];
    }
    for (int i = 0; i < 2; i++) {
      m& addressGen1WeightLoopIndex[i];
    }
    m& DP_VEC1;
    m& ADDRESS_GEN2_OFFSET;

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        m& addressGen2Loops[i][j];
      }
    }
    for (int i = 0; i < 2; i++) {
      m& addressGen2InputXLoopIndex[i];
    }
    for (int i = 0; i < 2; i++) {
      m& addressGen2InputYLoopIndex[i];
    }
    for (int i = 0; i < 2; i++) {
      m& addressGen2WeightLoopIndex[i];
    }
    m& DP_VEC2;
    m& VECTOR_OUTPUT_OFFSET;
    m& SCALAR_OUTPUT_OFFSET;
    m& scalarOutputCount;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        m& outputLoops[i][j];
      }
    }
    for (int i = 0; i < 2; i++) {
      m& outputXLoopIndex[i];
    }
    for (int i = 0; i < 2; i++) {
      m& outputYLoopIndex[i];
    }
    for (int i = 0; i < 2; i++) {
      m& outputWeightLoopIndex[i];
    }
    m& FULL_HEAD_SIZE;
    m& SPLIT_OUTPUT;
    m& DP_OUTPUT;
    m& addressGen0Enable;
    m& addressGen0Broadcast;
    m& addressGen0BroadcastCount;
    m& addressGen1Mode;
    m& addressGen2Mode;
    m& MAXPOOL;
    m& AVGPOOL;
  }

  inline friend void sc_trace(sc_trace_file* tf, const VectorParams& params,
                              const std::string& name) {
    // TODO
  }
#endif

  inline friend std::ostream& operator<<(ostream& os,
                                         const VectorParams& params) {
    // TODO
    return os;
  }
};

struct VectorInstructionConfig : BaseParams {
  VectorInstructions inst[8];
  int instCount[8];
  int instLen;
  int instLoopCount;

  static const unsigned int width =
      VectorInstructions::width * 8 + 32 * 8 + 32 + 32;

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    for (int j = 0; j < 8; j++) {
      m& inst[j].instType;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].vInput;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].vOp0Src1;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].vOp0;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].vOp1;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].vOp2;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].vOp3Src1;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].vOp3;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].vOp4;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].vAccumulatePush;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].vDest;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].rCount;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].rOp;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].rInvSqrt;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].rDuplicate;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].rDest;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].rBroadcast;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].immediate0;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].immediate1;
    }

    for (int i = 0; i < 8; i++) {
      m& instCount[i];
    }
    m& instLen;
    m& instLoopCount;
  }

  inline friend void sc_trace(sc_trace_file* tf,
                              const VectorInstructionConfig& params,
                              const std::string& name) {
    // TODO
  }
#endif

  inline friend std::ostream& operator<<(
      ostream& os, const VectorInstructionConfig& params) {
    // TODO
    return os;
  }
};
