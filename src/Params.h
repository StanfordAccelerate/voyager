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
#ifndef __SYNTHESIS__
  MatrixParams() {
    INPUT_OFFSET = 0;
    WEIGHT_OFFSET = 0;

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 6; j++) {
        loops[i][j] = 0;
      }
    }
    for (int i = 0; i < 2; i++) {
      inputXLoopIndex[i] = 0;
      inputYLoopIndex[i] = 0;
      reductionLoopIndex[i] = 0;
      weightLoopIndex[i] = 0;
      weightReuseIndex[i] = 0;
    }
    fxIndex = 0;
    fyIndex = 0;

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 5; j++) {
        weightAddressGenLoops[i][j] = 0;
      }
    }
    for (int i = 0; i < 2; i++) {
      weightAddressGenReductionLoopIndex[i] = 0;
      weightAddressGenWeightLoopIndex[i] = 0;
    }
    weightAddressGenFxIndex = 0;
    weightAddressGenFyIndex = 0;

    STRIDE = 1;

    WEIGHT_TRANSPOSE = false;
    REPLICATION = false;

    STORE_IN_ACC = false;
    ACC_FROM_ACC = false;
    CONCAT_INPUT = false;
    CONCAT_HEAD_WEIGHTS = false;
    TRANPOSE_INPUTS = false;
  }
#endif

  int INPUT_OFFSET;
  int WEIGHT_OFFSET;

  // systolic array loop
  ac_int<10, false> loops[2][6];
  ac_int<3, false> inputXLoopIndex[2];
  ac_int<3, false> inputYLoopIndex[2];
  ac_int<3, false> reductionLoopIndex[2];
  ac_int<3, false> weightLoopIndex[2];
  ac_int<3, false> fxIndex;
  ac_int<3, false> fyIndex;
  ac_int<3, false> weightReuseIndex[2];

  // weight address generator loop
  ac_int<10, false> weightAddressGenLoops[2][5];
  // in the inner loop, there are actually 2 reduction loops: the
  // standard reduction loop and the reduction that is parallelized in
  // the systolic array
  ac_int<3, false> weightAddressGenReductionLoopIndex[2];
  ac_int<3, false> weightAddressGenWeightLoopIndex[2];
  ac_int<3, false> weightAddressGenFxIndex;
  ac_int<3, false> weightAddressGenFyIndex;

  ac_int<2, false> STRIDE;

  bool WEIGHT_TRANSPOSE;
  bool REPLICATION;

  bool STORE_IN_ACC;
  bool ACC_FROM_ACC;
  bool CONCAT_INPUT;
  bool CONCAT_HEAD_WEIGHTS;
  bool TRANPOSE_INPUTS;

  bool BIAS;
  int BIAS_OFFSET;

  static const unsigned int width =
      3 * 32 /* OFFSETS */ + (12 + 10) * 10 /* Loops */ +
      (6 + 3) * 2 * 3 /* Loop indices */ + 8 * 1 /* Bools */ + 2;

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m & INPUT_OFFSET;
    m & WEIGHT_OFFSET;
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
    m & fxIndex;
    m & fyIndex;
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
    m & weightAddressGenFxIndex;
    m & weightAddressGenFyIndex;
    m & STRIDE;
    m & WEIGHT_TRANSPOSE;
    m & REPLICATION;
    m & STORE_IN_ACC;
    m & ACC_FROM_ACC;
    m & CONCAT_INPUT;
    m & CONCAT_HEAD_WEIGHTS;
    m & TRANPOSE_INPUTS;
    m & BIAS;
    m & BIAS_OFFSET;
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
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 5; j++) {
        os << "weightAddressGenLoops[" << i << "][" << j
           << "]: " << params.weightAddressGenLoops[i][j] << std::endl;
      }
    }
    for (int i = 0; i < 2; i++) {
      os << "weightAddressGenReductionLoopIndex[" << i
         << "]: " << params.weightAddressGenReductionLoopIndex[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "weightAddressGenWeightLoopIndex[" << i
         << "]: " << params.weightAddressGenWeightLoopIndex[i] << std::endl;
    }
    os << "weightAddressGenFxIndex: " << params.weightAddressGenFxIndex
       << std::endl;
    os << "weightAddressGenFyIndex: " << params.weightAddressGenFyIndex
       << std::endl;
    os << "STRIDE: " << params.STRIDE << std::endl;
    os << "REPLICATION: " << params.REPLICATION << std::endl;
    return os;
  }

  inline friend bool operator==(const MatrixParams& lhs,
                                const MatrixParams& rhs) {
    if (lhs.INPUT_OFFSET != rhs.INPUT_OFFSET ||
        lhs.WEIGHT_OFFSET != rhs.WEIGHT_OFFSET)
      return false;

    // Compare the 2D arrays
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 6; j++) {
        if (lhs.loops[i][j] != rhs.loops[i][j]) return false;
      }
    }

    for (int i = 0; i < 2; i++) {
      if (lhs.inputXLoopIndex[i] != rhs.inputXLoopIndex[i]) return false;
      if (lhs.inputYLoopIndex[i] != rhs.inputYLoopIndex[i]) return false;
      if (lhs.reductionLoopIndex[i] != rhs.reductionLoopIndex[i]) return false;
      if (lhs.weightLoopIndex[i] != rhs.weightLoopIndex[i]) return false;
      if (lhs.weightReuseIndex[i] != rhs.weightReuseIndex[i]) return false;
      if (lhs.weightAddressGenReductionLoopIndex[i] !=
          rhs.weightAddressGenReductionLoopIndex[i])
        return false;
      if (lhs.weightAddressGenWeightLoopIndex[i] !=
          rhs.weightAddressGenWeightLoopIndex[i])
        return false;
    }

    // Compare other members
    if (lhs.fxIndex != rhs.fxIndex || lhs.fyIndex != rhs.fyIndex) return false;
    if (lhs.weightAddressGenFxIndex != rhs.weightAddressGenFxIndex ||
        lhs.weightAddressGenFyIndex != rhs.weightAddressGenFyIndex)
      return false;
    if (lhs.STRIDE != rhs.STRIDE) return false;

    // Compare boolean values
    if (lhs.WEIGHT_TRANSPOSE != rhs.WEIGHT_TRANSPOSE) return false;
    if (lhs.REPLICATION != rhs.REPLICATION) return false;
    if (lhs.STORE_IN_ACC != rhs.STORE_IN_ACC) return false;
    if (lhs.ACC_FROM_ACC != rhs.ACC_FROM_ACC) return false;
    if (lhs.CONCAT_INPUT != rhs.CONCAT_INPUT) return false;
    if (lhs.CONCAT_HEAD_WEIGHTS != rhs.CONCAT_HEAD_WEIGHTS) return false;
    if (lhs.TRANPOSE_INPUTS != rhs.TRANPOSE_INPUTS) return false;

    if (lhs.BIAS != rhs.BIAS || lhs.BIAS_OFFSET != rhs.BIAS_OFFSET)
      return false;

    // If all members are equal, return true
    return true;
  }
};

// TODO: this should be parameterized on VECTOR_DATATYPE
struct VectorInstructions {
  /*
   * Vector Instruction
   * instType determines if it's for vector or reduction unit
   *
   * Vector Unit Pipeline Configuration
   * Reduce Unit Configuration
   */

#ifndef __SYNTHESIS__
  VectorInstructions() {
    instType = 0;
    vInput = 0;
    vOp0Src1 = 0;
    vOp0 = 0;
    vOp1 = 0;
    vOp1Src1 = 0;
    vOp2 = 0;
    vOp3Src1 = 0;
    vOp3 = 0;
    vOp4 = 0;
    vAccumulatePush = 0;
    vDest = 0;
    rCount = 0;
    rOp = 0;
    rSqrt = 0;
    rReciprocal = 0;
    rMax1 = 0;
    rDuplicate = 0;
    rDest = 0;
    rBroadcast = 0;
    immediate0 = 0;
    immediate1 = 0;
  }
#endif

  ac_int<2, false> instType;
  static const unsigned int vector = 0;
  static const unsigned int reduction = 1;
  static const unsigned int accumulation = 2;

  ac_int<3, false> vInput;
  static const unsigned int readFromSystolicArray = 1;
  static const unsigned int readFromVectorFetch = 2;
  static const unsigned int readFromAccumulation = 3;
  static const unsigned int readFromReduce = 4;

  ac_int<1, false> vDequantize;
  ac_int<16, false> vDequantizeScale;

  // src0 refers to lhs and src1 refers to rhs

  // Stage 0: add, mult
  ac_int<3, false> vOp0Src1;
  static const unsigned int readInterface = 1;
  static const unsigned int op0immediate = 2;
  // static const unsigned int readFromReduce = 4;

  ac_int<2, false> vOp0;  // add, sub, mult
  static const unsigned int nop = 0;
  static const unsigned int vadd = 1;
  static const unsigned int vmult = 2;
  static const unsigned int vsub = 3;

  // Stage 1: exp
  ac_int<2, false> vOp1;
  static const unsigned int vexp = 1;
  static const unsigned int vscaleexp = 2;

  ac_int<1, false> vOp1Src1;
  static const unsigned int op1immediate = 0;

  // Stage 2: send to reduce unit
  ac_int<1, false> vOp2;
  static const unsigned int toReduce = 1;

  // Stage 3: add, div
  ac_int<3, false> vOp3Src1;  // don't read, read from reduce interface 2, or
                              // normal interface
  static const unsigned int readReduceInterface = 1;
  static const unsigned int readNormalInterface = 2;
  static const unsigned int op3immediate = 3;
  ac_int<3, false> vOp3;  // add, mult
  // static const unsigned int vadd = 1;
  // static const unsigned int vmult = 2;
  static const unsigned int vsquare = 3;

  // Stage 4: relu
  ac_int<2, false> vOp4;
  static const unsigned int vrelu = 1;
  static const unsigned int vrelumask = 2;

  // Stage 5: quantize
  ac_int<1, false> vOp5;
  static const unsigned int vquantize = 1;

  ac_int<1, false> vAccumulatePush;

  // Vector Unit write out
  ac_int<1, false> vDest;
  static const unsigned int vWriteOut = 1;

  ac_int<10, false> rCount;
  ac_int<2, false> rOp;
  static const unsigned int radd = 1;
  static const unsigned int rmax = 2;

  ac_int<1, false> rSqrt;
  ac_int<1, false> rReciprocal;
  ac_int<1, false> rMax1;
  ac_int<1, false> rDuplicate;

  ac_int<3, false> rDest;
  static const unsigned int toVectorOp0Src0 = 1;
  static const unsigned int toVectorOp0Src1 = 2;
  static const unsigned int toVectorOp3Src1 = 3;
  static const unsigned int sWriteOut = 4;

  // broadcast count comes from {immediate1,immediate0}
  ac_int<1, false> rBroadcast;

  ac_int<16, false> immediate0;
  ac_int<16, false> immediate1;

  static const unsigned int width = 94;

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m & instType;
    m & vInput;
    m & vDequantize;
    m & vDequantizeScale;
    m & vOp0Src1;
    m & vOp0;
    m & vOp1;
    m & vOp1Src1;
    m & vOp2;
    m & vOp3Src1;
    m & vOp3;
    m & vOp4;
    m & vOp5;
    m & vAccumulatePush;
    m & vDest;
    m & rCount;
    m & rOp;
    m & rSqrt;
    m & rReciprocal;
    m & rMax1;
    m & rDuplicate;
    m & rDest;
    m & rBroadcast;
    m & immediate0;
    m & immediate1;
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

  inline friend bool operator==(const VectorInstructions& lhs,
                                const VectorInstructions& rhs) {
    if (lhs.instType != rhs.instType || lhs.vInput != rhs.vInput ||
        lhs.vDequantize != rhs.vDequantize || lhs.vOp0Src1 != rhs.vOp0Src1 ||
        lhs.vOp0 != rhs.vOp0 || lhs.vOp1 != rhs.vOp1 ||
        lhs.vOp1Src1 != rhs.vOp1Src1 || lhs.vOp2 != rhs.vOp2 ||
        lhs.vOp3Src1 != rhs.vOp3Src1 || lhs.vOp3 != rhs.vOp3 ||
        lhs.vOp4 != rhs.vOp4 || lhs.vOp5 != rhs.vOp5 ||
        lhs.vAccumulatePush != rhs.vAccumulatePush || lhs.vDest != rhs.vDest ||
        lhs.rCount != rhs.rCount || lhs.rOp != rhs.rOp ||
        lhs.rSqrt != rhs.rSqrt || lhs.rReciprocal != rhs.rReciprocal ||
        lhs.rMax1 != rhs.rMax1 || lhs.rDuplicate != rhs.rDuplicate ||
        lhs.rDest != rhs.rDest || lhs.rBroadcast != rhs.rBroadcast ||
        lhs.immediate0 != rhs.immediate0 || lhs.immediate1 != rhs.immediate1)
      return false;

    return true;
  }
};

struct VectorParams : BaseParams {
  // 3 address generators:
  // - Vector Input
  // - Residual/Op0Src1
  // - Bias/Op3Src1

#ifndef __SYNTHESIS__
  VectorParams() {
    VECTOR_OFFSET = 0;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        addressGen0Loop[i][j] = 0;
      }
    }
    DP_VEC0 = false;

    ADDRESS_GEN1_OFFSET = 0;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        addressGen1Loops[i][j] = 0;
      }
    }
    for (int i = 0; i < 2; i++) {
      addressGen1InputXLoopIndex[i] = 0;
      addressGen1InputYLoopIndex[i] = 0;
      addressGen1WeightLoopIndex[i] = 0;
    }
    DP_VEC1 = false;

    ADDRESS_GEN2_OFFSET = 0;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        addressGen2Loops[i][j] = 0;
      }
    }
    for (int i = 0; i < 2; i++) {
      addressGen2InputXLoopIndex[i] = 0;
      addressGen2InputYLoopIndex[i] = 0;
      addressGen2WeightLoopIndex[i] = 0;
    }
    DP_VEC2 = false;

    VECTOR_OUTPUT_OFFSET = 0;
    SCALAR_OUTPUT_OFFSET = 0;

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        outputLoops[i][j] = 0;
      }
    }
    for (int i = 0; i < 2; i++) {
      outputXLoopIndex[i] = 0;
      outputYLoopIndex[i] = 0;
      outputWeightLoopIndex[i] = 0;
    }
    SPLIT_OUTPUT = false;

    DP_OUTPUT = false;

    addressGen0Mode = 0;
    addressGen0Broadcast = false;
    addressGen0BroadcastCount = 0;
    addressGen1Mode = 0;
    addressGen2Mode = 0;
    MAXPOOL = false;
    AVGPOOL = false;
  }
#endif

  // Address Gen 0 (vector input)
  int VECTOR_OFFSET;
  ac_int<11, false> addressGen0Loop[2][3];  // tiled 2d tensor
  ac_int<3, false> addressGen0InputXLoopIndex[2];
  ac_int<3, false> addressGen0InputYLoopIndex[2];
  ac_int<3, false> addressGen0WeightLoopIndex[2];
  bool DP_VEC0;
  ac_int<16, false> vec0DequantizeScale;

  // Address Gen 1 (residual/op0src1)
  int ADDRESS_GEN1_OFFSET;
  ac_int<11, false> addressGen1Loops[2][3];
  ac_int<3, false> addressGen1InputXLoopIndex[2];
  ac_int<3, false> addressGen1InputYLoopIndex[2];
  ac_int<3, false> addressGen1WeightLoopIndex[2];
  bool DP_VEC1;
  ac_int<16, false> vec1DequantizeScale;

  // Address Gen 2 (bias/op3src1)
  int ADDRESS_GEN2_OFFSET;
  ac_int<11, false> addressGen2Loops[2][3];
  ac_int<3, false> addressGen2InputXLoopIndex[2];
  ac_int<3, false> addressGen2InputYLoopIndex[2];
  ac_int<3, false> addressGen2WeightLoopIndex[2];
  bool DP_VEC2;
  ac_int<16, false> vec2DequantizeScale;

  int VECTOR_OUTPUT_OFFSET;
  int SCALAR_OUTPUT_OFFSET;

  ac_int<11, false> outputLoops[2][3];
  ac_int<3, false> outputXLoopIndex[2];
  ac_int<3, false> outputYLoopIndex[2];
  ac_int<3, false> outputWeightLoopIndex[2];
  bool SPLIT_OUTPUT;
  bool CONCAT_OUTPUT;

  bool DP_OUTPUT;
  ac_int<16, false> outputQuantizeScale;

  // 1: 3d-tensor, 2: 2d-tensor, 3: 1d-tensor
  ac_int<2, false> addressGen0Mode;
  bool addressGen0Broadcast;
  ac_int<10, false> addressGen0BroadcastCount;
  ac_int<2, false> addressGen1Mode;
  ac_int<2, false> addressGen2Mode;
  bool MAXPOOL;
  bool AVGPOOL;

  static const unsigned int width =
      5 * 32 /* OFFSETS */ + 4 * 6 * 11 /* Loops */ +
      3 * 6 * 4 /* Loop indices */ + 9 * 1 /* Bools */ + 10 + 3 * 2 +
      16 * 4 /* Dequantize scale */;

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m & VECTOR_OFFSET;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        m& addressGen0Loop[i][j];
      }
    }
    for (int i = 0; i < 2; i++) {
      m& addressGen0InputXLoopIndex[i];
    }
    for (int i = 0; i < 2; i++) {
      m& addressGen0InputYLoopIndex[i];
    }
    for (int i = 0; i < 2; i++) {
      m& addressGen0WeightLoopIndex[i];
    }
    m & DP_VEC0;
    m & vec0DequantizeScale;
    m & ADDRESS_GEN1_OFFSET;
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
    m & DP_VEC1;
    m & vec1DequantizeScale;
    m & ADDRESS_GEN2_OFFSET;

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
    m & DP_VEC2;
    m & vec2DequantizeScale;
    m & VECTOR_OUTPUT_OFFSET;
    m & SCALAR_OUTPUT_OFFSET;
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
    m & SPLIT_OUTPUT;
    m & CONCAT_OUTPUT;
    m & DP_OUTPUT;
    m & outputQuantizeScale;
    m & addressGen0Mode;
    m & addressGen0Broadcast;
    m & addressGen0BroadcastCount;
    m & addressGen1Mode;
    m & addressGen2Mode;
    m & MAXPOOL;
    m & AVGPOOL;
  }

  inline friend void sc_trace(sc_trace_file* tf, const VectorParams& params,
                              const std::string& name) {
    // TODO
  }
#endif

  inline friend std::ostream& operator<<(ostream& os,
                                         const VectorParams& params) {
    os << "VECTOR_OFFSET: " << params.VECTOR_OFFSET << std::endl;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        os << "addressGen0Loop[" << i << "][" << j
           << "]: " << params.addressGen0Loop[i][j] << std::endl;
      }
    }
    for (int i = 0; i < 2; i++) {
      os << "addressGen0InputXLoopIndex[" << i
         << "]: " << params.addressGen0InputXLoopIndex[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "addressGen0InputYLoopIndex[" << i
         << "]: " << params.addressGen0InputYLoopIndex[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "addressGen0WeightLoopIndex[" << i
         << "]: " << params.addressGen0WeightLoopIndex[i] << std::endl;
    }
    os << "DP_VEC0: " << params.DP_VEC0 << std::endl;
    os << "ADDRESS_GEN1_OFFSET: " << params.ADDRESS_GEN1_OFFSET << std::endl;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        os << "addressGen1Loops[" << i << "][" << j
           << "]: " << params.addressGen1Loops[i][j] << std::endl;
      }
    }
    for (int i = 0; i < 2; i++) {
      os << "addressGen1InputXLoopIndex[" << i
         << "]: " << params.addressGen1InputXLoopIndex[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "addressGen1InputYLoopIndex[" << i
         << "]: " << params.addressGen1InputYLoopIndex[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "addressGen1WeightLoopIndex[" << i
         << "]: " << params.addressGen1WeightLoopIndex[i] << std::endl;
    }
    os << "DP_VEC1: " << params.DP_VEC1 << std::endl;
    os << "ADDRESS_GEN2_OFFSET: " << params.ADDRESS_GEN2_OFFSET << std::endl;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        os << "addressGen2Loops[" << i << "][" << j
           << "]: " << params.addressGen2Loops[i][j] << std::endl;
      }
    }
    for (int i = 0; i < 2; i++) {
      os << "addressGen2InputXLoopIndex[" << i
         << "]: " << params.addressGen2InputXLoopIndex[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "addressGen2InputYLoopIndex[" << i
         << "]: " << params.addressGen2InputYLoopIndex[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "addressGen2WeightLoopIndex[" << i
         << "]: " << params.addressGen2WeightLoopIndex[i] << std::endl;
    }
    os << "DP_VEC2: " << params.DP_VEC2 << std::endl;
    os << "VECTOR_OUTPUT_OFFSET: " << params.VECTOR_OUTPUT_OFFSET << std::endl;
    os << "SCALAR_OUTPUT_OFFSET: " << params.SCALAR_OUTPUT_OFFSET << std::endl;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        os << "outputLoops[" << i << "][" << j
           << "]: " << params.outputLoops[i][j] << std::endl;
      }
    }
    for (int i = 0; i < 2; i++) {
      os << "outputXLoopIndex[" << i << "]: " << params.outputXLoopIndex[i]
         << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "outputYLoopIndex[" << i << "]: " << params.outputYLoopIndex[i]
         << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "outputWeightLoopIndex[" << i
         << "]: " << params.outputWeightLoopIndex[i] << std::endl;
    }
    os << "SPLIT_OUTPUT: " << params.SPLIT_OUTPUT << std::endl;
    os << "DP_OUTPUT: " << params.DP_OUTPUT << std::endl;
    os << "addressGen0Mode: " << params.addressGen0Mode << std::endl;
    os << "addressGen0Broadcast: " << params.addressGen0Broadcast << std::endl;
    os << "addressGen0BroadcastCount: " << params.addressGen0BroadcastCount
       << std::endl;
    os << "addressGen1Mode: " << params.addressGen1Mode << std::endl;
    os << "addressGen2Mode: " << params.addressGen2Mode << std::endl;
    os << "MAXPOOL: " << params.MAXPOOL << std::endl;
    os << "AVGPOOL: " << params.AVGPOOL << std::endl;
    return os;
  }

  inline friend bool operator==(const VectorParams& lhs,
                                const VectorParams& rhs) {
    // Compare Address Gen 0 members
    if (lhs.VECTOR_OFFSET != rhs.VECTOR_OFFSET) return false;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        if (lhs.addressGen0Loop[i][j] != rhs.addressGen0Loop[i][j])
          return false;
      }
    }
    for (int i = 0; i < 2; i++) {
      if (lhs.addressGen0InputXLoopIndex[i] !=
          rhs.addressGen0InputXLoopIndex[i])
        return false;
      if (lhs.addressGen0InputYLoopIndex[i] !=
          rhs.addressGen0InputYLoopIndex[i])
        return false;
      if (lhs.addressGen0WeightLoopIndex[i] !=
          rhs.addressGen0WeightLoopIndex[i])
        return false;
    }
    if (lhs.DP_VEC0 != rhs.DP_VEC0) return false;
    if (lhs.vec0DequantizeScale != rhs.vec0DequantizeScale) return false;

    // Compare Address Gen 1 members
    if (lhs.ADDRESS_GEN1_OFFSET != rhs.ADDRESS_GEN1_OFFSET) return false;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        if (lhs.addressGen1Loops[i][j] != rhs.addressGen1Loops[i][j])
          return false;
      }
    }
    for (int i = 0; i < 2; i++) {
      if (lhs.addressGen1InputXLoopIndex[i] !=
          rhs.addressGen1InputXLoopIndex[i])
        return false;
      if (lhs.addressGen1InputYLoopIndex[i] !=
          rhs.addressGen1InputYLoopIndex[i])
        return false;
      if (lhs.addressGen1WeightLoopIndex[i] !=
          rhs.addressGen1WeightLoopIndex[i])
        return false;
    }
    if (lhs.DP_VEC1 != rhs.DP_VEC1) return false;
    if (lhs.vec1DequantizeScale != rhs.vec1DequantizeScale) return false;

    // Compare Address Gen 2 members
    if (lhs.ADDRESS_GEN2_OFFSET != rhs.ADDRESS_GEN2_OFFSET) return false;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        if (lhs.addressGen2Loops[i][j] != rhs.addressGen2Loops[i][j])
          return false;
      }
    }
    for (int i = 0; i < 2; i++) {
      if (lhs.addressGen2InputXLoopIndex[i] !=
          rhs.addressGen2InputXLoopIndex[i])
        return false;
      if (lhs.addressGen2InputYLoopIndex[i] !=
          rhs.addressGen2InputYLoopIndex[i])
        return false;
      if (lhs.addressGen2WeightLoopIndex[i] !=
          rhs.addressGen2WeightLoopIndex[i])
        return false;
    }
    if (lhs.DP_VEC2 != rhs.DP_VEC2) return false;
    if (lhs.vec2DequantizeScale != rhs.vec2DequantizeScale) return false;

    // Compare output and other members
    if (lhs.VECTOR_OUTPUT_OFFSET != rhs.VECTOR_OUTPUT_OFFSET) return false;
    if (lhs.SCALAR_OUTPUT_OFFSET != rhs.SCALAR_OUTPUT_OFFSET) return false;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        if (lhs.outputLoops[i][j] != rhs.outputLoops[i][j]) return false;
      }
    }
    for (int i = 0; i < 2; i++) {
      if (lhs.outputXLoopIndex[i] != rhs.outputXLoopIndex[i]) return false;
      if (lhs.outputYLoopIndex[i] != rhs.outputYLoopIndex[i]) return false;
      if (lhs.outputWeightLoopIndex[i] != rhs.outputWeightLoopIndex[i])
        return false;
    }
    if (lhs.SPLIT_OUTPUT != rhs.SPLIT_OUTPUT) return false;
    if (lhs.CONCAT_OUTPUT != rhs.CONCAT_OUTPUT) return false;
    if (lhs.DP_OUTPUT != rhs.DP_OUTPUT) return false;
    if (lhs.outputQuantizeScale != rhs.outputQuantizeScale) return false;

    // Compare address generation modes and pooling settings
    if (lhs.addressGen0Mode != rhs.addressGen0Mode) return false;
    if (lhs.addressGen0Broadcast != rhs.addressGen0Broadcast) return false;
    if (lhs.addressGen0BroadcastCount != rhs.addressGen0BroadcastCount)
      return false;
    if (lhs.addressGen1Mode != rhs.addressGen1Mode) return false;
    if (lhs.addressGen2Mode != rhs.addressGen2Mode) return false;
    if (lhs.MAXPOOL != rhs.MAXPOOL) return false;
    if (lhs.AVGPOOL != rhs.AVGPOOL) return false;

    // If all members are equal, return true
    return true;
  }
};

struct VectorInstructionConfig : BaseParams {
#ifndef __SYNTHESIS__
  VectorInstructionConfig() {
    for (int i = 0; i < 8; i++) {
      instCount[i] = 0;
    }
    instLen = 0;
    instLoopCount = 0;
  }
#endif

  VectorInstructions inst[8];
  ac_int<20, false> instCount[8];
  ac_int<3, false> instLen;
  ac_int<16, false> instLoopCount;

  static const unsigned int width =
      VectorInstructions::width * 8 + 20 * 8 + 3 + 16;

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
      m& inst[j].vDequantize;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].vDequantizeScale;
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
      m& inst[j].vOp1Src1;
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
      m& inst[j].vOp5;
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
      m& inst[j].rSqrt;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].rReciprocal;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].rMax1;
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
    m & instLen;
    m & instLoopCount;
  }

  inline friend void sc_trace(sc_trace_file* tf,
                              const VectorInstructionConfig& params,
                              const std::string& name) {
    // TODO
  }
#endif

  inline friend std::ostream& operator<<(
      ostream& os, const VectorInstructionConfig& params) {
    for (int i = 0; i < 8; i++) {
      os << "instIndex: " << i << std::endl;
      os << "instCount: " << params.instCount[i] << std::endl;
      os << params.inst[i] << std::endl;
    }
    os << "instLen: " << params.instLen << std::endl;
    os << "instLoopCount: " << params.instLoopCount << std::endl;
    return os;
  }

  inline friend bool operator==(const VectorInstructionConfig& lhs,
                                const VectorInstructionConfig& rhs) {
    for (int i = 0; i < 8; i++) {
      if (lhs.instCount[i] != rhs.instCount[i]) return false;
      if (!(lhs.inst[i] == rhs.inst[i])) return false;
    }
    if (lhs.instLen != rhs.instLen || lhs.instLoopCount != rhs.instLoopCount)
      return false;

    return true;
  }
};
