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
    INPUT_SCALE_OFFSET = 0;
    WEIGHT_OFFSET = 0;
    WEIGHT_SCALE_OFFSET = 0;
    BIAS_OFFSET = 0;

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
    for (int i = 0; i < 3; i++) {
      weightAddressGenReductionLoopIndex[i] = 0;
    }
    for (int i = 0; i < 2; i++) {
      weightAddressGenWeightLoopIndex[i] = 0;
    }
    weightAddressGenFxIndex = 0;
    weightAddressGenFyIndex = 0;

    STRIDE = 1;
    head_size_power_of_two = 0;

    has_bias = false;
    has_input_transpose = false;
    has_weight_transpose = false;
    is_replication = false;
    has_attn_output_permute = false;
    is_mx_op = false;
  }
#endif

  uint64_t INPUT_OFFSET;
  uint64_t INPUT_SCALE_OFFSET;
  uint64_t WEIGHT_OFFSET;
  uint64_t WEIGHT_SCALE_OFFSET;
  uint64_t BIAS_OFFSET;

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
  ac_int<3, false> weightAddressGenReductionLoopIndex[3];
  ac_int<3, false> weightAddressGenWeightLoopIndex[2];
  ac_int<3, false> weightAddressGenFxIndex;
  ac_int<3, false> weightAddressGenFyIndex;

  ac_int<2, false> STRIDE;
  ac_int<8, false> head_size_power_of_two;

  bool has_bias;
  bool has_input_transpose;
  bool has_weight_transpose;
  bool is_replication;
  bool has_attn_output_permute;
  bool is_mx_op;

  static const unsigned int width =
      5 * 64 /* OFFSETS */ + (12 + 10) * 10 /* Loops */ +
      19 * 3 /* Loop indices */ + 6 * 1 /* Bools */ + 2 + 8;

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m & INPUT_OFFSET;
    m & INPUT_SCALE_OFFSET;
    m & WEIGHT_OFFSET;
    m & WEIGHT_SCALE_OFFSET;
    m & BIAS_OFFSET;

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
    for (int i = 0; i < 3; i++) {
      m& weightAddressGenReductionLoopIndex[i];
    }
    for (int i = 0; i < 2; i++) {
      m& weightAddressGenWeightLoopIndex[i];
    }
    m & weightAddressGenFxIndex;
    m & weightAddressGenFyIndex;
    m & STRIDE;
    m & head_size_power_of_two;

    m & has_bias;
    m & has_input_transpose;
    m & has_weight_transpose;
    m & is_replication;
    m & has_attn_output_permute;
    m & is_mx_op;
  }

  inline friend void sc_trace(sc_trace_file* tf, const MatrixParams& params,
                              const std::string& name) {
    // TODO
  }
#endif

  inline friend std::ostream& operator<<(ostream& os,
                                         const MatrixParams& params) {
    os << "INPUT_OFFSET: " << params.INPUT_OFFSET << std::endl;
    os << "INPUT_SCALE_OFFSET: " << params.INPUT_SCALE_OFFSET << std::endl;
    os << "WEIGHT_OFFSET: " << params.WEIGHT_OFFSET << std::endl;
    os << "WEIGHT_SCALE_OFFSET: " << params.WEIGHT_SCALE_OFFSET << std::endl;
    os << "BIAS_OFFSET: " << params.BIAS_OFFSET << std::endl;

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
    for (int i = 0; i < 3; i++) {
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
    os << "head_size_power_of_two: " << params.head_size_power_of_two
       << std::endl;

    os << "has_bias: " << params.has_bias << std::endl;
    os << "has_input_transpose: " << params.has_input_transpose << std::endl;
    os << "has_weight_transpose: " << params.has_weight_transpose << std::endl;
    os << "is_replication: " << params.is_replication << std::endl;
    os << "has_attn_output_permute: " << params.has_attn_output_permute
       << std::endl;
    os << "is_mx_op: " << params.is_mx_op << std::endl;
    return os;
  }

  inline friend bool operator==(const MatrixParams& lhs,
                                const MatrixParams& rhs) {
    if (lhs.INPUT_OFFSET != rhs.INPUT_OFFSET ||
        lhs.INPUT_SCALE_OFFSET != rhs.INPUT_SCALE_OFFSET ||
        lhs.WEIGHT_OFFSET != rhs.WEIGHT_OFFSET ||
        lhs.WEIGHT_SCALE_OFFSET != rhs.WEIGHT_SCALE_OFFSET ||
        lhs.BIAS_OFFSET != rhs.BIAS_OFFSET)
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

    if (lhs.weightAddressGenReductionLoopIndex[2] !=
        rhs.weightAddressGenReductionLoopIndex[2])
      return false;

    // Compare other members
    if (lhs.fxIndex != rhs.fxIndex || lhs.fyIndex != rhs.fyIndex) return false;
    if (lhs.weightAddressGenFxIndex != rhs.weightAddressGenFxIndex ||
        lhs.weightAddressGenFyIndex != rhs.weightAddressGenFyIndex)
      return false;
    if (lhs.STRIDE != rhs.STRIDE) return false;
    if (lhs.head_size_power_of_two != rhs.head_size_power_of_two) return false;

    // Compare boolean values
    if (lhs.has_bias != rhs.has_bias || lhs.BIAS_OFFSET != rhs.BIAS_OFFSET)
      return false;
    if (lhs.has_input_transpose != rhs.has_input_transpose) return false;
    if (lhs.has_weight_transpose != rhs.has_weight_transpose) return false;
    if (lhs.is_replication != rhs.is_replication) return false;
    if (lhs.has_attn_output_permute != rhs.has_attn_output_permute)
      return false;
    if (lhs.is_mx_op != rhs.is_mx_op) return false;

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
    vector_op0_src0 = 0;
    vector_op0_src1 = 0;
    vector_op2_src1 = 0;
    vector_op3_src1 = 0;
    vector_op0 = 0;
    vector_op1 = 0;
    vector_op2 = 0;
    vector_op3 = 0;
    vdest = 0;
    rCount = 0;
    rOp = 0;
    rSqrt = 0;
    rReciprocal = 0;
    rMax1 = 0;
    rDuplicate = 0;
    rBroadcast = 0;
    rdest = 0;
    immediate0 = 0;
    immediate1 = 0;
    VMAP_OFFSET = 0;
  }
#endif

  ac_int<2, false> instType;
  static const unsigned int vector = 0;
  static const unsigned int reduction = 1;
  static const unsigned int accumulation = 2;

  ac_int<4, false> vector_op0_src0;
  ac_int<4, false> vector_op0_src1;
  ac_int<4, false> vector_op2_src1;
  ac_int<4, false> vector_op3_src1;

  static const unsigned int from_matrix_unit = 1;
  static const unsigned int from_vector_fetch_0 = 2;
  static const unsigned int from_vector_fetch_1 = 3;
  static const unsigned int from_vector_fetch_2 = 4;
  static const unsigned int from_accumulation = 5;
  static const unsigned int from_reduction_0 = 6;
  static const unsigned int from_reduction_1 = 7;
  static const unsigned int from_immediate_0 = 8;
  static const unsigned int from_immediate_1 = 9;

  ac_int<1, false> vdequantize;
  ac_int<16, false> vector_dq_scale;

  // Stage 0: add, sub, mult
  ac_int<2, false> vector_op0;
  static const unsigned int nop = 0;
  static const unsigned int vadd = 1;
  static const unsigned int vmult = 2;
  static const unsigned int vsub = 3;

  // Stage 1: exp, abs, activations
  ac_int<3, false> vector_op1;
  static const unsigned int vexp = 1;
  static const unsigned int vabs = 2;
  static const unsigned int vrelu = 3;
  static const unsigned int vgelu = 4;
  static const unsigned int vsilu = 5;
  static const unsigned int vmap = 6;

  // Stage 2: add, mult, square, div
  ac_int<2, false> vector_op2;
  // static const unsigned int vadd = 1;
  // static const unsigned int vmult = 2;
  static const unsigned int vsquare = 3;

  ac_int<1, false> vector_op3;
  static const unsigned int vdiv = 1;

  ac_int<10, false> rCount;
  ac_int<3, false> rOp;
  static const unsigned int radd = 1;
  static const unsigned int rmax = 2;

  ac_int<1, false> rSqrt;
  ac_int<1, false> rReciprocal;
  ac_int<1, false> rMax1;
  ac_int<1, false> rDuplicate;
  ac_int<1, false> rBroadcast;  // broadcast by immediate0

  ac_int<1, false> rdest;
  static const unsigned int to_op0 = 0;
  static const unsigned int to_op2 = 1;

  ac_int<2, false> vdest;
  static const unsigned int to_output = 1;
  static const unsigned int to_reduce = 2;
  static const unsigned int to_accumulate = 3;

  ac_int<16, false> immediate0;
  ac_int<16, false> immediate1;
  ac_int<64, false> VMAP_OFFSET;

  static const unsigned int width = 160;

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m & instType;
    m & vector_op0_src0;
    m & vector_op0_src1;
    m & vector_op2_src1;
    m & vector_op3_src1;
    m & vdequantize;
    m & vector_dq_scale;
    m & vector_op0;
    m & vector_op1;
    m & vector_op2;
    m & vector_op3;
    m & rCount;
    m & rOp;
    m & rSqrt;
    m & rReciprocal;
    m & rMax1;
    m & rDuplicate;
    m & rBroadcast;
    m & rdest;
    m & vdest;
    m & immediate0;
    m & immediate1;
    m & VMAP_OFFSET;
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
    os << "vector_op0_src0: " << params.vector_op0_src0 << std::endl;
    os << "vector_op0_src1: " << params.vector_op0_src1 << std::endl;
    os << "vector_op2_src1: " << params.vector_op2_src1 << std::endl;
    os << "vector_op3_src1: " << params.vector_op3_src1 << std::endl;
    os << "vdequantize: " << params.vdequantize << std::endl;
    os << "vector_dq_scale: " << params.vector_dq_scale << std::endl;
    os << "vector_op0: " << params.vector_op0 << std::endl;
    os << "vector_op1: " << params.vector_op1 << std::endl;
    os << "vector_op2: " << params.vector_op2 << std::endl;
    os << "vector_op3: " << params.vector_op3 << std::endl;
    os << "rCount: " << params.rCount << std::endl;
    os << "rOp: " << params.rOp << std::endl;
    os << "rSqrt: " << params.rSqrt << std::endl;
    os << "rReciprocal: " << params.rReciprocal << std::endl;
    os << "rMax1: " << params.rMax1 << std::endl;
    os << "rDuplicate: " << params.rDuplicate << std::endl;
    os << "rBroadcast: " << params.rBroadcast << std::endl;
    os << "rdest: " << params.rdest << std::endl;
    os << "vdest: " << params.vdest << std::endl;
    os << "immediate0: " << params.immediate0 << std::endl;
    os << "immediate1: " << params.immediate1 << std::endl;
    os << "VMAP_OFFSET: " << params.VMAP_OFFSET << std::endl;
    return os;
  }

  inline friend bool operator==(const VectorInstructions& lhs,
                                const VectorInstructions& rhs) {
    return lhs.instType == rhs.instType &&
           lhs.vector_op0_src0 == rhs.vector_op0_src0 &&
           lhs.vector_op0_src1 == rhs.vector_op0_src1 &&
           lhs.vector_op2_src1 == rhs.vector_op2_src1 &&
           lhs.vector_op3_src1 == rhs.vector_op3_src1 &&
           lhs.vector_op0 == rhs.vector_op0 &&
           lhs.vector_op1 == rhs.vector_op1 &&
           lhs.vector_op2 == rhs.vector_op2 &&
           lhs.vector_op3 == rhs.vector_op3 &&
           lhs.vdequantize == rhs.vdequantize &&
           lhs.vector_dq_scale == rhs.vector_dq_scale &&
           lhs.rCount == rhs.rCount && lhs.rOp == rhs.rOp &&
           lhs.rSqrt == rhs.rSqrt && lhs.rReciprocal == rhs.rReciprocal &&
           lhs.rMax1 == rhs.rMax1 && lhs.rDuplicate == rhs.rDuplicate &&
           lhs.rBroadcast == rhs.rBroadcast && lhs.rdest == rhs.rdest &&
           lhs.vdest == rhs.vdest && lhs.immediate0 == rhs.immediate0 &&
           lhs.immediate1 == rhs.immediate1 &&
           lhs.VMAP_OFFSET == rhs.VMAP_OFFSET;
  }
};

struct VectorParams : BaseParams {
#ifndef __SYNTHESIS__
  VectorParams() {
    addressGen0Mode = 0;
    VECTOR_OFFSET = 0;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        addressGen0Loop[i][j] = 0;
      }
    }
    for (int i = 0; i < 2; i++) {
      addressGen0InputYLoopIndex[i] = 0;
      addressGen0InputXLoopIndex[i] = 1;
      addressGen0WeightLoopIndex[i] = 2;
    }
    vec0_dq_scale = 0;
    vector_input_0_type = 0;

    addressGen1Mode = 0;
    ADDRESS_GEN1_OFFSET = 0;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        addressGen1Loops[i][j] = 0;
      }
    }
    for (int i = 0; i < 2; i++) {
      addressGen1InputYLoopIndex[i] = 0;
      addressGen1InputXLoopIndex[i] = 1;
      addressGen1WeightLoopIndex[i] = 2;
    }
    vec1_dq_scale = 0;
    vector_input_1_type = 0;

    addressGen2Mode = 0;
    ADDRESS_GEN2_OFFSET = 0;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        addressGen2Loops[i][j] = 0;
      }
    }
    for (int i = 0; i < 2; i++) {
      addressGen2InputYLoopIndex[i] = 0;
      addressGen2InputXLoopIndex[i] = 1;
      addressGen2WeightLoopIndex[i] = 2;
    }
    vec2_dq_scale = 0;
    vector_input_2_type = 0;

    outputAddressMode = 1;
    VECTOR_OUTPUT_OFFSET = 0;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        outputLoops[i][j] = 0;
      }
    }
    for (int i = 0; i < 2; i++) {
      outputYLoopIndex[i] = 0;
      outputXLoopIndex[i] = 1;
      outputWeightLoopIndex[i] = 2;
    }
    output_scale = 0;
    output_types = 0;

    vec0_broadcast = 0;
    vec1_broadcast = 0;
    vec2_broadcast = 0;

    has_slicing = false;
    vec0_dim = 0;
    vec0_start = 0;
    vec0_end = 0;
    vec0_step = 0;

    has_permute = false;
    for (int i = 0; i < 6; i++) {
      vec0_dim_order[i] = i;
    }

    has_transpose = false;

    head_size_power_of_two = 32;
    has_attn_head_permute = false;
    CONCAT_OUTPUT = false;

    quantize_output_mx = false;
    SCALE_OFFSET = 0;
  }
#endif

  // Address generator 0 (vector input)
  ac_int<2, false> addressGen0Mode;
  uint64_t VECTOR_OFFSET;
  ac_int<11, false> addressGen0Loop[2][3];
  ac_int<3, false> addressGen0InputXLoopIndex[2];
  ac_int<3, false> addressGen0InputYLoopIndex[2];
  ac_int<3, false> addressGen0WeightLoopIndex[2];
  ac_int<16, false> vec0_dq_scale;
  ac_int<2, false> vector_input_0_type;

  // Address generator 1 (op0src1)
  ac_int<2, false> addressGen1Mode;
  uint64_t ADDRESS_GEN1_OFFSET;
  ac_int<11, false> addressGen1Loops[2][3];
  ac_int<3, false> addressGen1InputXLoopIndex[2];
  ac_int<3, false> addressGen1InputYLoopIndex[2];
  ac_int<3, false> addressGen1WeightLoopIndex[2];
  ac_int<16, false> vec1_dq_scale;
  ac_int<2, false> vector_input_1_type;

  // Address generator 2 (op3src1)
  ac_int<2, false> addressGen2Mode;
  uint64_t ADDRESS_GEN2_OFFSET;
  ac_int<11, false> addressGen2Loops[2][3];
  ac_int<3, false> addressGen2InputXLoopIndex[2];
  ac_int<3, false> addressGen2InputYLoopIndex[2];
  ac_int<3, false> addressGen2WeightLoopIndex[2];
  ac_int<16, false> vec2_dq_scale;
  ac_int<2, false> vector_input_2_type;

  // Output address generator
  ac_int<2, false> outputAddressMode;
  uint64_t VECTOR_OUTPUT_OFFSET;
  ac_int<11, false> outputLoops[2][3];
  ac_int<3, false> outputXLoopIndex[2];
  ac_int<3, false> outputYLoopIndex[2];
  ac_int<3, false> outputWeightLoopIndex[2];
  ac_int<16, false> output_scale;
  ac_int<2, false> output_types;

  ac_int<6, false> vec0_broadcast;
  ac_int<3, false> vec1_broadcast;
  ac_int<3, false> vec2_broadcast;

  // Address generator 0 slicing and reshape
  bool has_slicing;
  ac_int<3, false> vec0_dim;
  ac_int<11, false> vec0_start;
  ac_int<11, false> vec0_end;
  ac_int<11, false> vec0_step;

  bool has_permute;
  ac_int<3, false> vec0_dim_order[6];

  bool has_transpose;

  // Transformer head permutation
  ac_int<4, false> head_size_power_of_two;
  bool has_attn_head_permute;
  bool CONCAT_OUTPUT;

  bool quantize_output_mx;
  uint64_t SCALE_OFFSET;

  // Each address generator has a 2-bit mode flag, 64-bit address, 6 11-bit loop
  // boundaries, 6 3-bit loop indices, and a 16-bit dequantize scale
  static const unsigned int address_gen_width =
      2 + 64 + 6 * 11 + 6 * 3 + 16 + 2;

  // There are 4 address generators in total + 12-bit broadcasting flag + 36-bit
  // slicing params + 18-bit reshape params + 4-bit head size + 6 boolean flags
  // + 64-bit scale offset
  static const unsigned int width =
      4 * address_gen_width + 12 + 36 + 18 + 4 + 6 + 64;

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    // Address generator 0
    m & addressGen0Mode;
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
    m & vec0_dq_scale;
    m & vector_input_0_type;

    // Address generator 1
    m & addressGen1Mode;
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
    m & vec1_dq_scale;
    m & vector_input_1_type;

    // Address generator 2
    m & addressGen2Mode;
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
    m & vec2_dq_scale;
    m & vector_input_2_type;

    // Output address generator
    m & outputAddressMode;
    m & VECTOR_OUTPUT_OFFSET;
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
    m & output_scale;
    m & output_types;

    m & vec0_broadcast;
    m & vec1_broadcast;
    m & vec2_broadcast;

    // Slicing and reshape
    m & has_slicing;
    m & vec0_dim;
    m & vec0_start;
    m & vec0_end;
    m & vec0_step;

    m & has_permute;
    for (int i = 0; i < 6; i++) {
      m& vec0_dim_order[i];
    }

    m & has_transpose;

    // Transformer head permutation flags
    m & head_size_power_of_two;
    m & has_attn_head_permute;
    m & CONCAT_OUTPUT;

    m & quantize_output_mx;
    m & SCALE_OFFSET;
  }

  inline friend void sc_trace(sc_trace_file* tf, const VectorParams& params,
                              const std::string& name) {
    // TODO
  }
#endif

  inline friend std::ostream& operator<<(ostream& os,
                                         const VectorParams& params) {
    os << "addressGen0Mode: " << params.addressGen0Mode << std::endl;
    os << "VECTOR_OFFSET: " << params.VECTOR_OFFSET << std::endl;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        os << "addressGen0Loop[" << i << "][" << j
           << "]: " << params.addressGen0Loop[i][j] << std::endl;
      }
    }
    for (int i = 0; i < 2; i++) {
      os << "addressGen0InputYLoopIndex[" << i
         << "]: " << params.addressGen0InputYLoopIndex[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "addressGen0InputXLoopIndex[" << i
         << "]: " << params.addressGen0InputXLoopIndex[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "addressGen0WeightLoopIndex[" << i
         << "]: " << params.addressGen0WeightLoopIndex[i] << std::endl;
    }
    os << "vec0_dq_scale: " << params.vec0_dq_scale << std::endl;
    os << "vector_input_0_type: " << params.vector_input_0_type << std::endl;

    os << "addressGen1Mode: " << params.addressGen1Mode << std::endl;
    os << "ADDRESS_GEN1_OFFSET: " << params.ADDRESS_GEN1_OFFSET << std::endl;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        os << "addressGen1Loops[" << i << "][" << j
           << "]: " << params.addressGen1Loops[i][j] << std::endl;
      }
    }
    for (int i = 0; i < 2; i++) {
      os << "addressGen1InputYLoopIndex[" << i
         << "]: " << params.addressGen1InputYLoopIndex[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "addressGen1InputXLoopIndex[" << i
         << "]: " << params.addressGen1InputXLoopIndex[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "addressGen1WeightLoopIndex[" << i
         << "]: " << params.addressGen1WeightLoopIndex[i] << std::endl;
    }
    os << "vec1_dq_scale: " << params.vec1_dq_scale << std::endl;
    os << "vector_input_1_type: " << params.vector_input_1_type << std::endl;

    os << "addressGen2Mode: " << params.addressGen2Mode << std::endl;
    os << "ADDRESS_GEN2_OFFSET: " << params.ADDRESS_GEN2_OFFSET << std::endl;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        os << "addressGen2Loops[" << i << "][" << j
           << "]: " << params.addressGen2Loops[i][j] << std::endl;
      }
    }
    for (int i = 0; i < 2; i++) {
      os << "addressGen2InputYLoopIndex[" << i
         << "]: " << params.addressGen2InputYLoopIndex[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "addressGen2InputXLoopIndex[" << i
         << "]: " << params.addressGen2InputXLoopIndex[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "addressGen2WeightLoopIndex[" << i
         << "]: " << params.addressGen2WeightLoopIndex[i] << std::endl;
    }
    os << "vec2_dq_scale: " << params.vec2_dq_scale << std::endl;
    os << "vector_input_2_type: " << params.vector_input_2_type << std::endl;

    os << "outputAddressMode: " << params.outputAddressMode << std::endl;
    os << "VECTOR_OUTPUT_OFFSET: " << params.VECTOR_OUTPUT_OFFSET << std::endl;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        os << "outputLoops[" << i << "][" << j
           << "]: " << params.outputLoops[i][j] << std::endl;
      }
    }
    for (int i = 0; i < 2; i++) {
      os << "outputYLoopIndex[" << i << "]: " << params.outputYLoopIndex[i]
         << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "outputXLoopIndex[" << i << "]: " << params.outputXLoopIndex[i]
         << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "outputWeightLoopIndex[" << i
         << "]: " << params.outputWeightLoopIndex[i] << std::endl;
    }
    os << "output_scale: " << params.output_scale << std::endl;
    os << "output_types: " << params.output_types << std::endl;

    os << "vec0_broadcast: " << params.vec0_broadcast << std::endl;
    os << "vec1_broadcast: " << params.vec1_broadcast << std::endl;
    os << "vec2_broadcast: " << params.vec2_broadcast << std::endl;

    os << "has_slicing: " << params.has_slicing << std::endl;
    os << "vec0_dim: " << params.vec0_dim << std::endl;
    os << "vec0_start: " << params.vec0_start << std::endl;
    os << "vec0_end: " << params.vec0_end << std::endl;
    os << "vec0_step: " << params.vec0_step << std::endl;

    os << "has_permute: " << params.has_permute << std::endl;
    for (int i = 0; i < 6; i++) {
      os << "vec0_dim_order[" << i << "]: " << params.vec0_dim_order[i]
         << std::endl;
    }

    os << "has_transpose: " << params.has_transpose << std::endl;

    os << "head_size_power_of_two: " << params.head_size_power_of_two
       << std::endl;
    os << "has_attn_head_permute: " << params.has_attn_head_permute
       << std::endl;
    os << "CONCAT_OUTPUT: " << params.CONCAT_OUTPUT << std::endl;

    os << "quantize_output_mx: " << params.quantize_output_mx << std::endl;
    os << "SCALE_OFFSET: " << params.SCALE_OFFSET << std::endl;

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
    if (lhs.vec0_dq_scale != rhs.vec0_dq_scale) return false;
    if (lhs.vector_input_0_type != rhs.vector_input_0_type) return false;

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
    if (lhs.vec1_dq_scale != rhs.vec1_dq_scale) return false;
    if (lhs.vector_input_1_type != rhs.vector_input_1_type) return false;

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
    if (lhs.vec2_dq_scale != rhs.vec2_dq_scale) return false;
    if (lhs.vector_input_2_type != rhs.vector_input_2_type) return false;

    // Compare output and other members
    if (lhs.VECTOR_OUTPUT_OFFSET != rhs.VECTOR_OUTPUT_OFFSET) return false;
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
    if (lhs.output_scale != rhs.output_scale) return false;
    if (lhs.output_types != rhs.output_types) return false;

    if (lhs.vec0_broadcast != rhs.vec0_broadcast) return false;
    if (lhs.vec1_broadcast != rhs.vec1_broadcast) return false;
    if (lhs.vec2_broadcast != rhs.vec2_broadcast) return false;

    if (lhs.has_slicing != rhs.has_slicing) return false;
    if (lhs.vec0_dim != rhs.vec0_dim) return false;
    if (lhs.vec0_start != rhs.vec0_start) return false;
    if (lhs.vec0_end != rhs.vec0_end) return false;
    if (lhs.vec0_step != rhs.vec0_step) return false;

    if (lhs.has_permute != rhs.has_permute) return false;
    for (int i = 0; i < 6; i++) {
      if (lhs.vec0_dim_order[i] != rhs.vec0_dim_order[i]) return false;
    }

    if (lhs.has_transpose != rhs.has_transpose) return false;

    if (lhs.head_size_power_of_two != rhs.head_size_power_of_two) return false;
    if (lhs.has_attn_head_permute != rhs.has_attn_head_permute) return false;
    if (lhs.CONCAT_OUTPUT != rhs.CONCAT_OUTPUT) return false;

    if (lhs.quantize_output_mx != rhs.quantize_output_mx) return false;

    // Compare address generation modes and pooling settings
    if (lhs.addressGen0Mode != rhs.addressGen0Mode) return false;
    if (lhs.addressGen1Mode != rhs.addressGen1Mode) return false;
    if (lhs.addressGen2Mode != rhs.addressGen2Mode) return false;
    if (lhs.outputAddressMode != rhs.outputAddressMode) return false;

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
      m& inst[j].vector_op0_src0;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].vector_op0_src1;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].vector_op2_src1;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].vector_op3_src1;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].vdequantize;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].vector_dq_scale;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].vector_op0;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].vector_op1;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].vector_op2;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].vector_op3;
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
      m& inst[j].rBroadcast;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].rdest;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].vdest;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].immediate0;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].immediate1;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].VMAP_OFFSET;
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
