#pragma once

#ifndef NO_SYSC
#include <mc_connections.h>
#endif

#include "ArchitectureParams.h"

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
    stride = 1;

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

    input_dtype = 0;
    weight_dtype = 0;

    use_input_codebook = false;
    use_weight_codebook = false;

    for (int i = 0; i < NUM_CODEBOOK_ENTRIES; i++) {
      input_code[i] = 0;
      weight_code[i] = 0;
    }

    head_size_power_of_two = 0;

    has_bias = false;
    has_input_transpose = false;
    has_weight_transpose = false;
    is_replication = false;
    has_attn_output_permute = false;
    is_mx_op = false;
  }
#endif

  ac_int<ADDRESS_WIDTH, false> INPUT_OFFSET;
  ac_int<ADDRESS_WIDTH, false> INPUT_SCALE_OFFSET;
  ac_int<ADDRESS_WIDTH, false> WEIGHT_OFFSET;
  ac_int<ADDRESS_WIDTH, false> WEIGHT_SCALE_OFFSET;
  ac_int<ADDRESS_WIDTH, false> BIAS_OFFSET;

  // systolic array loop
  ac_int<10, false> loops[2][6];
  ac_int<3, false> inputXLoopIndex[2];
  ac_int<3, false> inputYLoopIndex[2];
  ac_int<3, false> reductionLoopIndex[2];
  ac_int<3, false> weightLoopIndex[2];
  ac_int<3, false> fxIndex;
  ac_int<3, false> fyIndex;
  ac_int<3, false> weightReuseIndex[2];
  ac_int<2, false> stride;

  // weight address generator loop
  ac_int<10, false> weightAddressGenLoops[2][5];
  // in the inner loop, there are actually 2 reduction loops: the
  // standard reduction loop and the reduction that is parallelized in
  // the systolic array
  ac_int<3, false> weightAddressGenReductionLoopIndex[3];
  ac_int<3, false> weightAddressGenWeightLoopIndex[2];
  ac_int<3, false> weightAddressGenFxIndex;
  ac_int<3, false> weightAddressGenFyIndex;

  ac_int<DTYPE_INDEX_WIDTH, false> input_dtype;
  ac_int<DTYPE_INDEX_WIDTH, false> weight_dtype;

  bool use_input_codebook;
  bool use_weight_codebook;

  ac_int<DECODED_INPUT_DTYPE_WIDTH, false> input_code[NUM_CODEBOOK_ENTRIES];
  ac_int<DECODED_WEIGHT_DTYPE_WIDTH, false> weight_code[NUM_CODEBOOK_ENTRIES];

  ac_int<8, false> head_size_power_of_two;

  bool has_bias;
  bool has_input_transpose;
  bool has_weight_transpose;
  bool is_replication;
  bool has_attn_output_permute;
  bool is_mx_op;

  static const unsigned int base_width =
      5 * 64 /* OFFSETS */ + (12 + 10) * 10 /* Loops */ +
      19 * 3 /* Loop indices */ + 2 /* stride */ + 8 /* Head Size */ +
      8 * 1 /* Bools */;

  static const unsigned int extra_width =
      2 * DTYPE_INDEX_WIDTH + NUM_CODEBOOK_ENTRIES * DECODED_INPUT_DTYPE_WIDTH +
      NUM_CODEBOOK_ENTRIES * DECODED_WEIGHT_DTYPE_WIDTH;

  static const unsigned int width = base_width + extra_width;

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
    m & stride;

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

    m & input_dtype;
    m & weight_dtype;

    m & use_input_codebook;
    m & use_weight_codebook;

    for (int i = 0; i < NUM_CODEBOOK_ENTRIES; i++) {
      m& input_code[i];
    }

    for (int i = 0; i < NUM_CODEBOOK_ENTRIES; i++) {
      m& weight_code[i];
    }

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
    os << "stride: " << params.stride << std::endl;

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

    os << "input_dtype: " << params.input_dtype << std::endl;
    os << "weight_dtype: " << params.weight_dtype << std::endl;

    os << "use_input_codebook: " << params.use_input_codebook << std::endl;
    os << "use_weight_codebook: " << params.use_weight_codebook << std::endl;

    for (int i = 0; i < NUM_CODEBOOK_ENTRIES; i++) {
      os << "input_code[" << i << "]: " << params.input_code[i] << std::endl;
    }

    for (int i = 0; i < NUM_CODEBOOK_ENTRIES; i++) {
      os << "weight_code[" << i << "]: " << params.weight_code[i] << std::endl;
    }

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
    if ((lhs.weightAddressGenFxIndex != rhs.weightAddressGenFxIndex ||
         lhs.weightAddressGenFyIndex != rhs.weightAddressGenFyIndex))
      return false;
    if (lhs.stride != rhs.stride) return false;

    if (lhs.input_dtype != rhs.input_dtype) return false;
    if (lhs.weight_dtype != rhs.weight_dtype) return false;

    for (int i = 0; i < NUM_CODEBOOK_ENTRIES; i++) {
      if (lhs.input_code[i] != rhs.input_code[i]) return false;
      if (lhs.weight_code[i] != rhs.weight_code[i]) return false;
    }

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
   * op_type determines if it's for vector or reduction unit
   *
   * Vector Unit Pipeline Configuration
   * Reduce Unit Configuration
   */

#ifndef __SYNTHESIS__
  VectorInstructions() {
    op_type = 0;
    vector_op0_src0 = 0;
    vector_op0_src1 = 0;
    vector_op2_src1 = 0;
    vector_op3_src1 = 0;
    vector_op0 = 0;
    vector_op1 = 0;
    vector_op2 = 0;
    vector_op3 = 0;
    vdest = 0;
    reduce_count = 0;
    reduce_op = 0;
    rsqrt = 0;
    rreciprocal = 0;
    rduplicate = 0;
    rbroadcast = 0;
    rdest = 0;
    immediate0 = 0;
    immediate1 = 0;
    immediate2 = 0;
    VMAP_OFFSET = 0;
  }
#endif

  ac_int<2, false> op_type;
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
  static const unsigned int from_immediate_2 = 10;

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
  static const unsigned int vsquare = 3;

  ac_int<2, false> vector_op3;
  static const unsigned int vdiv = 1;
  static const unsigned int vquantize_mx = 2;

  ac_int<10, false> reduce_count;
  ac_int<3, false> reduce_op;
  static const unsigned int radd = 1;
  static const unsigned int rmax = 2;

  ac_int<1, false> rsqrt;
  ac_int<1, false> rreciprocal;
  ac_int<1, false> rduplicate;
  ac_int<1, false> rbroadcast;  // broadcast by immediate0

  ac_int<1, false> rdest;
  static const unsigned int to_op0 = 0;
  static const unsigned int to_op2 = 1;

  ac_int<2, false> vdest;
  static const unsigned int to_output = 1;
  static const unsigned int to_reduce = 2;
  static const unsigned int to_accumulate = 3;

  ac_int<16, false> immediate0;
  ac_int<16, false> immediate1;
  ac_int<16, false> immediate2;
  ac_int<64, false> VMAP_OFFSET;

  static const unsigned int width = 176;

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m & op_type;
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
    m & reduce_count;
    m & reduce_op;
    m & rsqrt;
    m & rreciprocal;
    m & rduplicate;
    m & rbroadcast;
    m & rdest;
    m & vdest;
    m & immediate0;
    m & immediate1;
    m & immediate2;
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
    os << "op_type: " << params.op_type << std::endl;
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
    os << "reduce_count: " << params.reduce_count << std::endl;
    os << "reduce_op: " << params.reduce_op << std::endl;
    os << "rsqrt: " << params.rsqrt << std::endl;
    os << "rreciprocal: " << params.rreciprocal << std::endl;
    os << "rduplicate: " << params.rduplicate << std::endl;
    os << "rbroadcast: " << params.rbroadcast << std::endl;
    os << "rdest: " << params.rdest << std::endl;
    os << "vdest: " << params.vdest << std::endl;
    os << "immediate0: " << params.immediate0 << std::endl;
    os << "immediate1: " << params.immediate1 << std::endl;
    os << "immediate2: " << params.immediate2 << std::endl;
    os << "VMAP_OFFSET: " << params.VMAP_OFFSET << std::endl;
    return os;
  }

  inline friend bool operator==(const VectorInstructions& lhs,
                                const VectorInstructions& rhs) {
    return lhs.op_type == rhs.op_type &&
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
           lhs.reduce_count == rhs.reduce_count &&
           lhs.reduce_op == rhs.reduce_op && lhs.rsqrt == rhs.rsqrt &&
           lhs.rreciprocal == rhs.rreciprocal &&
           lhs.rduplicate == rhs.rduplicate &&
           lhs.rbroadcast == rhs.rbroadcast && lhs.rdest == rhs.rdest &&
           lhs.vdest == rhs.vdest && lhs.immediate0 == rhs.immediate0 &&
           lhs.immediate1 == rhs.immediate1 &&
           lhs.immediate2 == rhs.immediate2 &&
           lhs.VMAP_OFFSET == rhs.VMAP_OFFSET;
  }
};

struct VectorParams : BaseParams {
#ifndef __SYNTHESIS__
  VectorParams() {
    addr_gen0_mode = 0;
    ADDRESS_GEN0_OFFSET = 0;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        addr_gen0_loops[i][j] = 0;
      }
    }
    for (int i = 0; i < 2; i++) {
      addr_gen0_y_loop_idx[i] = 0;
      addr_gen0_x_loop_idx[i] = 1;
      addr_gen0_k_loop_idx[i] = 2;
    }
    addr_gen0_dq_scale = 0;
    addr_gen0_dtype = 0;

    addr_gen1_mode = 0;
    ADDRESS_GEN1_OFFSET = 0;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        addr_gen1_loops[i][j] = 0;
      }
    }
    for (int i = 0; i < 2; i++) {
      addr_gen1_y_loop_idx[i] = 0;
      addr_gen1_x_loop_idx[i] = 1;
      addr_gen1_k_loop_idx[i] = 2;
    }
    addr_gen1_dq_scale = 0;
    addr_gen1_dtype = 0;

    addr_gen2_mode = 0;
    ADDRESS_GEN2_OFFSET = 0;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        addr_gen2_loops[i][j] = 0;
      }
    }
    for (int i = 0; i < 2; i++) {
      addr_gen2_y_loop_idx[i] = 0;
      addr_gen2_x_loop_idx[i] = 1;
      addr_gen2_k_loop_idx[i] = 2;
    }
    addr_gen2_dq_scale = 0;
    addr_gen2_dtype = 0;

    output_mode = 1;
    VECTOR_OUTPUT_OFFSET = 0;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        output_loops[i][j] = 0;
      }
    }
    for (int i = 0; i < 2; i++) {
      output_y_loop_idx[i] = 0;
      output_x_loop_idx[i] = 1;
      output_k_loop_idx[i] = 2;
    }
    output_dtype = 0;

    addr_gen0_broadcast = 0;
    addr_gen1_broadcast = 0;
    addr_gen2_broadcast = 0;

    has_slicing = false;
    addr_gen0_dim = 0;
    addr_gen0_start = 0;
    addr_gen0_end = 0;
    addr_gen0_step = 0;

    has_permute = false;
    for (int i = 0; i < 6; i++) {
      addr_gen0_dims[i] = i;
    }

    has_transpose = false;

    head_size_power_of_two = 32;
    has_attn_head_permute = false;
    has_output_permute = false;

    quantize_output_mx = false;
    SCALE_OFFSET = 0;
    use_output_codebook = false;
    for (int i = 0; i < NUM_CODEBOOK_ENTRIES - 1; i++) {
      output_code[i] = 0;
    }
  }
#endif

  // Address generator 0 (vector input)
  ac_int<2, false> addr_gen0_mode;
  ac_int<ADDRESS_WIDTH, false> ADDRESS_GEN0_OFFSET;
  ac_int<11, false> addr_gen0_loops[2][3];
  ac_int<3, false> addr_gen0_x_loop_idx[2];
  ac_int<3, false> addr_gen0_y_loop_idx[2];
  ac_int<3, false> addr_gen0_k_loop_idx[2];
  ac_int<16, false> addr_gen0_dq_scale;
  ac_int<4, false> addr_gen0_dtype;

  // Address generator 1 (op0src1)
  ac_int<2, false> addr_gen1_mode;
  ac_int<ADDRESS_WIDTH, false> ADDRESS_GEN1_OFFSET;
  ac_int<11, false> addr_gen1_loops[2][3];
  ac_int<3, false> addr_gen1_x_loop_idx[2];
  ac_int<3, false> addr_gen1_y_loop_idx[2];
  ac_int<3, false> addr_gen1_k_loop_idx[2];
  ac_int<16, false> addr_gen1_dq_scale;
  ac_int<4, false> addr_gen1_dtype;

  // Address generator 2 (op3src1)
  ac_int<2, false> addr_gen2_mode;
  ac_int<ADDRESS_WIDTH, false> ADDRESS_GEN2_OFFSET;
  ac_int<11, false> addr_gen2_loops[2][3];
  ac_int<3, false> addr_gen2_x_loop_idx[2];
  ac_int<3, false> addr_gen2_y_loop_idx[2];
  ac_int<3, false> addr_gen2_k_loop_idx[2];
  ac_int<16, false> addr_gen2_dq_scale;
  ac_int<4, false> addr_gen2_dtype;

  // Output address generator
  ac_int<2, false> output_mode;
  ac_int<ADDRESS_WIDTH, false> VECTOR_OUTPUT_OFFSET;
  ac_int<11, false> output_loops[2][3];
  ac_int<3, false> output_x_loop_idx[2];
  ac_int<3, false> output_y_loop_idx[2];
  ac_int<3, false> output_k_loop_idx[2];
  ac_int<4, false> output_dtype;

  ac_int<6, false> addr_gen0_broadcast;
  ac_int<3, false> addr_gen1_broadcast;
  ac_int<3, false> addr_gen2_broadcast;

  // Address generator 0 slicing and reshape
  bool has_slicing;
  ac_int<3, false> addr_gen0_dim;
  ac_int<11, false> addr_gen0_start;
  ac_int<11, false> addr_gen0_end;
  ac_int<11, false> addr_gen0_step;

  bool has_permute;
  ac_int<3, false> addr_gen0_dims[6];

  bool has_transpose;

  // Transformer head permutation
  ac_int<4, false> head_size_power_of_two;
  bool has_attn_head_permute;
  bool has_output_permute;

  bool quantize_output_mx;
  ac_int<ADDRESS_WIDTH, false> SCALE_OFFSET;

  bool use_output_codebook;
  ac_int<MAX_DECODED_DTYPE_WIDTH + 1, true>
      output_code[NUM_CODEBOOK_ENTRIES - 1];

  // Each address generator has a 2-bit mode flag, 64-bit address, 6 11-bit loop
  // boundaries, 6 3-bit loop indices, a 16-bit dequantize scale, and a 2-bit
  // data type
  static const unsigned int address_gen_width =
      2 + ADDRESS_WIDTH + 6 * 11 + 6 * 3 + 16 + 4;

  static const unsigned int codebook_params_width =
      1 + (NUM_CODEBOOK_ENTRIES - 1) * (MAX_DECODED_DTYPE_WIDTH + 1);

  // There are 4 address generators in total + 12-bit broadcasting flag + 36-bit
  // slicing params + 18-bit reshape params + 4-bit head size + 6 boolean flags
  // + 64-bit scale offset
  static const unsigned int width = 4 * address_gen_width + 12 + 36 + 18 + 4 +
                                    6 + ADDRESS_WIDTH - 16 +
                                    codebook_params_width;

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    // Address generator 0
    m & addr_gen0_mode;
    m & ADDRESS_GEN0_OFFSET;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        m& addr_gen0_loops[i][j];
      }
    }
    for (int i = 0; i < 2; i++) {
      m& addr_gen0_x_loop_idx[i];
    }
    for (int i = 0; i < 2; i++) {
      m& addr_gen0_y_loop_idx[i];
    }
    for (int i = 0; i < 2; i++) {
      m& addr_gen0_k_loop_idx[i];
    }
    m & addr_gen0_dq_scale;
    m & addr_gen0_dtype;

    // Address generator 1
    m & addr_gen1_mode;
    m & ADDRESS_GEN1_OFFSET;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        m& addr_gen1_loops[i][j];
      }
    }
    for (int i = 0; i < 2; i++) {
      m& addr_gen1_x_loop_idx[i];
    }
    for (int i = 0; i < 2; i++) {
      m& addr_gen1_y_loop_idx[i];
    }
    for (int i = 0; i < 2; i++) {
      m& addr_gen1_k_loop_idx[i];
    }
    m & addr_gen1_dq_scale;
    m & addr_gen1_dtype;

    // Address generator 2
    m & addr_gen2_mode;
    m & ADDRESS_GEN2_OFFSET;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        m& addr_gen2_loops[i][j];
      }
    }
    for (int i = 0; i < 2; i++) {
      m& addr_gen2_x_loop_idx[i];
    }
    for (int i = 0; i < 2; i++) {
      m& addr_gen2_y_loop_idx[i];
    }
    for (int i = 0; i < 2; i++) {
      m& addr_gen2_k_loop_idx[i];
    }
    m & addr_gen2_dq_scale;
    m & addr_gen2_dtype;

    // Output address generator
    m & output_mode;
    m & VECTOR_OUTPUT_OFFSET;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        m& output_loops[i][j];
      }
    }
    for (int i = 0; i < 2; i++) {
      m& output_x_loop_idx[i];
    }
    for (int i = 0; i < 2; i++) {
      m& output_y_loop_idx[i];
    }
    for (int i = 0; i < 2; i++) {
      m& output_k_loop_idx[i];
    }
    m & output_dtype;

    m & addr_gen0_broadcast;
    m & addr_gen1_broadcast;
    m & addr_gen2_broadcast;

    // Slicing and reshape
    m & has_slicing;
    m & addr_gen0_dim;
    m & addr_gen0_start;
    m & addr_gen0_end;
    m & addr_gen0_step;

    m & has_permute;
    for (int i = 0; i < 6; i++) {
      m& addr_gen0_dims[i];
    }

    m & has_transpose;

    // Transformer head permutation flags
    m & head_size_power_of_two;
    m & has_attn_head_permute;
    m & has_output_permute;

    m & quantize_output_mx;
    m & SCALE_OFFSET;

    m & use_output_codebook;
    for (int i = 0; i < NUM_CODEBOOK_ENTRIES - 1; i++) {
      m& output_code[i];
    }
  }

  inline friend void sc_trace(sc_trace_file* tf, const VectorParams& params,
                              const std::string& name) {
    // TODO
  }
#endif

  inline friend std::ostream& operator<<(ostream& os,
                                         const VectorParams& params) {
    os << "addr_gen0_mode: " << params.addr_gen0_mode << std::endl;
    os << "ADDRESS_GEN0_OFFSET: " << params.ADDRESS_GEN0_OFFSET << std::endl;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        os << "addr_gen0_loops[" << i << "][" << j
           << "]: " << params.addr_gen0_loops[i][j] << std::endl;
      }
    }
    for (int i = 0; i < 2; i++) {
      os << "addr_gen0_y_loop_idx[" << i
         << "]: " << params.addr_gen0_y_loop_idx[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "addr_gen0_x_loop_idx[" << i
         << "]: " << params.addr_gen0_x_loop_idx[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "addr_gen0_k_loop_idx[" << i
         << "]: " << params.addr_gen0_k_loop_idx[i] << std::endl;
    }
    os << "addr_gen0_dq_scale: " << params.addr_gen0_dq_scale << std::endl;
    os << "addr_gen0_dtype: " << params.addr_gen0_dtype << std::endl;

    os << "addr_gen1_mode: " << params.addr_gen1_mode << std::endl;
    os << "ADDRESS_GEN1_OFFSET: " << params.ADDRESS_GEN1_OFFSET << std::endl;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        os << "addr_gen1_loops[" << i << "][" << j
           << "]: " << params.addr_gen1_loops[i][j] << std::endl;
      }
    }
    for (int i = 0; i < 2; i++) {
      os << "addr_gen1_y_loop_idx[" << i
         << "]: " << params.addr_gen1_y_loop_idx[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "addr_gen1_x_loop_idx[" << i
         << "]: " << params.addr_gen1_x_loop_idx[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "addr_gen1_k_loop_idx[" << i
         << "]: " << params.addr_gen1_k_loop_idx[i] << std::endl;
    }
    os << "addr_gen1_dq_scale: " << params.addr_gen1_dq_scale << std::endl;
    os << "addr_gen1_dtype: " << params.addr_gen1_dtype << std::endl;

    os << "addr_gen2_mode: " << params.addr_gen2_mode << std::endl;
    os << "ADDRESS_GEN2_OFFSET: " << params.ADDRESS_GEN2_OFFSET << std::endl;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        os << "addr_gen2_loops[" << i << "][" << j
           << "]: " << params.addr_gen2_loops[i][j] << std::endl;
      }
    }
    for (int i = 0; i < 2; i++) {
      os << "addr_gen2_y_loop_idx[" << i
         << "]: " << params.addr_gen2_y_loop_idx[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "addr_gen2_x_loop_idx[" << i
         << "]: " << params.addr_gen2_x_loop_idx[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "addr_gen2_k_loop_idx[" << i
         << "]: " << params.addr_gen2_k_loop_idx[i] << std::endl;
    }
    os << "addr_gen2_dq_scale: " << params.addr_gen2_dq_scale << std::endl;
    os << "addr_gen2_dtype: " << params.addr_gen2_dtype << std::endl;

    os << "output_mode: " << params.output_mode << std::endl;
    os << "VECTOR_OUTPUT_OFFSET: " << params.VECTOR_OUTPUT_OFFSET << std::endl;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        os << "output_loops[" << i << "][" << j
           << "]: " << params.output_loops[i][j] << std::endl;
      }
    }
    for (int i = 0; i < 2; i++) {
      os << "output_y_loop_idx[" << i << "]: " << params.output_y_loop_idx[i]
         << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "output_x_loop_idx[" << i << "]: " << params.output_x_loop_idx[i]
         << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "output_k_loop_idx[" << i << "]: " << params.output_k_loop_idx[i]
         << std::endl;
    }
    os << "output_dtype: " << params.output_dtype << std::endl;

    os << "addr_gen0_broadcast: " << params.addr_gen0_broadcast << std::endl;
    os << "addr_gen1_broadcast: " << params.addr_gen1_broadcast << std::endl;
    os << "addr_gen2_broadcast: " << params.addr_gen2_broadcast << std::endl;

    os << "has_slicing: " << params.has_slicing << std::endl;
    os << "addr_gen0_dim: " << params.addr_gen0_dim << std::endl;
    os << "addr_gen0_start: " << params.addr_gen0_start << std::endl;
    os << "addr_gen0_end: " << params.addr_gen0_end << std::endl;
    os << "addr_gen0_step: " << params.addr_gen0_step << std::endl;

    os << "has_permute: " << params.has_permute << std::endl;
    for (int i = 0; i < 6; i++) {
      os << "addr_gen0_dims[" << i << "]: " << params.addr_gen0_dims[i]
         << std::endl;
    }

    os << "has_transpose: " << params.has_transpose << std::endl;

    os << "head_size_power_of_two: " << params.head_size_power_of_two
       << std::endl;
    os << "has_attn_head_permute: " << params.has_attn_head_permute
       << std::endl;
    os << "has_output_permute: " << params.has_output_permute << std::endl;

    os << "quantize_output_mx: " << params.quantize_output_mx << std::endl;
    os << "SCALE_OFFSET: " << params.SCALE_OFFSET << std::endl;

    os << "use_output_codebook: " << params.use_output_codebook << std::endl;
    for (int i = 0; i < NUM_CODEBOOK_ENTRIES - 1; i++) {
      os << "output_code[" << i << "]: " << params.output_code[i] << std::endl;
    }

    return os;
  }

  inline friend bool operator==(const VectorParams& lhs,
                                const VectorParams& rhs) {
    // Compare Address Gen 0 members
    if (lhs.addr_gen0_mode != rhs.addr_gen0_mode) return false;
    if (lhs.ADDRESS_GEN0_OFFSET != rhs.ADDRESS_GEN0_OFFSET) return false;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        if (lhs.addr_gen0_loops[i][j] != rhs.addr_gen0_loops[i][j])
          return false;
      }
    }
    for (int i = 0; i < 2; i++) {
      if (lhs.addr_gen0_x_loop_idx[i] != rhs.addr_gen0_x_loop_idx[i])
        return false;
      if (lhs.addr_gen0_y_loop_idx[i] != rhs.addr_gen0_y_loop_idx[i])
        return false;
      if (lhs.addr_gen0_k_loop_idx[i] != rhs.addr_gen0_k_loop_idx[i])
        return false;
    }
    if (lhs.addr_gen0_dq_scale != rhs.addr_gen0_dq_scale) return false;
    if (lhs.addr_gen0_dtype != rhs.addr_gen0_dtype) return false;

    // Compare Address Gen 1 members
    if (lhs.addr_gen1_mode != rhs.addr_gen1_mode) return false;
    if (lhs.ADDRESS_GEN1_OFFSET != rhs.ADDRESS_GEN1_OFFSET) return false;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        if (lhs.addr_gen1_loops[i][j] != rhs.addr_gen1_loops[i][j])
          return false;
      }
    }
    for (int i = 0; i < 2; i++) {
      if (lhs.addr_gen1_x_loop_idx[i] != rhs.addr_gen1_x_loop_idx[i])
        return false;
      if (lhs.addr_gen1_y_loop_idx[i] != rhs.addr_gen1_y_loop_idx[i])
        return false;
      if (lhs.addr_gen1_k_loop_idx[i] != rhs.addr_gen1_k_loop_idx[i])
        return false;
    }
    if (lhs.addr_gen1_dq_scale != rhs.addr_gen1_dq_scale) return false;
    if (lhs.addr_gen1_dtype != rhs.addr_gen1_dtype) return false;

    // Compare Address Gen 2 members
    if (lhs.addr_gen2_mode != rhs.addr_gen2_mode) return false;
    if (lhs.ADDRESS_GEN2_OFFSET != rhs.ADDRESS_GEN2_OFFSET) return false;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        if (lhs.addr_gen2_loops[i][j] != rhs.addr_gen2_loops[i][j])
          return false;
      }
    }
    for (int i = 0; i < 2; i++) {
      if (lhs.addr_gen2_x_loop_idx[i] != rhs.addr_gen2_x_loop_idx[i])
        return false;
      if (lhs.addr_gen2_y_loop_idx[i] != rhs.addr_gen2_y_loop_idx[i])
        return false;
      if (lhs.addr_gen2_k_loop_idx[i] != rhs.addr_gen2_k_loop_idx[i])
        return false;
    }
    if (lhs.addr_gen2_dq_scale != rhs.addr_gen2_dq_scale) return false;
    if (lhs.addr_gen2_dtype != rhs.addr_gen2_dtype) return false;

    // Compare output and other members
    if (lhs.output_mode != rhs.output_mode) return false;
    if (lhs.VECTOR_OUTPUT_OFFSET != rhs.VECTOR_OUTPUT_OFFSET) return false;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        if (lhs.output_loops[i][j] != rhs.output_loops[i][j]) return false;
      }
    }
    for (int i = 0; i < 2; i++) {
      if (lhs.output_x_loop_idx[i] != rhs.output_x_loop_idx[i]) return false;
      if (lhs.output_y_loop_idx[i] != rhs.output_y_loop_idx[i]) return false;
      if (lhs.output_k_loop_idx[i] != rhs.output_k_loop_idx[i]) return false;
    }
    if (lhs.output_dtype != rhs.output_dtype) return false;

    if (lhs.addr_gen0_broadcast != rhs.addr_gen0_broadcast) return false;
    if (lhs.addr_gen1_broadcast != rhs.addr_gen1_broadcast) return false;
    if (lhs.addr_gen2_broadcast != rhs.addr_gen2_broadcast) return false;

    if (lhs.has_slicing != rhs.has_slicing) return false;
    if (lhs.addr_gen0_dim != rhs.addr_gen0_dim) return false;
    if (lhs.addr_gen0_start != rhs.addr_gen0_start) return false;
    if (lhs.addr_gen0_end != rhs.addr_gen0_end) return false;
    if (lhs.addr_gen0_step != rhs.addr_gen0_step) return false;

    if (lhs.has_permute != rhs.has_permute) return false;
    for (int i = 0; i < 6; i++) {
      if (lhs.addr_gen0_dims[i] != rhs.addr_gen0_dims[i]) return false;
    }

    if (lhs.has_transpose != rhs.has_transpose) return false;

    if (lhs.head_size_power_of_two != rhs.head_size_power_of_two) return false;
    if (lhs.has_attn_head_permute != rhs.has_attn_head_permute) return false;
    if (lhs.has_output_permute != rhs.has_output_permute) return false;

    if (lhs.quantize_output_mx != rhs.quantize_output_mx) return false;
    if (lhs.SCALE_OFFSET != rhs.SCALE_OFFSET) return false;
    if (lhs.use_output_codebook != rhs.use_output_codebook) return false;
    for (int i = 0; i < NUM_CODEBOOK_ENTRIES - 1; i++) {
      if (lhs.output_code[i] != rhs.output_code[i]) return false;
    }

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
      m& inst[j].op_type;
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
      m& inst[j].reduce_count;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].reduce_op;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].rsqrt;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].rreciprocal;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].rduplicate;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].rbroadcast;
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
      m& inst[j].immediate2;
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
