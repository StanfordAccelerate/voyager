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
      fyIndex[i] = 0;
    }
    fxIndex = 0;
    stride = 1;
    padding = 0;

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
    for (int i = 0; i < 2; i++) {
      weightAddressGenFyIndex[i] = 0;
    }
    weightAddressGenFxIndex = 0;

    input_dtype = 0;
    use_input_codebook = false;
    input_burst_size = 0;
    input_num_beats = 1;
    input_packing_factor_power = 1;

    weight_dtype = 0;
    use_weight_codebook = false;
    weight_burst_size = 0;
    weight_num_beats = 1;
    weight_packing_factor_power = 1;

    for (int i = 0; i < NUM_CODEBOOK_ENTRIES; i++) {
      input_code[i] = 0;
      weight_code[i] = 0;
    }

    head_size_power_of_two = 0;

    has_bias = false;
    has_input_transpose = false;
    has_weight_transpose = false;

    is_resnet_replication = false;
    is_generic_replication = false;
    num_channels = 0;
    fx_unrolling_lg2 = 0;

    has_attn_output_permute = false;
    is_mx_op = false;
    is_fc = false;
    write_output_to_accum_buffer = false;
  }
#endif

  static constexpr int LOOP_WIDTH = 16;

  ac_int<ADDRESS_WIDTH, false> INPUT_OFFSET;
  ac_int<ADDRESS_WIDTH, false> INPUT_SCALE_OFFSET;
  ac_int<ADDRESS_WIDTH, false> WEIGHT_OFFSET;
  ac_int<ADDRESS_WIDTH, false> WEIGHT_SCALE_OFFSET;
  ac_int<ADDRESS_WIDTH, false> BIAS_OFFSET;

  // systolic array loop
  ac_int<LOOP_WIDTH, false> loops[2][6];
  ac_int<3, false> inputXLoopIndex[2];
  ac_int<3, false> inputYLoopIndex[2];
  ac_int<3, false> reductionLoopIndex[2];
  ac_int<3, false> weightLoopIndex[2];
  ac_int<3, false> fyIndex[2];
  ac_int<3, false> fxIndex;
  ac_int<3, false> weightReuseIndex[2];
  ac_int<5, false> stride;
  ac_int<2, false> padding;

  // weight address generator loop
  ac_int<LOOP_WIDTH, false> weightAddressGenLoops[2][5];
  // in the inner loop, there are actually 2 reduction loops: the
  // standard reduction loop and the reduction that is parallelized in
  // the systolic array
  ac_int<3, false> weightAddressGenReductionLoopIndex[3];
  ac_int<3, false> weightAddressGenWeightLoopIndex[2];
  ac_int<3, false> weightAddressGenFyIndex[2];
  ac_int<3, false> weightAddressGenFxIndex;

  ac_int<DTYPE_INDEX_WIDTH, false> input_dtype;
  bool use_input_codebook;
  ac_int<10, false> input_burst_size;
  ac_int<4, false> input_num_beats;
  ac_int<4, false> input_packing_factor_power;

  ac_int<DTYPE_INDEX_WIDTH, false> weight_dtype;
  bool use_weight_codebook;
  ac_int<10, false> weight_burst_size;
  ac_int<4, false> weight_num_beats;
  ac_int<4, false> weight_packing_factor_power;

  ac_int<DECODED_INPUT_DTYPE_WIDTH, false> input_code[NUM_CODEBOOK_ENTRIES];
  ac_int<DECODED_WEIGHT_DTYPE_WIDTH, false> weight_code[NUM_CODEBOOK_ENTRIES];

  ac_int<8, false> head_size_power_of_two;

  bool has_bias;
  bool has_input_transpose;
  bool has_weight_transpose;

  bool is_resnet_replication;
  bool is_generic_replication;
  ac_int<2, false> num_channels;
  ac_int<3, false> fx_unrolling_lg2;

  bool has_attn_output_permute;
  bool is_mx_op;
  bool is_fc;
  bool write_output_to_accum_buffer;

  static const unsigned int base_width =
      5 * 64 /* OFFSETS */ + (12 + 10) * LOOP_WIDTH /* Loops */ +
      21 * 3 /* Loop indices */ + 5 /* stride */ + 2 /* padding */ +
      8 /* Head Size */ + 2 /* num_channels */ + 3 /*fx_unrolling_lg2*/ +
      11 * 1 /* Bools */;

  static const unsigned int extra_width =
      2 * DTYPE_INDEX_WIDTH + 36 +
      NUM_CODEBOOK_ENTRIES * DECODED_INPUT_DTYPE_WIDTH +
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
    for (int i = 0; i < 2; i++) {
      m& fyIndex[i];
    }
    m & fxIndex;
    for (int i = 0; i < 2; i++) {
      m& weightReuseIndex[i];
    }
    m & stride;
    m & padding;

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
    for (int i = 0; i < 2; i++) {
      m& weightAddressGenFyIndex[i];
    }
    m & weightAddressGenFxIndex;

    m & input_dtype;
    m & use_input_codebook;
    m & input_burst_size;
    m & input_num_beats;
    m & input_packing_factor_power;

    m & weight_dtype;
    m & use_weight_codebook;
    m & weight_burst_size;
    m & weight_num_beats;
    m & weight_packing_factor_power;

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
    m & is_resnet_replication;
    m & is_generic_replication;
    m & num_channels;
    m & fx_unrolling_lg2;
    m & has_attn_output_permute;
    m & is_mx_op;
    m & is_fc;
    m & write_output_to_accum_buffer;
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
    for (int i = 0; i < 2; i++) {
      os << "fyIndex[" << i << "]: " << params.fyIndex[i] << std::endl;
    }
    os << "fxIndex: " << params.fxIndex << std::endl;
    for (int i = 0; i < 2; i++) {
      os << "weightReuseIndex[" << i << "]: " << params.weightReuseIndex[i]
         << std::endl;
    }
    os << "stride: " << params.stride << std::endl;
    os << "padding: " << params.padding << std::endl;
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
    for (int i = 0; i < 2; i++) {
      os << "weightAddressGenFyIndex[" << i
         << "]: " << params.weightAddressGenFyIndex[i] << std::endl;
    }
    os << "weightAddressGenFxIndex: " << params.weightAddressGenFxIndex
       << std::endl;

    os << "input_dtype: " << params.input_dtype << std::endl;
    os << "use_input_codebook: " << params.use_input_codebook << std::endl;
    os << "input_num_beats: " << params.input_num_beats << std::endl;
    os << "input_packing_factor_power: " << params.input_packing_factor_power
       << std::endl;
    os << "input_burst_size: " << params.input_burst_size << std::endl;

    os << "weight_dtype: " << params.weight_dtype << std::endl;
    os << "use_weight_codebook: " << params.use_weight_codebook << std::endl;
    os << "weight_num_beats: " << params.weight_num_beats << std::endl;
    os << "weight_packing_factor_power: " << params.weight_packing_factor_power
       << std::endl;
    os << "weight_burst_size: " << params.weight_burst_size << std::endl;

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
    os << "is_resnet_replication: " << params.is_resnet_replication
       << std::endl;
    os << "is_generic_replication: " << params.is_generic_replication
       << std::endl;
    os << "num_channels: " << params.num_channels << std::endl;
    os << "fx_unrolling_lg2: " << params.fx_unrolling_lg2 << std::endl;
    os << "has_attn_output_permute: " << params.has_attn_output_permute
       << std::endl;
    os << "is_mx_op: " << params.is_mx_op << std::endl;
    os << "is_fc: " << params.is_fc << std::endl;
    os << "write_output_to_accum_buffer: "
       << params.write_output_to_accum_buffer << std::endl;
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
      if (lhs.fyIndex[i] != rhs.fyIndex[i]) return false;
      if (lhs.weightReuseIndex[i] != rhs.weightReuseIndex[i]) return false;
      if (lhs.weightAddressGenReductionLoopIndex[i] !=
          rhs.weightAddressGenReductionLoopIndex[i])
        return false;
      if (lhs.weightAddressGenWeightLoopIndex[i] !=
          rhs.weightAddressGenWeightLoopIndex[i])
        return false;
      if (lhs.weightAddressGenFyIndex[i] != rhs.weightAddressGenFyIndex[i])
        return false;
    }

    if (lhs.weightAddressGenReductionLoopIndex[2] !=
        rhs.weightAddressGenReductionLoopIndex[2])
      return false;

    // Compare other members
    if (lhs.fxIndex != rhs.fxIndex) return false;
    if (lhs.weightAddressGenFxIndex != rhs.weightAddressGenFxIndex)
      return false;
    if (lhs.stride != rhs.stride) return false;
    if (lhs.padding != rhs.padding) return false;

    if (lhs.input_dtype != rhs.input_dtype) return false;
    if (lhs.use_input_codebook != rhs.use_input_codebook) return false;
    if (lhs.input_num_beats != rhs.input_num_beats) return false;
    if (lhs.input_packing_factor_power != rhs.input_packing_factor_power)
      return false;
    if (lhs.input_burst_size != rhs.input_burst_size) return false;

    if (lhs.weight_dtype != rhs.weight_dtype) return false;
    if (lhs.use_weight_codebook != rhs.use_weight_codebook) return false;
    if (lhs.weight_num_beats != rhs.weight_num_beats) return false;
    if (lhs.weight_packing_factor_power != rhs.weight_packing_factor_power)
      return false;
    if (lhs.weight_burst_size != rhs.weight_burst_size) return false;

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
    if (lhs.is_resnet_replication != rhs.is_resnet_replication) return false;
    if (lhs.has_attn_output_permute != rhs.has_attn_output_permute)
      return false;
    if (lhs.is_mx_op != rhs.is_mx_op) return false;
    if (lhs.is_fc != rhs.is_fc) return false;

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
    inst_count = 1;
    vector_op0_src0 = 0;
    vector_op0_src1 = 0;
    vector_op2_src1 = 0;
    vector_op3_src1 = 0;
    vdequantize = 0;
    vector_dq_scale = 0;
    vector_op0 = 0;
    vector_op1 = 0;
    vector_op2 = 0;
    vector_op3 = 0;
    vdest = 0;
    reduce_count = 0;
    reduce_op = 0;
    rsqrt = 0;
    rreciprocal = 0;
    rscale = 0;
    rduplicate = 0;
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

  ac_int<24, false> inst_count;

  ac_int<4, false> vector_op0_src0;
  ac_int<4, false> vector_op0_src1;
  ac_int<4, false> vector_op2_src1;
  ac_int<4, false> vector_op3_src1;

  static const unsigned int from_matrix_unit = 1;
  static const unsigned int from_accumulation_buffer = 2;
  static const unsigned int from_matrix_vector_unit = 3;
  static const unsigned int from_vector_fetch_0 = 4;
  static const unsigned int from_vector_fetch_1 = 5;
  static const unsigned int from_vector_fetch_2 = 6;
  static const unsigned int from_accumulation = 7;
  static const unsigned int from_reduction_0 = 8;
  static const unsigned int from_reduction_1 = 9;
  static const unsigned int from_immediate_0 = 10;
  static const unsigned int from_immediate_1 = 11;
  static const unsigned int from_immediate_2 = 12;

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
  ac_int<1, false> rscale;
  ac_int<1, false> rduplicate;

  ac_int<2, false> rdest;
  static const unsigned int to_op0 = 1;
  static const unsigned int to_op2 = 2;
  static const unsigned int to_memory = 3;

  ac_int<2, false> vdest;
  static const unsigned int to_output = 1;
  static const unsigned int to_reduce = 2;
  static const unsigned int to_accumulate = 3;

  ac_int<16, false> immediate0;
  ac_int<16, false> immediate1;
  ac_int<16, false> immediate2;
  ac_int<64, false> VMAP_OFFSET;

  static const unsigned int width = 201;

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m & op_type;
    m & inst_count;
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
    m & rscale;
    m & rduplicate;
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
    os << "inst_count: " << params.inst_count << std::endl;
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
    os << "rscale: " << params.rscale << std::endl;
    os << "rduplicate: " << params.rduplicate << std::endl;
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
    return lhs.op_type == rhs.op_type && lhs.inst_count == rhs.inst_count &&
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
           lhs.rscale == rhs.rscale && lhs.rduplicate == rhs.rduplicate &&
           lhs.rdest == rhs.rdest &&
           lhs.vdest == rhs.vdest && lhs.immediate0 == rhs.immediate0 &&
           lhs.immediate1 == rhs.immediate1 &&
           lhs.immediate2 == rhs.immediate2 &&
           lhs.VMAP_OFFSET == rhs.VMAP_OFFSET;
  }
};

struct VectorParams : BaseParams {
#ifndef __SYNTHESIS__
  VectorParams() {
    vector_fetch_0_mode = 0;
    vector_fetch_0_offset = 0;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        vector_fetch_0_loops[i][j] = 0;
      }
    }
    for (int i = 0; i < 2; i++) {
      vector_fetch_0_y_loop_idx[i] = 0;
      vector_fetch_0_x_loop_idx[i] = 1;
      vector_fetch_0_k_loop_idx[i] = 2;
    }
    vector_fetch_0_dq_scale = 0;
    vector_fetch_0_dtype = 0;
    vector_fetch_0_burst_size = 0;
    vector_fetch_0_num_beats = 1;
    vector_fetch_0_packing_factor = 1;

    vector_fetch_1_mode = 0;
    vector_fetch_1_offset = 0;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        vector_fetch_1_loops[i][j] = 0;
      }
    }
    for (int i = 0; i < 2; i++) {
      vector_fetch_1_y_loop_idx[i] = 0;
      vector_fetch_1_x_loop_idx[i] = 1;
      vector_fetch_1_k_loop_idx[i] = 2;
    }
    vector_fetch_1_dq_scale = 0;
    vector_fetch_1_dtype = 0;
    vector_fetch_1_burst_size = 0;
    vector_fetch_1_num_beats = 1;
    vector_fetch_1_packing_factor = 1;

    vector_fetch_2_mode = 0;
    vector_fetch_2_offset = 0;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        vector_fetch_2_loops[i][j] = 0;
      }
    }
    for (int i = 0; i < 2; i++) {
      vector_fetch_2_y_loop_idx[i] = 0;
      vector_fetch_2_x_loop_idx[i] = 1;
      vector_fetch_2_k_loop_idx[i] = 2;
    }
    vector_fetch_2_dq_scale = 0;
    vector_fetch_2_dtype = 0;
    vector_fetch_2_burst_size = 0;
    vector_fetch_2_num_beats = 1;
    vector_fetch_2_packing_factor = 1;

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

    output_pad_dimension = false;
    output_pad_dim_size = 0;
    for (int i = 0; i < 2; i++) {
      output_pad_dim_idx[i] = 0;
    }

    vector_fetch_0_broadcast = 0;
    vector_fetch_1_broadcast = 0;
    vector_fetch_2_broadcast = 0;

    has_slicing = false;
    vector_fetch_0_dim = 0;
    vector_fetch_0_start = 0;
    vector_fetch_0_end = 0;
    vector_fetch_0_step = 0;

    has_permute = false;
    for (int i = 0; i < 6; i++) {
      vector_fetch_0_dims[i] = i;
    }

    has_transpose = false;

    is_maxpool = false;
    for (int i = 0; i < 2; i++) {
      stride[i] = 0;
    }
    for (int i = 0; i < 2; i++) {
      padding[i] = 0;
    }

    head_size_power_of_two = 32;
    has_attn_head_permute = false;

    quantize_output_mx = false;
    SCALE_OFFSET = 0;
    use_output_codebook = false;
    for (int i = 0; i < NUM_CODEBOOK_ENTRIES - 1; i++) {
      output_code[i] = 0;
    }
  }
#endif

  static constexpr int LOOP_WIDTH = 16;

  // Address generator 0
  ac_int<2, false> vector_fetch_0_mode;
  ac_int<ADDRESS_WIDTH, false> vector_fetch_0_offset;
  ac_int<LOOP_WIDTH, false> vector_fetch_0_loops[2][3];
  ac_int<3, false> vector_fetch_0_x_loop_idx[2];
  ac_int<3, false> vector_fetch_0_y_loop_idx[2];
  ac_int<3, false> vector_fetch_0_k_loop_idx[2];
  ac_int<16, false> vector_fetch_0_dq_scale;
  ac_int<4, false> vector_fetch_0_dtype;

  ac_int<10, false> vector_fetch_0_burst_size;
  ac_int<4, false> vector_fetch_0_num_beats;
  ac_int<4, false> vector_fetch_0_packing_factor;

  // Address generator 1
  ac_int<2, false> vector_fetch_1_mode;
  ac_int<ADDRESS_WIDTH, false> vector_fetch_1_offset;
  ac_int<LOOP_WIDTH, false> vector_fetch_1_loops[2][3];
  ac_int<3, false> vector_fetch_1_x_loop_idx[2];
  ac_int<3, false> vector_fetch_1_y_loop_idx[2];
  ac_int<3, false> vector_fetch_1_k_loop_idx[2];
  ac_int<16, false> vector_fetch_1_dq_scale;
  ac_int<4, false> vector_fetch_1_dtype;

  ac_int<10, false> vector_fetch_1_burst_size;
  ac_int<4, false> vector_fetch_1_num_beats;
  ac_int<4, false> vector_fetch_1_packing_factor;

  // Address generator 2
  ac_int<2, false> vector_fetch_2_mode;
  ac_int<ADDRESS_WIDTH, false> vector_fetch_2_offset;
  ac_int<LOOP_WIDTH, false> vector_fetch_2_loops[2][3];
  ac_int<3, false> vector_fetch_2_x_loop_idx[2];
  ac_int<3, false> vector_fetch_2_y_loop_idx[2];
  ac_int<3, false> vector_fetch_2_k_loop_idx[2];
  ac_int<16, false> vector_fetch_2_dq_scale;
  ac_int<4, false> vector_fetch_2_dtype;

  ac_int<10, false> vector_fetch_2_burst_size;
  ac_int<4, false> vector_fetch_2_num_beats;
  ac_int<4, false> vector_fetch_2_packing_factor;

  // Output address generator
  ac_int<2, false> output_mode;
  ac_int<ADDRESS_WIDTH, false> VECTOR_OUTPUT_OFFSET;
  ac_int<LOOP_WIDTH, false> output_loops[2][3];
  ac_int<3, false> output_x_loop_idx[2];
  ac_int<3, false> output_y_loop_idx[2];
  ac_int<3, false> output_k_loop_idx[2];
  ac_int<4, false> output_dtype;

  // support for shapes that need to be padded up
  bool output_pad_dimension;
  ac_int<11, false> output_pad_dim_size;
  ac_int<3, false> output_pad_dim_idx[2];

  ac_int<6, false> vector_fetch_0_broadcast;
  ac_int<3, false> vector_fetch_1_broadcast;
  ac_int<3, false> vector_fetch_2_broadcast;

  // Address generator 0 slicing and reshape
  bool has_slicing;
  ac_int<3, false> vector_fetch_0_dim;
  ac_int<11, false> vector_fetch_0_start;
  ac_int<11, false> vector_fetch_0_end;
  ac_int<11, false> vector_fetch_0_step;

  bool has_permute;
  ac_int<3, false> vector_fetch_0_dims[6];

  bool has_transpose;

  bool is_maxpool;
  ac_int<8, false> stride[2];
  ac_int<8, false> padding[2];

  // Transformer head permutation
  ac_int<4, false> head_size_power_of_two;
  bool has_attn_head_permute;

  bool quantize_output_mx;
  ac_int<ADDRESS_WIDTH, false> SCALE_OFFSET;

  bool use_output_codebook;
  ac_int<MAX_DECODED_DTYPE_WIDTH + 1, true>
      output_code[NUM_CODEBOOK_ENTRIES - 1];

  // Each address generator has a 2-bit mode flag, 64-bit address, 6 x 11-bit
  // loop boundaries, 6 x 3-bit loop indices, a 16-bit dequantize scale, a 4-bit
  // data type, and a 18-bit packing factor param
  static const unsigned int address_gen_width =
      2 + ADDRESS_WIDTH + 6 * LOOP_WIDTH + 6 * 3 + 16 + 4 + 18;

  static const unsigned int codebook_params_width =
      1 + (NUM_CODEBOOK_ENTRIES - 1) * (MAX_DECODED_DTYPE_WIDTH + 1);

  // There are 4 address generators in total + 12-bit broadcasting flag + 36-bit
  // slicing params + 32-bit pooling param + 18-bit reshape params + 17-bit
  // padded transpose params + 4-bit head size + 7 boolean flags + 64-bit scale
  // offset
  static const unsigned int width = 4 * address_gen_width + 12 + 36 + 32 + 18 +
                                    17 + 4 + 7 + ADDRESS_WIDTH - 16 - 18 +
                                    codebook_params_width;

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    // Address generator 0
    m & vector_fetch_0_mode;
    m & vector_fetch_0_offset;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        m& vector_fetch_0_loops[i][j];
      }
    }
    for (int i = 0; i < 2; i++) {
      m& vector_fetch_0_x_loop_idx[i];
    }
    for (int i = 0; i < 2; i++) {
      m& vector_fetch_0_y_loop_idx[i];
    }
    for (int i = 0; i < 2; i++) {
      m& vector_fetch_0_k_loop_idx[i];
    }
    m & vector_fetch_0_dq_scale;
    m & vector_fetch_0_dtype;
    m & vector_fetch_0_burst_size;
    m & vector_fetch_0_num_beats;
    m & vector_fetch_0_packing_factor;

    // Address generator 1
    m & vector_fetch_1_mode;
    m & vector_fetch_1_offset;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        m& vector_fetch_1_loops[i][j];
      }
    }
    for (int i = 0; i < 2; i++) {
      m& vector_fetch_1_x_loop_idx[i];
    }
    for (int i = 0; i < 2; i++) {
      m& vector_fetch_1_y_loop_idx[i];
    }
    for (int i = 0; i < 2; i++) {
      m& vector_fetch_1_k_loop_idx[i];
    }
    m & vector_fetch_1_dq_scale;
    m & vector_fetch_1_dtype;
    m & vector_fetch_1_burst_size;
    m & vector_fetch_1_num_beats;
    m & vector_fetch_1_packing_factor;

    // Address generator 2
    m & vector_fetch_2_mode;
    m & vector_fetch_2_offset;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        m& vector_fetch_2_loops[i][j];
      }
    }
    for (int i = 0; i < 2; i++) {
      m& vector_fetch_2_x_loop_idx[i];
    }
    for (int i = 0; i < 2; i++) {
      m& vector_fetch_2_y_loop_idx[i];
    }
    for (int i = 0; i < 2; i++) {
      m& vector_fetch_2_k_loop_idx[i];
    }
    m & vector_fetch_2_dq_scale;
    m & vector_fetch_2_dtype;
    m & vector_fetch_2_burst_size;
    m & vector_fetch_2_num_beats;
    m & vector_fetch_2_packing_factor;

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
    m & output_pad_dimension;
    m & output_pad_dim_size;
    for (int i = 0; i < 2; i++) {
      m& output_pad_dim_idx[i];
    }

    m & vector_fetch_0_broadcast;
    m & vector_fetch_1_broadcast;
    m & vector_fetch_2_broadcast;

    // Slicing and reshape
    m & has_slicing;
    m & vector_fetch_0_dim;
    m & vector_fetch_0_start;
    m & vector_fetch_0_end;
    m & vector_fetch_0_step;

    m & has_permute;
    for (int i = 0; i < 6; i++) {
      m& vector_fetch_0_dims[i];
    }

    m & has_transpose;

    m & is_maxpool;
    for (int i = 0; i < 2; i++) {
      m& stride[i];
    }
    for (int i = 0; i < 2; i++) {
      m& padding[i];
    }

    // Transformer head permutation flags
    m & head_size_power_of_two;
    m & has_attn_head_permute;

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
    os << "vector_fetch_0_mode: " << params.vector_fetch_0_mode << std::endl;
    os << "vector_fetch_0_broadcast: " << params.vector_fetch_0_broadcast
       << std::endl;
    os << "vector_fetch_0_offset: " << params.vector_fetch_0_offset
       << std::endl;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        os << "vector_fetch_0_loops[" << i << "][" << j
           << "]: " << params.vector_fetch_0_loops[i][j] << std::endl;
      }
    }
    for (int i = 0; i < 2; i++) {
      os << "vector_fetch_0_y_loop_idx[" << i
         << "]: " << params.vector_fetch_0_y_loop_idx[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "vector_fetch_0_x_loop_idx[" << i
         << "]: " << params.vector_fetch_0_x_loop_idx[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "vector_fetch_0_k_loop_idx[" << i
         << "]: " << params.vector_fetch_0_k_loop_idx[i] << std::endl;
    }
    os << "vector_fetch_0_dq_scale: " << params.vector_fetch_0_dq_scale
       << std::endl;
    os << "vector_fetch_0_dtype: " << params.vector_fetch_0_dtype << std::endl;
    os << "vector_fetch_0_burst_size: " << params.vector_fetch_0_burst_size
       << std::endl;
    os << "vector_fetch_0_num_beats: " << params.vector_fetch_0_num_beats
       << std::endl;
    os << "vector_fetch_0_packing_factor: "
       << params.vector_fetch_0_packing_factor << std::endl;

    os << "vector_fetch_1_mode: " << params.vector_fetch_1_mode << std::endl;
    os << "vector_fetch_1_broadcast: " << params.vector_fetch_1_broadcast
       << std::endl;
    os << "vector_fetch_1_offset: " << params.vector_fetch_1_offset
       << std::endl;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        os << "vector_fetch_1_loops[" << i << "][" << j
           << "]: " << params.vector_fetch_1_loops[i][j] << std::endl;
      }
    }
    for (int i = 0; i < 2; i++) {
      os << "vector_fetch_1_y_loop_idx[" << i
         << "]: " << params.vector_fetch_1_y_loop_idx[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "vector_fetch_1_x_loop_idx[" << i
         << "]: " << params.vector_fetch_1_x_loop_idx[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "vector_fetch_1_k_loop_idx[" << i
         << "]: " << params.vector_fetch_1_k_loop_idx[i] << std::endl;
    }
    os << "vector_fetch_1_dq_scale: " << params.vector_fetch_1_dq_scale
       << std::endl;
    os << "vector_fetch_1_dtype: " << params.vector_fetch_1_dtype << std::endl;
    os << "vector_fetch_1_burst_size: " << params.vector_fetch_1_burst_size
       << std::endl;
    os << "vector_fetch_1_num_beats: " << params.vector_fetch_1_num_beats
       << std::endl;
    os << "vector_fetch_1_packing_factor: "
       << params.vector_fetch_1_packing_factor << std::endl;

    os << "vector_fetch_2_mode: " << params.vector_fetch_2_mode << std::endl;
    os << "vector_fetch_2_broadcast: " << params.vector_fetch_2_broadcast
       << std::endl;
    os << "vector_fetch_2_offset: " << params.vector_fetch_2_offset
       << std::endl;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        os << "vector_fetch_2_loops[" << i << "][" << j
           << "]: " << params.vector_fetch_2_loops[i][j] << std::endl;
      }
    }
    for (int i = 0; i < 2; i++) {
      os << "vector_fetch_2_y_loop_idx[" << i
         << "]: " << params.vector_fetch_2_y_loop_idx[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "vector_fetch_2_x_loop_idx[" << i
         << "]: " << params.vector_fetch_2_x_loop_idx[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "vector_fetch_2_k_loop_idx[" << i
         << "]: " << params.vector_fetch_2_k_loop_idx[i] << std::endl;
    }
    os << "vector_fetch_2_dq_scale: " << params.vector_fetch_2_dq_scale
       << std::endl;
    os << "vector_fetch_2_dtype: " << params.vector_fetch_2_dtype << std::endl;
    os << "vector_fetch_2_burst_size: " << params.vector_fetch_2_burst_size
       << std::endl;
    os << "vector_fetch_2_num_beats: " << params.vector_fetch_2_num_beats
       << std::endl;
    os << "vector_fetch_2_packing_factor: "
       << params.vector_fetch_2_packing_factor << std::endl;

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

    os << "output_pad_dimension: " << params.output_pad_dimension << std::endl;
    os << "output_pad_dim_size: " << params.output_pad_dim_size << std::endl;
    for (int i = 0; i < 2; i++) {
      os << "output_pad_dim_idx[" << i << "]: " << params.output_pad_dim_idx[i]
         << std::endl;
    }

    os << "has_slicing: " << params.has_slicing << std::endl;
    os << "vector_fetch_0_dim: " << params.vector_fetch_0_dim << std::endl;
    os << "vector_fetch_0_start: " << params.vector_fetch_0_start << std::endl;
    os << "vector_fetch_0_end: " << params.vector_fetch_0_end << std::endl;
    os << "vector_fetch_0_step: " << params.vector_fetch_0_step << std::endl;

    os << "has_permute: " << params.has_permute << std::endl;
    for (int i = 0; i < 6; i++) {
      os << "vector_fetch_0_dims[" << i
         << "]: " << params.vector_fetch_0_dims[i] << std::endl;
    }

    os << "has_transpose: " << params.has_transpose << std::endl;

    os << "is_maxpool: " << params.is_maxpool << std::endl;
    for (int i = 0; i < 2; i++) {
      os << "stride[" << i << "]: " << params.stride[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "padding[" << i << "]: " << params.padding[i] << std::endl;
    }

    os << "head_size_power_of_two: " << params.head_size_power_of_two
       << std::endl;
    os << "has_attn_head_permute: " << params.has_attn_head_permute
       << std::endl;

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
    if (lhs.vector_fetch_0_mode != rhs.vector_fetch_0_mode) return false;
    if (lhs.vector_fetch_0_offset != rhs.vector_fetch_0_offset) return false;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        if (lhs.vector_fetch_0_loops[i][j] != rhs.vector_fetch_0_loops[i][j])
          return false;
      }
    }
    for (int i = 0; i < 2; i++) {
      if (lhs.vector_fetch_0_x_loop_idx[i] != rhs.vector_fetch_0_x_loop_idx[i])
        return false;
      if (lhs.vector_fetch_0_y_loop_idx[i] != rhs.vector_fetch_0_y_loop_idx[i])
        return false;
      if (lhs.vector_fetch_0_k_loop_idx[i] != rhs.vector_fetch_0_k_loop_idx[i])
        return false;
    }
    if (lhs.vector_fetch_0_dq_scale != rhs.vector_fetch_0_dq_scale)
      return false;
    if (lhs.vector_fetch_0_dtype != rhs.vector_fetch_0_dtype) return false;
    if (lhs.vector_fetch_0_burst_size != rhs.vector_fetch_0_burst_size)
      return false;
    if (lhs.vector_fetch_0_num_beats != rhs.vector_fetch_0_num_beats)
      return false;
    if (lhs.vector_fetch_0_packing_factor != rhs.vector_fetch_0_packing_factor)
      return false;

    // Compare Address Gen 1 members
    if (lhs.vector_fetch_1_mode != rhs.vector_fetch_1_mode) return false;
    if (lhs.vector_fetch_1_offset != rhs.vector_fetch_1_offset) return false;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        if (lhs.vector_fetch_1_loops[i][j] != rhs.vector_fetch_1_loops[i][j])
          return false;
      }
    }
    for (int i = 0; i < 2; i++) {
      if (lhs.vector_fetch_1_x_loop_idx[i] != rhs.vector_fetch_1_x_loop_idx[i])
        return false;
      if (lhs.vector_fetch_1_y_loop_idx[i] != rhs.vector_fetch_1_y_loop_idx[i])
        return false;
      if (lhs.vector_fetch_1_k_loop_idx[i] != rhs.vector_fetch_1_k_loop_idx[i])
        return false;
    }
    if (lhs.vector_fetch_1_dq_scale != rhs.vector_fetch_1_dq_scale)
      return false;
    if (lhs.vector_fetch_1_dtype != rhs.vector_fetch_1_dtype) return false;
    if (lhs.vector_fetch_1_burst_size != rhs.vector_fetch_1_burst_size)
      return false;
    if (lhs.vector_fetch_1_num_beats != rhs.vector_fetch_1_num_beats)
      return false;
    if (lhs.vector_fetch_1_packing_factor != rhs.vector_fetch_1_packing_factor)
      return false;

    // Compare Address Gen 2 members
    if (lhs.vector_fetch_2_mode != rhs.vector_fetch_2_mode) return false;
    if (lhs.vector_fetch_2_offset != rhs.vector_fetch_2_offset) return false;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        if (lhs.vector_fetch_2_loops[i][j] != rhs.vector_fetch_2_loops[i][j])
          return false;
      }
    }
    for (int i = 0; i < 2; i++) {
      if (lhs.vector_fetch_2_x_loop_idx[i] != rhs.vector_fetch_2_x_loop_idx[i])
        return false;
      if (lhs.vector_fetch_2_y_loop_idx[i] != rhs.vector_fetch_2_y_loop_idx[i])
        return false;
      if (lhs.vector_fetch_2_k_loop_idx[i] != rhs.vector_fetch_2_k_loop_idx[i])
        return false;
    }
    if (lhs.vector_fetch_2_dq_scale != rhs.vector_fetch_2_dq_scale)
      return false;
    if (lhs.vector_fetch_2_dtype != rhs.vector_fetch_2_dtype) return false;
    if (lhs.vector_fetch_2_burst_size != rhs.vector_fetch_2_burst_size)
      return false;
    if (lhs.vector_fetch_2_num_beats != rhs.vector_fetch_2_num_beats)
      return false;
    if (lhs.vector_fetch_2_packing_factor != rhs.vector_fetch_2_packing_factor)
      return false;

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

    if (lhs.output_pad_dimension != rhs.output_pad_dimension) return false;
    if (lhs.output_pad_dim_size != rhs.output_pad_dim_size) return false;
    for (int i = 0; i < 2; i++) {
      if (lhs.output_pad_dim_idx[i] != rhs.output_pad_dim_idx[i]) return false;
    }

    if (lhs.vector_fetch_0_broadcast != rhs.vector_fetch_0_broadcast)
      return false;
    if (lhs.vector_fetch_1_broadcast != rhs.vector_fetch_1_broadcast)
      return false;
    if (lhs.vector_fetch_2_broadcast != rhs.vector_fetch_2_broadcast)
      return false;

    if (lhs.has_slicing != rhs.has_slicing) return false;
    if (lhs.vector_fetch_0_dim != rhs.vector_fetch_0_dim) return false;
    if (lhs.vector_fetch_0_start != rhs.vector_fetch_0_start) return false;
    if (lhs.vector_fetch_0_end != rhs.vector_fetch_0_end) return false;
    if (lhs.vector_fetch_0_step != rhs.vector_fetch_0_step) return false;

    if (lhs.has_permute != rhs.has_permute) return false;
    for (int i = 0; i < 6; i++) {
      if (lhs.vector_fetch_0_dims[i] != rhs.vector_fetch_0_dims[i])
        return false;
    }

    if (lhs.has_transpose != rhs.has_transpose) return false;

    if (lhs.is_maxpool != rhs.is_maxpool) return false;
    for (int i = 0; i < 2; i++) {
      if (lhs.stride[i] != rhs.stride[i]) return false;
    }
    for (int i = 0; i < 2; i++) {
      if (lhs.padding[i] != rhs.padding[i]) return false;
    }

    if (lhs.head_size_power_of_two != rhs.head_size_power_of_two) return false;
    if (lhs.has_attn_head_permute != rhs.has_attn_head_permute) return false;

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
      m& inst[j].inst_count;
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
      m& inst[j].rscale;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].rduplicate;
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
    for (int i = 0; i < params.instLen; i++) {
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
