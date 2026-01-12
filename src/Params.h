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
    input_offset = 0;
    input_scale_offset = 0;
    weight_offset = 0;
    weight_scale_offset = 0;
    bias_offset = 0;

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 6; j++) {
        loops[i][j] = 0;
      }
    }
    for (int i = 0; i < 2; i++) {
      x_loop_idx[i] = 0;
      y_loop_idx[i] = 0;
      reduction_loop_idx[i] = 0;
      weight_loop_idx[i] = 0;
      weight_reuse_idx[i] = 0;
      fy_loop_idx[i] = 0;
    }
    fx_loop_idx = 0;
    stride = 1;
    padding = 0;

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 5; j++) {
        weight_addr_loops[i][j] = 0;
      }
    }
    for (int i = 0; i < 3; i++) {
      weight_addr_reduction_loop_idx[i] = 0;
    }
    for (int i = 0; i < 2; i++) {
      weight_addr_weight_loop_idx[i] = 0;
    }
    for (int i = 0; i < 2; i++) {
      weight_addr_fy_idx[i] = 0;
    }
    weight_addr_fx_idx = 0;

    input_dtype = 0;
    use_input_codebook = false;
    input_burst_size = 0;
    input_num_beats = 1;
    input_pack_factor_lg2 = 1;

    weight_dtype = 0;
    use_weight_codebook = false;
    weight_burst_size = 0;
    weight_num_beats = 1;
    weight_pack_factor_lg2 = 1;

    for (int i = 0; i < NUM_CODEBOOK_ENTRIES; i++) {
      input_code[i] = 0;
      weight_code[i] = 0;
    }

    input_code_zero_idx = 0;

    head_size_lg2 = 0;

    is_resnet_replication = false;
    is_generic_replication = false;
    num_channels = 0;
    fx_unrolling_lg2 = 0;

    input_y = 0;
    input_x = 0;

    has_bias = false;
    is_mx_op = false;
    is_fc = false;
    merge_heads = false;
    input_transpose = false;
    weight_transpose = false;
    write_output_to_accum_buffer = false;

    weight_dequant = false;
    dq_scale_offset = 0;
    dq_zero_point_offset = 0;

#if SUPPORT_SPMM
    is_spmm = false;
    spmm_data_offset = 0;
    spmm_indices_offset = 0;
    spmm_indptr_offset = 0;
#endif
  }
#endif

  static constexpr int LOOP_WIDTH = 16;

  ac_int<ADDRESS_WIDTH, false> input_offset;
  ac_int<ADDRESS_WIDTH, false> input_scale_offset;
  ac_int<ADDRESS_WIDTH, false> weight_offset;
  ac_int<ADDRESS_WIDTH, false> weight_scale_offset;
  ac_int<ADDRESS_WIDTH, false> bias_offset;

  // systolic array loop
  ac_int<LOOP_WIDTH, false> loops[2][6];
  ac_int<3, false> x_loop_idx[2];
  ac_int<3, false> y_loop_idx[2];
  ac_int<3, false> reduction_loop_idx[2];
  ac_int<3, false> weight_loop_idx[2];
  ac_int<3, false> fy_loop_idx[2];
  ac_int<3, false> fx_loop_idx;
  ac_int<3, false> weight_reuse_idx[2];
  ac_int<8, false> stride;
  ac_int<8, false> padding;

  // weight address generator loop
  ac_int<LOOP_WIDTH, false> weight_addr_loops[2][5];
  // in the inner loop, there are actually 2 reduction loops: the
  // standard reduction loop and the reduction that is parallelized in
  // the systolic array
  ac_int<3, false> weight_addr_reduction_loop_idx[3];
  ac_int<3, false> weight_addr_weight_loop_idx[2];
  ac_int<3, false> weight_addr_fy_idx[2];
  ac_int<3, false> weight_addr_fx_idx;

  ac_int<DTYPE_INDEX_WIDTH, false> input_dtype;
  bool use_input_codebook;
  ac_int<10, false> input_burst_size;
  ac_int<4, false> input_num_beats;
  ac_int<4, false> input_pack_factor_lg2;

  ac_int<DTYPE_INDEX_WIDTH, false> weight_dtype;
  bool use_weight_codebook;
  ac_int<10, false> weight_burst_size;
  ac_int<4, false> weight_num_beats;
  ac_int<4, false> weight_pack_factor_lg2;

  ac_int<DECODED_INPUT_DTYPE_WIDTH, false> input_code[NUM_CODEBOOK_ENTRIES];
  ac_int<DECODED_WEIGHT_DTYPE_WIDTH, false> weight_code[NUM_CODEBOOK_ENTRIES];

  ac_int<4, false> input_code_zero_idx;

  ac_int<8, false> head_size_lg2;

  bool is_resnet_replication;
  bool is_generic_replication;
  ac_int<2, false> num_channels;
  ac_int<3, false> fx_unrolling_lg2;

  ac_int<16, false> input_y;
  ac_int<16, false> input_x;

  bool has_bias;
  bool is_mx_op;
  bool is_fc;
  bool merge_heads;
  bool input_transpose;
  bool weight_transpose;
  bool write_output_to_accum_buffer;

  bool weight_dequant;
  ac_int<ADDRESS_WIDTH, false> dq_scale_offset;
  ac_int<ADDRESS_WIDTH, false> dq_zero_point_offset;

#if SUPPORT_SPMM
  bool is_spmm;
  ac_int<ADDRESS_WIDTH, false> spmm_data_offset;
  ac_int<ADDRESS_WIDTH, false> spmm_indices_offset;
  ac_int<ADDRESS_WIDTH, false> spmm_indptr_offset;
#endif

  static const unsigned int base_width =
      7 * ADDRESS_WIDTH /* addresses */ + (12 + 10) * LOOP_WIDTH /* loops */ +
      21 * 3 /* loop indices */ + 8 /* stride */ + 8 /* padding */ +
      8 /* Head Size */ + 2 /* num_channels */ + 3 /* fx_unrolling_lg2 */ +
      12 * 1 /* Bools */ + 32 /* input shapes */;

  static const unsigned int extra_width =
      2 * DTYPE_INDEX_WIDTH + 36 +
      NUM_CODEBOOK_ENTRIES * DECODED_INPUT_DTYPE_WIDTH +
      NUM_CODEBOOK_ENTRIES * DECODED_WEIGHT_DTYPE_WIDTH + 4;

#if SUPPORT_SPMM
  static const unsigned int spmm_extra_width = 3 * ADDRESS_WIDTH + 1;
#else
  static const unsigned int spmm_extra_width = 0;
#endif

  static const unsigned int width = base_width + extra_width + spmm_extra_width;

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m & input_offset;
    m & input_scale_offset;
    m & weight_offset;
    m & weight_scale_offset;
    m & bias_offset;

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 6; j++) {
        m& loops[i][j];
      }
    }
    for (int i = 0; i < 2; i++) {
      m& x_loop_idx[i];
    }
    for (int i = 0; i < 2; i++) {
      m& y_loop_idx[i];
    }
    for (int i = 0; i < 2; i++) {
      m& reduction_loop_idx[i];
    }
    for (int i = 0; i < 2; i++) {
      m& weight_loop_idx[i];
    }
    for (int i = 0; i < 2; i++) {
      m& fy_loop_idx[i];
    }
    m & fx_loop_idx;
    for (int i = 0; i < 2; i++) {
      m& weight_reuse_idx[i];
    }
    m & stride;
    m & padding;

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 5; j++) {
        m& weight_addr_loops[i][j];
      }
    }
    for (int i = 0; i < 3; i++) {
      m& weight_addr_reduction_loop_idx[i];
    }
    for (int i = 0; i < 2; i++) {
      m& weight_addr_weight_loop_idx[i];
    }
    for (int i = 0; i < 2; i++) {
      m& weight_addr_fy_idx[i];
    }
    m & weight_addr_fx_idx;

    m & input_dtype;
    m & use_input_codebook;
    m & input_burst_size;
    m & input_num_beats;
    m & input_pack_factor_lg2;

    m & weight_dtype;
    m & use_weight_codebook;
    m & weight_burst_size;
    m & weight_num_beats;
    m & weight_pack_factor_lg2;

    for (int i = 0; i < NUM_CODEBOOK_ENTRIES; i++) {
      m& input_code[i];
    }

    for (int i = 0; i < NUM_CODEBOOK_ENTRIES; i++) {
      m& weight_code[i];
    }

    m & input_code_zero_idx;

    m & head_size_lg2;

    m & is_resnet_replication;
    m & is_generic_replication;
    m & num_channels;
    m & fx_unrolling_lg2;

    m & input_y;
    m & input_x;

    m & has_bias;
    m & is_mx_op;
    m & is_fc;
    m & merge_heads;
    m & input_transpose;
    m & weight_transpose;
    m & write_output_to_accum_buffer;

    m & weight_dequant;
    m & dq_scale_offset;
    m & dq_zero_point_offset;

#if SUPPORT_SPMM
    m & is_spmm;
    m & spmm_data_offset;
    m & spmm_indices_offset;
    m & spmm_indptr_offset;
#endif
  }

  inline friend void sc_trace(sc_trace_file* tf, const MatrixParams& params,
                              const std::string& name) {
    // TODO
  }
#endif

  inline friend std::ostream& operator<<(ostream& os,
                                         const MatrixParams& params) {
    os << "input_offset: " << params.input_offset << std::endl;
    os << "input_scale_offset: " << params.input_scale_offset << std::endl;
    os << "weight_offset: " << params.weight_offset << std::endl;
    os << "weight_scale_offset: " << params.weight_scale_offset << std::endl;
    os << "bias_offset: " << params.bias_offset << std::endl;

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 6; j++) {
        os << "loops[" << i << "][" << j << "]: " << params.loops[i][j]
           << std::endl;
      }
    }
    for (int i = 0; i < 2; i++) {
      os << "x_loop_idx[" << i << "]: " << params.x_loop_idx[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "y_loop_idx[" << i << "]: " << params.y_loop_idx[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "reduction_loop_idx[" << i << "]: " << params.reduction_loop_idx[i]
         << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "weight_loop_idx[" << i << "]: " << params.weight_loop_idx[i]
         << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "fy_loop_idx[" << i << "]: " << params.fy_loop_idx[i] << std::endl;
    }
    os << "fx_loop_idx: " << params.fx_loop_idx << std::endl;
    for (int i = 0; i < 2; i++) {
      os << "weight_reuse_idx[" << i << "]: " << params.weight_reuse_idx[i]
         << std::endl;
    }
    os << "stride: " << params.stride << std::endl;
    os << "padding: " << params.padding << std::endl;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 5; j++) {
        os << "weight_addr_loops[" << i << "][" << j
           << "]: " << params.weight_addr_loops[i][j] << std::endl;
      }
    }
    for (int i = 0; i < 3; i++) {
      os << "weight_addr_reduction_loop_idx[" << i
         << "]: " << params.weight_addr_reduction_loop_idx[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "weight_addr_weight_loop_idx[" << i
         << "]: " << params.weight_addr_weight_loop_idx[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "weight_addr_fy_idx[" << i << "]: " << params.weight_addr_fy_idx[i]
         << std::endl;
    }
    os << "weight_addr_fx_idx: " << params.weight_addr_fx_idx << std::endl;

    os << "input_dtype: " << params.input_dtype << std::endl;
    os << "use_input_codebook: " << params.use_input_codebook << std::endl;
    os << "input_num_beats: " << params.input_num_beats << std::endl;
    os << "input_pack_factor_lg2: " << params.input_pack_factor_lg2
       << std::endl;
    os << "input_burst_size: " << params.input_burst_size << std::endl;

    os << "weight_dtype: " << params.weight_dtype << std::endl;
    os << "use_weight_codebook: " << params.use_weight_codebook << std::endl;
    os << "weight_num_beats: " << params.weight_num_beats << std::endl;
    os << "weight_pack_factor_lg2: " << params.weight_pack_factor_lg2
       << std::endl;
    os << "weight_burst_size: " << params.weight_burst_size << std::endl;

    for (int i = 0; i < NUM_CODEBOOK_ENTRIES; i++) {
      os << "input_code[" << i << "]: " << params.input_code[i] << std::endl;
    }

    for (int i = 0; i < NUM_CODEBOOK_ENTRIES; i++) {
      os << "weight_code[" << i << "]: " << params.weight_code[i] << std::endl;
    }

    os << "input_code_zero_idx: " << params.input_code_zero_idx << std::endl;

    os << "head_size_lg2: " << params.head_size_lg2 << std::endl;

    os << "is_resnet_replication: " << params.is_resnet_replication
       << std::endl;
    os << "is_generic_replication: " << params.is_generic_replication
       << std::endl;
    os << "num_channels: " << params.num_channels << std::endl;
    os << "fx_unrolling_lg2: " << params.fx_unrolling_lg2 << std::endl;

    os << "input_y: " << params.input_y << std::endl;
    os << "input_x: " << params.input_x << std::endl;

    os << "has_bias: " << params.has_bias << std::endl;
    os << "is_mx_op: " << params.is_mx_op << std::endl;
    os << "is_fc: " << params.is_fc << std::endl;
    os << "merge_heads: " << params.merge_heads << std::endl;
    os << "input_transpose: " << params.input_transpose << std::endl;
    os << "weight_transpose: " << params.weight_transpose << std::endl;
    os << "write_output_to_accum_buffer: "
       << params.write_output_to_accum_buffer << std::endl;
    os << "weight_dequant: " << params.weight_dequant << std::endl;
    os << "dq_scale_offset: " << params.dq_scale_offset << std::endl;
    os << "dq_zero_point_offset: " << params.dq_zero_point_offset << std::endl;

#if SUPPORT_SPMM
    os << "is_spmm: " << params.is_spmm << std::endl;
    os << "spmm_data_offset: " << params.spmm_data_offset << std::endl;
    os << "spmm_indices_offset: " << params.spmm_indices_offset << std::endl;
    os << "spmm_indptr_offset: " << params.spmm_indptr_offset << std::endl;
#endif
    return os;
  }

  inline friend bool operator==(const MatrixParams& lhs,
                                const MatrixParams& rhs) {
    if (lhs.input_offset != rhs.input_offset ||
        lhs.input_scale_offset != rhs.input_scale_offset ||
        lhs.weight_offset != rhs.weight_offset ||
        lhs.weight_scale_offset != rhs.weight_scale_offset ||
        lhs.bias_offset != rhs.bias_offset)
      return false;

    // Compare the 2D arrays
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 6; j++) {
        if (lhs.loops[i][j] != rhs.loops[i][j]) return false;
      }
    }

    for (int i = 0; i < 2; i++) {
      if (lhs.x_loop_idx[i] != rhs.x_loop_idx[i]) return false;
      if (lhs.y_loop_idx[i] != rhs.y_loop_idx[i]) return false;
      if (lhs.reduction_loop_idx[i] != rhs.reduction_loop_idx[i]) return false;
      if (lhs.weight_loop_idx[i] != rhs.weight_loop_idx[i]) return false;
      if (lhs.fy_loop_idx[i] != rhs.fy_loop_idx[i]) return false;
      if (lhs.weight_reuse_idx[i] != rhs.weight_reuse_idx[i]) return false;
      if (lhs.weight_addr_reduction_loop_idx[i] !=
          rhs.weight_addr_reduction_loop_idx[i])
        return false;
      if (lhs.weight_addr_weight_loop_idx[i] !=
          rhs.weight_addr_weight_loop_idx[i])
        return false;
      if (lhs.weight_addr_fy_idx[i] != rhs.weight_addr_fy_idx[i]) return false;
    }

    if (lhs.weight_addr_reduction_loop_idx[2] !=
        rhs.weight_addr_reduction_loop_idx[2])
      return false;

    // Compare other members
    if (lhs.fx_loop_idx != rhs.fx_loop_idx) return false;
    if (lhs.weight_addr_fx_idx != rhs.weight_addr_fx_idx) return false;
    if (lhs.stride != rhs.stride) return false;
    if (lhs.padding != rhs.padding) return false;

    if (lhs.input_dtype != rhs.input_dtype) return false;
    if (lhs.use_input_codebook != rhs.use_input_codebook) return false;
    if (lhs.input_num_beats != rhs.input_num_beats) return false;
    if (lhs.input_pack_factor_lg2 != rhs.input_pack_factor_lg2) return false;
    if (lhs.input_burst_size != rhs.input_burst_size) return false;

    if (lhs.weight_dtype != rhs.weight_dtype) return false;
    if (lhs.use_weight_codebook != rhs.use_weight_codebook) return false;
    if (lhs.weight_num_beats != rhs.weight_num_beats) return false;
    if (lhs.weight_pack_factor_lg2 != rhs.weight_pack_factor_lg2) return false;
    if (lhs.weight_burst_size != rhs.weight_burst_size) return false;

    for (int i = 0; i < NUM_CODEBOOK_ENTRIES; i++) {
      if (lhs.input_code[i] != rhs.input_code[i]) return false;
      if (lhs.weight_code[i] != rhs.weight_code[i]) return false;
    }

    if (lhs.input_code_zero_idx != rhs.input_code_zero_idx) return false;

    if (lhs.head_size_lg2 != rhs.head_size_lg2) return false;

    if (lhs.is_resnet_replication != rhs.is_resnet_replication) return false;
    if (lhs.is_generic_replication != rhs.is_generic_replication) return false;
    if (lhs.num_channels != rhs.num_channels) return false;
    if (lhs.fx_unrolling_lg2 != rhs.fx_unrolling_lg2) return false;

    if (lhs.input_y != rhs.input_y) return false;
    if (lhs.input_x != rhs.input_x) return false;

    if (lhs.has_bias != rhs.has_bias || lhs.bias_offset != rhs.bias_offset)
      return false;
    if (lhs.is_mx_op != rhs.is_mx_op) return false;
    if (lhs.is_fc != rhs.is_fc) return false;
    if (lhs.merge_heads != rhs.merge_heads) return false;
    if (lhs.input_transpose != rhs.input_transpose) return false;
    if (lhs.weight_transpose != rhs.weight_transpose) return false;
    if (lhs.write_output_to_accum_buffer != rhs.write_output_to_accum_buffer)
      return false;

    if (lhs.weight_dequant != rhs.weight_dequant) return false;
    if (lhs.dq_scale_offset != rhs.dq_scale_offset) return false;
    if (lhs.dq_zero_point_offset != rhs.dq_zero_point_offset) return false;

#if SUPPORT_SPMM
    if (lhs.is_spmm != rhs.is_spmm) return false;
    if (lhs.spmm_data_offset != rhs.spmm_data_offset) return false;
    if (lhs.spmm_indices_offset != rhs.spmm_indices_offset) return false;
    if (lhs.spmm_indptr_offset != rhs.spmm_indptr_offset) return false;
#endif

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
    inst_loop_count = 1;
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
  }
#endif

  ac_int<2, false> op_type;
  static const unsigned int vector = 0;
  static const unsigned int reduction = 1;
  static const unsigned int accumulation = 2;

  ac_int<24, false> inst_loop_count;

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
  static const unsigned int from_dwc_unit = 13;
  static const unsigned int from_spmm_unit = 14;

  ac_int<1, false> vdequantize;
  ac_int<16, false> vector_dq_scale;

  // Stage 0: add, sub, mult
  ac_int<2, false> vector_op0;
  static const unsigned int nop = 0;
  static const unsigned int vadd = 1;
  static const unsigned int vmult = 2;
  static const unsigned int vsub = 3;

  // Stage 1: exp, abs, activations
  ac_int<2, false> vector_op1;
  static const unsigned int vpoly = 1;
  static const unsigned int vabs = 2;
  static const unsigned int vrelu = 3;

  // Stage 2: add, mult, square
  ac_int<2, false> vector_op2;
  static const unsigned int vsquare = 3;

  ac_int<2, false> vector_op3;
  static const unsigned int vdiv = 1;
  static const unsigned int vquantize_mx = 2;
  static const unsigned int vquantize_mx_outlier = 3;

  ac_int<10, false> reduce_count;
  ac_int<2, false> reduce_op;
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

  static const unsigned int width = 135;

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m & op_type;
    m & inst_loop_count;
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
    os << "inst_loop_count: " << params.inst_loop_count << std::endl;
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
    return os;
  }

  inline friend bool operator==(const VectorInstructions& lhs,
                                const VectorInstructions& rhs) {
    return lhs.op_type == rhs.op_type &&
           lhs.inst_loop_count == rhs.inst_loop_count &&
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
           lhs.rreciprocal == rhs.rreciprocal && lhs.rscale == rhs.rscale &&
           lhs.rduplicate == rhs.rduplicate && lhs.rdest == rhs.rdest &&
           lhs.vdest == rhs.vdest && lhs.immediate0 == rhs.immediate0 &&
           lhs.immediate1 == rhs.immediate1 && lhs.immediate2 == rhs.immediate2;
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
    vector_output_offset = 0;
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

    vector_fetch_0_broadcast = 0;
    vector_fetch_1_broadcast = 0;
    vector_fetch_2_broadcast = 0;

    has_slicing = false;
    vector_fetch_0_slice_dim = 0;
    vector_fetch_0_slice_start = 0;
    vector_fetch_0_slice_end = 0;
    vector_fetch_0_slice_step = 0;

    has_permute = false;
    for (int i = 0; i < 6; i++) {
      vector_fetch_0_permute_dims[i] = i;
    }

    has_transpose = false;

    is_maxpool = false;
    for (int i = 0; i < 2; i++) {
      stride[i] = 0;
    }
    for (int i = 0; i < 2; i++) {
      padding[i] = 0;
    }

    head_size_lg2 = 32;
    transpose_for_scores = false;

    quantize_output_mx = false;
    mx_scale_offset = 0;

    has_sparse_output = false;
    csr_data_offset = 0;
    csr_indices_offset = 0;
    csr_indptr_offset = 0;

    use_output_codebook = false;
    for (int i = 0; i < NUM_CODEBOOK_ENTRIES - 1; i++) {
      output_code[i] = 0;
    }

    is_dwc = false;
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
  ac_int<ADDRESS_WIDTH, false> vector_output_offset;
  ac_int<LOOP_WIDTH, false> output_loops[2][3];
  ac_int<3, false> output_x_loop_idx[2];
  ac_int<3, false> output_y_loop_idx[2];
  ac_int<3, false> output_k_loop_idx[2];
  ac_int<4, false> output_dtype;

  ac_int<6, false> vector_fetch_0_broadcast;
  ac_int<3, false> vector_fetch_1_broadcast;
  ac_int<3, false> vector_fetch_2_broadcast;

  // Address generator 0 slicing and reshape
  bool has_slicing;
  ac_int<3, false> vector_fetch_0_slice_dim;
  ac_int<11, false> vector_fetch_0_slice_start;
  ac_int<11, false> vector_fetch_0_slice_end;
  ac_int<11, false> vector_fetch_0_slice_step;

  bool has_permute;
  ac_int<3, false> vector_fetch_0_permute_dims[6];

  bool has_transpose;

  bool is_maxpool;
  ac_int<8, false> stride[2];
  ac_int<8, false> padding[2];

  // Transformer head permutation
  ac_int<4, false> head_size_lg2;
  bool transpose_for_scores;

  bool quantize_output_mx;
  ac_int<ADDRESS_WIDTH, false> mx_scale_offset;

  // Sparse tensor output
  bool has_sparse_output;
  ac_int<ADDRESS_WIDTH, false> csr_data_offset;
  ac_int<ADDRESS_WIDTH, false> csr_indices_offset;
  ac_int<ADDRESS_WIDTH, false> csr_indptr_offset;

  bool use_output_codebook;
  ac_int<MAX_DECODED_DTYPE_WIDTH + 1, true>
      output_code[NUM_CODEBOOK_ENTRIES - 1];

  bool is_dwc;

  // Each address generator has a 2-bit mode flag, 64-bit address, 6 x 11-bit
  // loop boundaries, 6 x 3-bit loop indices, a 16-bit dequantize scale, a 4-bit
  // data type, and a 18-bit packing factor param
  static const unsigned int address_gen_width =
      2 + ADDRESS_WIDTH + 6 * LOOP_WIDTH + 6 * 3 + 16 + 4 + 18;

  static const unsigned int codebook_params_width =
      (NUM_CODEBOOK_ENTRIES - 1) * (MAX_DECODED_DTYPE_WIDTH + 1);

  static const unsigned int sparse_params_width = 3 * ADDRESS_WIDTH + 1;

  // There are 4 address generators in total + 12-bit broadcasting flag + 36-bit
  // slicing params + 32-bit pooling param + 18-bit reshape params + 4-bit head
  // size + 8 boolean flags + 64-bit scale offset - output dequantize scale and
  // packing params + cookbook params
  static const unsigned int width = 4 * address_gen_width + 12 + 36 + 32 + 18 +
                                    4 + 8 + ADDRESS_WIDTH - 16 - 18 +
                                    codebook_params_width + sparse_params_width;

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
    m & vector_output_offset;
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

    m & vector_fetch_0_broadcast;
    m & vector_fetch_1_broadcast;
    m & vector_fetch_2_broadcast;

    // Slicing and reshape
    m & has_slicing;
    m & vector_fetch_0_slice_dim;
    m & vector_fetch_0_slice_start;
    m & vector_fetch_0_slice_end;
    m & vector_fetch_0_slice_step;

    m & has_permute;
    for (int i = 0; i < 6; i++) {
      m& vector_fetch_0_permute_dims[i];
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
    m & head_size_lg2;
    m & transpose_for_scores;

    m & quantize_output_mx;
    m & mx_scale_offset;

    // Sparse tensor output offsets
    m & has_sparse_output;
    m & csr_data_offset;
    m & csr_indices_offset;
    m & csr_indptr_offset;

    m & use_output_codebook;
    for (int i = 0; i < NUM_CODEBOOK_ENTRIES - 1; i++) {
      m& output_code[i];
    }

    m & is_dwc;
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
    os << "vector_output_offset: " << params.vector_output_offset << std::endl;
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

    os << "has_slicing: " << params.has_slicing << std::endl;
    os << "vector_fetch_0_slice_dim: " << params.vector_fetch_0_slice_dim
       << std::endl;
    os << "vector_fetch_0_slice_start: " << params.vector_fetch_0_slice_start
       << std::endl;
    os << "vector_fetch_0_slice_end: " << params.vector_fetch_0_slice_end
       << std::endl;
    os << "vector_fetch_0_slice_step: " << params.vector_fetch_0_slice_step
       << std::endl;

    os << "has_permute: " << params.has_permute << std::endl;
    for (int i = 0; i < 6; i++) {
      os << "vector_fetch_0_permute_dims[" << i
         << "]: " << params.vector_fetch_0_permute_dims[i] << std::endl;
    }

    os << "has_transpose: " << params.has_transpose << std::endl;

    os << "is_maxpool: " << params.is_maxpool << std::endl;
    for (int i = 0; i < 2; i++) {
      os << "stride[" << i << "]: " << params.stride[i] << std::endl;
    }
    for (int i = 0; i < 2; i++) {
      os << "padding[" << i << "]: " << params.padding[i] << std::endl;
    }

    os << "head_size_lg2: " << params.head_size_lg2 << std::endl;
    os << "transpose_for_scores: " << params.transpose_for_scores << std::endl;

    os << "quantize_output_mx: " << params.quantize_output_mx << std::endl;
    os << "mx_scale_offset: " << params.mx_scale_offset << std::endl;

    os << "has_sparse_output: " << params.has_sparse_output << std::endl;
    os << "csr_data_offset: " << params.csr_data_offset << std::endl;
    os << "csr_indices_offset: " << params.csr_indices_offset << std::endl;
    os << "csr_indptr_offset: " << params.csr_indptr_offset << std::endl;

    os << "use_output_codebook: " << params.use_output_codebook << std::endl;
    for (int i = 0; i < NUM_CODEBOOK_ENTRIES - 1; i++) {
      os << "output_code[" << i << "]: " << params.output_code[i] << std::endl;
    }

    os << "is_dwc: " << params.is_dwc << std::endl;

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
    if (lhs.vector_output_offset != rhs.vector_output_offset) return false;
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

    if (lhs.vector_fetch_0_broadcast != rhs.vector_fetch_0_broadcast)
      return false;
    if (lhs.vector_fetch_1_broadcast != rhs.vector_fetch_1_broadcast)
      return false;
    if (lhs.vector_fetch_2_broadcast != rhs.vector_fetch_2_broadcast)
      return false;

    if (lhs.has_slicing != rhs.has_slicing) return false;
    if (lhs.vector_fetch_0_slice_dim != rhs.vector_fetch_0_slice_dim)
      return false;
    if (lhs.vector_fetch_0_slice_start != rhs.vector_fetch_0_slice_start)
      return false;
    if (lhs.vector_fetch_0_slice_end != rhs.vector_fetch_0_slice_end)
      return false;
    if (lhs.vector_fetch_0_slice_step != rhs.vector_fetch_0_slice_step)
      return false;

    if (lhs.has_permute != rhs.has_permute) return false;
    for (int i = 0; i < 6; i++) {
      if (lhs.vector_fetch_0_permute_dims[i] !=
          rhs.vector_fetch_0_permute_dims[i])
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

    if (lhs.head_size_lg2 != rhs.head_size_lg2) return false;
    if (lhs.transpose_for_scores != rhs.transpose_for_scores) return false;

    if (lhs.quantize_output_mx != rhs.quantize_output_mx) return false;
    if (lhs.mx_scale_offset != rhs.mx_scale_offset) return false;

    if (lhs.has_sparse_output != rhs.has_sparse_output) return false;
    if (lhs.csr_data_offset != rhs.csr_data_offset) return false;
    if (lhs.csr_indices_offset != rhs.csr_indices_offset) return false;
    if (lhs.csr_indptr_offset != rhs.csr_indptr_offset) return false;

    if (lhs.use_output_codebook != rhs.use_output_codebook) return false;
    for (int i = 0; i < NUM_CODEBOOK_ENTRIES - 1; i++) {
      if (lhs.output_code[i] != rhs.output_code[i]) return false;
    }
    if (lhs.is_dwc != rhs.is_dwc) return false;

    // If all members are equal, return true
    return true;
  }
};

struct ApproxUnitConfig {
#ifndef __SYNTHESIS__
  ApproxUnitConfig() {
    for (int i = 0; i < NUM_MAXES; i++) {
      maxes[i] = 0;
    }
    for (int i = 0; i < NUM_RANGES; i++) {
      for (int j = 0; j < NUM_COEFFS; j++) {
        ranges[i][j] = 0;
      }
    }
    clamp_min = 0;
    clamp_max = 0;
  }
#endif

  VECTOR_DATATYPE maxes[NUM_MAXES];
  VECTOR_DATATYPE ranges[NUM_RANGES][NUM_COEFFS];

  ac_int<1, false> clamp_min;
  ac_int<1, false> clamp_max;

  static const unsigned int width =
      16 * (NUM_MAXES + NUM_RANGES * NUM_COEFFS) + 2;

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    for (int i = 0; i < NUM_MAXES; i++) {
      m& maxes[i];
    }
    for (int i = 0; i < NUM_RANGES; i++) {
      for (int j = 0; j < NUM_COEFFS; j++) {
        m& ranges[i][j];
      }
    }
    m & clamp_min;
    m & clamp_max;
  }

  inline friend void sc_trace(sc_trace_file* tf, const ApproxUnitConfig& config,
                              const std::string& name) {
    // TODO
  }
#endif

  inline friend std::ostream& operator<<(std::ostream& os,
                                         const ApproxUnitConfig& config) {
#ifndef __SYNTHESIS__
    for (int i = 0; i < NUM_MAXES; i++) {
      os << "maxes[" << i << "]: " << config.maxes[i] << std::endl;
    }
    for (int i = 0; i < NUM_RANGES; i++) {
      for (int j = 0; j < NUM_COEFFS; j++) {
        os << "range[" << i << "][" << j << "]: " << config.ranges[i][j]
           << std::endl;
      }
    }
    os << "clamp_min: " << config.clamp_min << std::endl;
    os << "clamp_max: " << config.clamp_max << std::endl;
#endif
    return os;
  }

  inline friend bool operator==(const ApproxUnitConfig& lhs,
                                const ApproxUnitConfig& rhs) {
    for (int i = 0; i < NUM_MAXES; i++) {
      if (lhs.maxes[i] != rhs.maxes[i]) return false;
    }
    for (int i = 0; i < NUM_RANGES; i++) {
      for (int j = 0; j < NUM_COEFFS; j++) {
        if (lhs.ranges[i][j] != rhs.ranges[i][j]) return false;
      }
    }

    return true;
  }
};

struct OutlierFilterConfig {
#ifndef __SYNTHESIS__
  OutlierFilterConfig() {
    outlier_threshold = 0;
    dense_input_shape[0] = 0;
    dense_input_shape[1] = 0;
  }
#endif

  ac_int<16, false> outlier_threshold;
  ac_int<16, false> dense_input_shape[2];

  static const unsigned int width = 16 + 16 * 2;

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m & outlier_threshold;
    m& dense_input_shape[0];
    m& dense_input_shape[1];
  }

  // TODO
  inline friend void sc_trace(sc_trace_file* tf,
                              const OutlierFilterConfig& config,
                              const std::string& name) {}
#endif

  inline friend std::ostream& operator<<(std::ostream& os,
                                         const OutlierFilterConfig& config) {
    os << "outlier_threshold: " << config.outlier_threshold << std::endl;
    os << "dense_input_shape[0]: " << config.dense_input_shape[0] << std::endl;
    os << "dense_input_shape[1]: " << config.dense_input_shape[1] << std::endl;
    return os;
  }

  inline friend bool operator==(const OutlierFilterConfig& lhs,
                                const OutlierFilterConfig& rhs) {
    if (lhs.outlier_threshold != rhs.outlier_threshold) return false;
    if (lhs.dense_input_shape[0] != rhs.dense_input_shape[0]) return false;
    if (lhs.dense_input_shape[1] != rhs.dense_input_shape[1]) return false;
    return true;
  }
};

struct VectorInstructionConfig : BaseParams {
#ifndef __SYNTHESIS__
  VectorInstructionConfig() {
    num_inst = 0;
    config_loop_count = 0;
  }
#endif

  VectorInstructions inst[8];
  ac_int<4, false> num_inst;
  ac_int<16, false> config_loop_count;
  ApproxUnitConfig approx;
  OutlierFilterConfig outlier_filter;

  static const unsigned int width = VectorInstructions::width * 8 + 4 + 16 +
                                    ApproxUnitConfig::width +
                                    OutlierFilterConfig::width;

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    for (int j = 0; j < 8; j++) {
      m& inst[j].op_type;
    }
    for (int j = 0; j < 8; j++) {
      m& inst[j].inst_loop_count;
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

    m & num_inst;
    m & config_loop_count;

    for (int i = 0; i < NUM_MAXES; i++) {
      m & approx.maxes[i];
    }
    for (int i = 0; i < NUM_RANGES; i++) {
      for (int j = 0; j < NUM_COEFFS; j++) {
        m & approx.ranges[i][j];
      }
    }
    m & approx.clamp_min;
    m & approx.clamp_max;

    m & outlier_filter.outlier_threshold;
    m & outlier_filter.dense_input_shape[0];
    m & outlier_filter.dense_input_shape[1];
  }

  inline friend void sc_trace(sc_trace_file* tf,
                              const VectorInstructionConfig& params,
                              const std::string& name) {
    // TODO
  }
#endif

  inline friend std::ostream& operator<<(
      ostream& os, const VectorInstructionConfig& params) {
    for (int i = 0; i < params.num_inst; i++) {
      os << "Instruction " << i << std::endl;
      os << params.inst[i] << std::endl;
    }
    os << "num_inst: " << params.num_inst << std::endl;
    os << "config_loop_count: " << params.config_loop_count << std::endl;
    return os;
  }

  inline friend bool operator==(const VectorInstructionConfig& lhs,
                                const VectorInstructionConfig& rhs) {
    for (int i = 0; i < 8; i++) {
      if (!(lhs.inst[i] == rhs.inst[i])) return false;
    }
    if (lhs.num_inst != rhs.num_inst ||
        lhs.config_loop_count != rhs.config_loop_count)
      return false;
    if (!(lhs.approx == rhs.approx)) return false;

    return true;
  }
};

struct DwCParams : BaseParams {
#ifndef __SYNTHESIS__
  DwCParams() {
    input_offset = 0;
    input_scale_offset = 0;
    weight_offset = 0;
    weight_scale_offset = 0;
    bias_offset = 0;
    output_offset = 0;

    for (int i = 0; i < 2; i++) {  // Outer -> Inner, Y X C
      for (int j = 0; j < 3; j++) {
        loops[i][j] = 0;
      }
    }

    for (int i = 0; i < 3; i++) {  // Y X C
      bounds[i] = 0;
    }

    stride = 1;

    for (int i = 0; i < 2; i++) {  // Y X
      for (int j = 0; j < 2; j++) {
        padding[i][j] = 0;
      }
    }

    fast_forward_mode = 0;  // 0: normal, 1: fast forward
    block_size = 0;
    use_mx = 0;
  }
#endif

  ac_int<ADDRESS_WIDTH, false> input_offset;
  ac_int<ADDRESS_WIDTH, false> input_scale_offset;
  ac_int<ADDRESS_WIDTH, false> weight_offset;
  ac_int<ADDRESS_WIDTH, false> weight_scale_offset;
  ac_int<ADDRESS_WIDTH, false> bias_offset;
  ac_int<ADDRESS_WIDTH, false> output_offset;

  // systolic array loop
  ac_int<10, false> loops[2][3];
  ac_int<10, false> bounds[3];
  ac_int<10, false> outloops[3];  // Y X1 X0

  ac_int<4, false> stride;
  ac_int<7, true> padding[2][2];       // Y X
  ac_int<1, false> fast_forward_mode;  // 0: normal, 1: fast forward
  ac_int<3, false> block_size;         // 2^bs
  ac_int<1, false> use_mx;             // 0: no mx, 1: use mx

  static const unsigned int width =
      (4 + 2) * ADDRESS_WIDTH /* OFFSETS */ + (6 + 3) * 10 /* Loops */ +
      (3) * 10 /* bounds */ + 4 /* stride */ + (2 * 2) * 7 /* padding */ +
      1 /* MODE */ + 3 /* block_size */ + 1 /* use_mx */;

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m & input_offset;
    m & input_scale_offset;
    m & weight_offset;
    m & weight_scale_offset;
    m & bias_offset;
    m & output_offset;

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        m& loops[i][j];
      }
    }

    for (int i = 0; i < 3; i++) {
      m& bounds[i];
    }

    for (int i = 0; i < 3; i++) {
      m& outloops[i];
    }

    m & stride;

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        m& padding[i][j];
      }
    }

    m & fast_forward_mode;
    m & block_size;
    m & use_mx;
  }

  inline friend void sc_trace(sc_trace_file* tf, const DwCParams& params,
                              const std::string& name) {}
#endif

  inline friend std::ostream& operator<<(std::ostream& os,
                                         const DwCParams& params) {
    os << "input_offset: " << params.input_offset << std::endl;
    os << "input_scale_offset: " << params.input_scale_offset << std::endl;
    os << "weight_offset: " << params.weight_offset << std::endl;
    os << "weight_scale_offset: " << params.weight_scale_offset << std::endl;
    os << "bias_offset: " << params.bias_offset << std::endl;
    os << "output_offset: " << params.output_offset << std::endl;

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        os << "loops[" << i << "][" << j << "]: " << params.loops[i][j]
           << std::endl;
      }
    }

    for (int i = 0; i < 3; i++) {
      os << "bounds[" << i << "]: " << params.bounds[i] << std::endl;
    }

    for (int i = 0; i < 3; i++) {
      os << "outloops[" << i << "]: " << params.outloops[i] << std::endl;
    }

    os << "stride: " << params.stride << std::endl;

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        os << "padding[" << i << "][" << j << "]: " << params.padding[i][j]
           << std::endl;
      }
    }

    os << "fast_forward_mode: " << params.fast_forward_mode << std::endl;
    os << "block_size: " << params.block_size << std::endl;
    os << "use_mx: " << params.use_mx << std::endl;
    return os;
  }

  inline friend bool operator==(const DwCParams& lhs, const DwCParams& rhs) {
    if (lhs.input_offset != rhs.input_offset ||
        lhs.input_scale_offset != rhs.input_scale_offset ||
        lhs.weight_offset != rhs.weight_offset ||
        lhs.weight_scale_offset != rhs.weight_scale_offset ||
        lhs.bias_offset != rhs.bias_offset ||
        lhs.output_offset != rhs.output_offset)
      return false;

    // Compare the 2D arrays
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        if (lhs.loops[i][j] != rhs.loops[i][j]) return false;
      }
    }

    // Compare other members
    for (int i = 0; i < 3; i++) {
      if (lhs.bounds[i] != rhs.bounds[i]) return false;
    }

    for (int i = 0; i < 3; i++) {
      if (lhs.outloops[i] != rhs.outloops[i]) return false;
    }

    if (lhs.stride != rhs.stride) return false;

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        if (lhs.padding[i][j] != rhs.padding[i][j]) return false;
      }
    }

    if (lhs.fast_forward_mode != rhs.fast_forward_mode) return false;
    if (lhs.block_size != rhs.block_size) return false;
    if (lhs.use_mx != rhs.use_mx) return false;
    // If all members are equal, return true
    return true;
  }
};
