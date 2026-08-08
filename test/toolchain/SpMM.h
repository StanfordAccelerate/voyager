#pragma once

#include <sstream>

#include "ArchitectureParams.h"
#include "spdlog/spdlog.h"
#include "src/Params.h"
#include "test/common/GraphUtils.h"
#include "test/common/Tiling.h"
#include "test/common/Utils.h"
#include "test/toolchain/Common.h"

void map_spmm(const voyager::Operation& operation, const ScalarEnv& env,
              std::deque<BaseParams*>& mapped_params, bool is_fused) {
  MatrixParams* spmm_param = new MatrixParams;

  const auto& op_list = get_prim_ops(operation);
  const auto& spmm_op = get_anchor_op(operation);

  bool is_matmul = spmm_op.target().find("matmul") != std::string::npos;
  std::string weight_key = is_matmul ? "other" : "weight";
  const auto weight_tensor = resolve(spmm_op, weight_key, env);
  const auto indptr_tensor = resolve(spmm_op, "A_indptr", env);
  const auto indices_tensor = resolve(spmm_op, "A_indices", env);
  const auto data_tensor = resolve(spmm_op, "A_data", env);

  spmm_param->is_spmm = true;

  // Only the indptr's length is needed here. Its contents are a run-time
  // value the outlier quantize writes into the scratchpad, not a constant
  // codegen dumps, so there is nothing on disk to read.
  // indptr is in y_loop_idx[0]
  // indices is in x_loop_idx[0]
  // K is in weight_loop_idx[0]
  const int indptr_len = get_size(indptr_tensor) - 1;
  const int K = get_shape(weight_tensor)[1];
  const int k0 = K / OC_DIMENSION;
  Tiling tiling;

  // for fused SPMM, the sparse unit needs to match the tiling of the matrix op
  if (is_fused) {
    tiling = get_tiling(operation, env);
  } else {
    tiling = {
        .loops = {{1, 2, 1, 1, 1, 1}, {k0, indptr_len / 2, 1, 1, 1, 1}},
        .x_loop_idx = {1, 1},
        .y_loop_idx = {2, 2},
        .reduction_loop_idx = {3, 3},
        .weight_loop_idx = {0, 0},
    };
  }

  std::ostringstream oss;
  oss << tiling;
  spdlog::info("Using tiling: \n{}\n", oss.str());

  spmm_param->spmm_indptr_offset = get_address(indptr_tensor);
  spmm_param->spmm_indices_offset = get_address(indices_tensor);
  spmm_param->spmm_data_offset = get_address(data_tensor);
  spmm_param->weight_offset = get_address(weight_tensor);

  spmm_param->weight_dtype =
      get_index_from_type_name<WEIGHT_DATATYPE>(weight_tensor.dtype);
  spmm_param->use_weight_codebook = has_arg(spmm_op, "weight_code");
  if (spmm_param->use_weight_codebook) {
    const auto code = resolve(spmm_op, "weight_code", env);
    const auto size = get_size(code);

    float* weight_code = read_constant_param(code);
    for (int i = 0; i < size; i++) {
      SA_WEIGHT_TYPE value = weight_code[i];
      spmm_param->weight_code[i] = value.bits_rep();
    }

    delete[] weight_code;
  }

  spmm_param->is_mx_op = has_arg(spmm_op, "weight_scale");
  if (spmm_param->is_mx_op) {
    const int block_size = arg_int(spmm_op, "block_size", env);
    assert(block_size == std::max(IC_DIMENSION, OC_DIMENSION));

    const auto scale_tensor = resolve(spmm_op, "weight_scale", env);
    spmm_param->weight_scale_offset = get_address(scale_tensor);
  }

  for (int i = 0; i < 2; i++) {
    int loop_index = 0;
    for (int j = 0; j < 6; j++) {
      // ignore anything but the x and weight loops
      if (j == tiling.x_loop_idx[i]) {
        spmm_param->loops[i][loop_index] = tiling.loops[i][j];
        spmm_param->x_loop_idx[i] = loop_index;
        loop_index++;
      }
      if (j == tiling.weight_loop_idx[i]) {
        spmm_param->loops[i][loop_index] = tiling.loops[i][j];
        spmm_param->weight_loop_idx[i] = loop_index;
        loop_index++;
      }
    }
    spmm_param->y_loop_idx[i] = tiling.y_loop_idx[i];
    spmm_param->reduction_loop_idx[i] = tiling.reduction_loop_idx[i];
  }

  // compute the reduction dimension for weight scale addressing
  int C = 1;
  for (int i = 0; i < 2; i++) {
    C *= tiling.loops[i][tiling.reduction_loop_idx[i]];
  }

  spmm_param->weight_addr_loops[0][0] = C;
  spmm_param->weight_addr_reduction_loop_idx[0] = 0;

  mapped_params.push_back(spmm_param);

  // for fused spmm, the vector instructions will be handled in matrix mapping
  if (is_fused) {
    return;
  }

  // vector instructions, just passing through the output for now
  VectorParams* vector_params = new VectorParams;

  const auto output = resolve_outputs(operation, env).back();
  vector_params->vector_output_offset = get_address(output);

  // set output mode
  vector_params->output_mode = 1;

  // outer loops not used in SPMM
  for (int i = 0; i < 3; i++) {
    vector_params->output_loops[0][i] = 1;
  }

  vector_params->output_y_loop_idx[0] = 0;
  vector_params->output_x_loop_idx[0] = 1;
  vector_params->output_k_loop_idx[0] = 2;

  vector_params->output_y_loop_idx[1] = 2;
  vector_params->output_x_loop_idx[1] = 1;
  vector_params->output_k_loop_idx[1] = 0;

  // outermost loop, K
  vector_params->output_loops[0][0] = 1;
  vector_params->output_loops[0][1] = 2;
  vector_params->output_loops[0][2] = 1;

  vector_params->output_loops[1][0] = k0;
  vector_params->output_loops[1][1] = indptr_len / 2;
  // only support matrices for now, no third loop
  vector_params->output_loops[1][2] = 1;

  vector_params->output_dtype =
      get_index_from_type_name<OUTPUT_DATATYPES>(output.dtype);

  VectorInstructions inst;
  inst.op_type = VectorInstructions::vector;
  inst.inst_loop_count = k0 * indptr_len;

  inst.vector_op0_src0 = VectorInstructions::from_spmm_unit;
  inst.vdest = VectorInstructions::to_output;

  VectorInstructionConfig* vector_instruction_config =
      new VectorInstructionConfig;
  vector_instruction_config->inst[0] = inst;
  vector_instruction_config->num_inst = 1;
  vector_instruction_config->config_loop_count = 1;

  mapped_params.push_back(vector_params);
  mapped_params.push_back(vector_instruction_config);
}
