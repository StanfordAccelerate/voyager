#pragma once

#include <sstream>

#include "ArchitectureParams.h"
#include "spdlog/spdlog.h"
#include "src/Params.h"
#include "test/common/Network.h"
#include "test/common/Tiling.h"
#include "test/common/Utils.h"
#include "test/compiler/proto/param.pb.h"
#include "test/toolchain/Common.h"

void map_spmm(const codegen::Operation& operation,
              std::deque<BaseParams*>& mapped_params) {
  MatrixParams* spmm_param = new MatrixParams;

  const auto op_list = get_op_list(operation);
  const auto spmm_op = op_list[0];

  const auto weight_tensor = spmm_op.kwargs().at("B").tensor();
  const auto indptr_tensor = spmm_op.kwargs().at("indptr").tensor();
  const auto indices_tensor = spmm_op.kwargs().at("indices").tensor();
  const auto data_tensor = spmm_op.kwargs().at("data").tensor();

  spmm_param->is_spmm = true;

  float* indptr_array = read_constant_param(indptr_tensor);

  // indptr is in y_loop_idx[0]
  // indices is in x_loop_idx[0]
  // K is in weight_loop_idx[0]
  const int indptr_len = get_size(indptr_tensor);
  const int indices_len = indptr_array[indptr_len - 1];
  const int K = get_shape(weight_tensor)[1];
  const int k0 = K / OC_DIMENSION;

  Tiling tiling = {
      .loops = {{indices_len, indptr_len, k0, 1, 1, 1}, {1, 1, 1, 1, 1, 1}},
      .x_loop_idx = {0, 5},
      .y_loop_idx = {1, 4},
      .reduction_loop_idx = {1, 2},
      .weight_loop_idx = {2, 3},
  };

  std::ostringstream oss;
  oss << tiling;
  spdlog::info("Using tiling: \n{}\n", oss.str());

  spmm_param->spmm_indptr_offset = get_address(indptr_tensor);
  spmm_param->spmm_indices_offset = get_address(indices_tensor);
  spmm_param->spmm_data_offset = get_address(data_tensor);
  spmm_param->weight_offset = get_address(weight_tensor);

  spmm_param->weight_dtype =
      get_index_from_type_name<WEIGHT_DATATYPE>(weight_tensor.dtype());
  spmm_param->use_weight_codebook = spmm_op.kwargs().contains("B_code");
  if (spmm_param->use_weight_codebook) {
    const auto code = spmm_op.kwargs().at("B_code").tensor();
    const auto size = get_size(code);

    float* weight_code = read_constant_param(code);
    for (int i = 0; i < size; i++) {
      SA_WEIGHT_TYPE value = weight_code[i];
      spmm_param->weight_code[i] = value.bits_rep();
    }

    delete[] weight_code;
  }

  spmm_param->is_mx_op = spmm_op.kwargs().contains("B_scale");
  if (spmm_param->is_mx_op) {
    const int block_size = spmm_op.kwargs().at("block_size").int_value();
    assert(block_size == std::max(IC_DIMENSION, OC_DIMENSION));

    const auto scale_tensor = spmm_op.kwargs().at("B_scale").tensor();
    spmm_param->weight_scale_offset = get_address(scale_tensor);
  }

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 6; j++) {
      spmm_param->loops[i][j] = tiling.loops[i][j];
    }
    spmm_param->x_loop_idx[i] = tiling.x_loop_idx[i];
    spmm_param->y_loop_idx[i] = tiling.y_loop_idx[i];
    spmm_param->reduction_loop_idx[i] = tiling.reduction_loop_idx[i];
    spmm_param->weight_loop_idx[i] = tiling.weight_loop_idx[i];
  }

  // vector insturctions, just passing through the output for now
  VectorParams* vector_params = new VectorParams;

  const auto output = get_op_outputs(operation).back();
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

  // outermost loop, K
  vector_params->output_loops[1][0] = k0;
  // second loop, indptr (a.k.a the number of rows) -1 to rid of the extra zero
  vector_params->output_loops[1][1] = indptr_len - 1;
  // only support matrices for now, no third loop
  vector_params->output_loops[1][2] = 1;
  vector_params->output_y_loop_idx[1] = 2;
  vector_params->output_x_loop_idx[1] = 1;
  vector_params->output_k_loop_idx[1] = 0;

  vector_params->output_dtype =
      get_index_from_type_name<OUTPUT_DATATYPES>(output.dtype());

  VectorInstructions inst;
  inst.op_type = VectorInstructions::vector;
  inst.inst_loop_count = k0 * (indptr_len - 1);

  inst.vector_op0_src0 = VectorInstructions::from_spmm_unit;
  inst.vdest = VectorInstructions::to_output;

  VectorInstructionConfig* vector_instruction_config =
      new VectorInstructionConfig;
  vector_instruction_config->inst[0] = inst;
  vector_instruction_config->num_inst = 1;
  vector_instruction_config->config_loop_count = 1;

  mapped_params.push_back(spmm_param);
  mapped_params.push_back(vector_params);
  mapped_params.push_back(vector_instruction_config);
}
