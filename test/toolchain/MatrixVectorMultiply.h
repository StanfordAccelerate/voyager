#pragma once

#include "test/toolchain/Common.h"

void map_matrix_vector_multiply(const voyager::Operation& operation,
                                const ScalarEnv& env,
                                std::deque<BaseParams*>& mapped_params) {
  const auto& op_list = get_prim_ops(operation);
  const auto& matrix_op = get_anchor_op(operation);

  const auto input = resolve(matrix_op, "input", env);
  const auto output = resolve_outputs(operation, env).back();

  bool is_matmul = matrix_op.target().find("matmul") != std::string::npos;
  std::string key = is_matmul ? "other" : "weight";
  const auto weight = resolve(matrix_op, key, env);
  bool has_bias = has_arg(matrix_op, "bias");

  std::vector<const voyager::PrimOp*> vector_ops;
  bool after_anchor = false;
  for (const voyager::PrimOp* op : op_list) {
    if (!after_anchor) {
      if (op->name() == matrix_op.name()) after_anchor = true;
      continue;
    }
    vector_ops.push_back(op);
  }

  const bool has_vector_pipeline = has_bias || !vector_ops.empty();

  const auto weight_shape = get_shape(weight);
  if (weight_shape.size() < 2) {
    throw std::runtime_error("MVM weight must have at least two dimensions.");
  }
  int output_dim = weight_shape[weight_shape.size() - 2];
  int reduction_dim = weight_shape.back();

  int packing_factor = OC_DIMENSION / VECTOR_UNIT_WIDTH;

  VectorParams* vector_params = new VectorParams;
  VectorInstructionConfig* vector_instruction_config =
      new VectorInstructionConfig;

  // input is a vector of size reduction_dim
  vector_params->vector_fetch_0_offset = get_address(input);
  vector_params->vector_fetch_0_mode = 2;
  vector_params->vector_fetch_0_broadcast = 0b010000;

  int input_dtype = get_index_from_type_name<VU_INPUT_TYPES>(input.dtype);
  int input_dtype_width = get_type_width<VU_INPUT_TYPES>(input_dtype);
  int input_fetch_width = OC_DIMENSION * input_dtype_width;

  vector_params->vector_fetch_0_dtype = input_dtype;
  vector_params->vector_fetch_0_stride = OC_DIMENSION;
  vector_params->vector_fetch_0_burst_size = input_fetch_width / 8;
  vector_params->vector_fetch_0_num_beats = input_fetch_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_0_packing_factor = packing_factor;

  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_0_loops[0][i] = 1;
  }
  vector_params->vector_fetch_0_loops[1][0] = 1;
  vector_params->vector_fetch_0_loops[1][1] = output_dim;
  vector_params->vector_fetch_0_loops[1][2] = reduction_dim / OC_DIMENSION;

  // weight is a matrix of output_dim x reduction_dim
  vector_params->vector_fetch_1_offset = get_address(weight);
  vector_params->vector_fetch_1_mode = true;

  int weight_dtype = get_index_from_type_name<VU_INPUT_TYPES>(weight.dtype);
  int weight_dtype_width = get_type_width<VU_INPUT_TYPES>(weight_dtype);
  int weight_fetch_width = OC_DIMENSION * weight_dtype_width;

  vector_params->vector_fetch_1_dtype = weight_dtype;
  vector_params->vector_fetch_1_stride = OC_DIMENSION;
  vector_params->vector_fetch_1_burst_size = weight_fetch_width / 8;
  vector_params->vector_fetch_1_num_beats = weight_fetch_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_1_packing_factor = packing_factor;

  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_1_loops[0][i] = 1;
  }
  vector_params->vector_fetch_1_loops[1][0] = 1;
  vector_params->vector_fetch_1_loops[1][1] = output_dim;
  vector_params->vector_fetch_1_loops[1][2] = reduction_dim / OC_DIMENSION;

  for (int i = 0; i < 2; i++) {
    vector_params->vector_fetch_1_y_loop_idx[i] = 0;
    vector_params->vector_fetch_1_x_loop_idx[i] = 1;
    vector_params->vector_fetch_1_k_loop_idx[i] = 2;
  }

  auto configure_vector_fetch_2 = [&](const Tensor& tensor) {
    int nonzero_dims = 0;
    for (const int dim : tensor.shape) {
      if (dim != 1) nonzero_dims++;
    }

    vector_params->vector_fetch_2_offset = get_address(tensor);
    vector_params->vector_fetch_2_mode = true;
    vector_params->vector_fetch_2_broadcast = nonzero_dims == 1 ? 0b011 : 0b000;

    const int dtype = get_index_from_type_name<VU_INPUT_TYPES>(tensor.dtype);
    const int dtype_width = get_type_width<VU_INPUT_TYPES>(dtype);
    const int fetch_width = OC_DIMENSION * dtype_width;

    vector_params->vector_fetch_2_dtype = dtype;
    vector_params->vector_fetch_2_stride = OC_DIMENSION;
    vector_params->vector_fetch_2_burst_size = fetch_width / 8;
    vector_params->vector_fetch_2_num_beats =
        (fetch_width + OC_PORT_WIDTH - 1) / OC_PORT_WIDTH;
    vector_params->vector_fetch_2_packing_factor = packing_factor;

    for (int i = 0; i < 3; i++) {
      vector_params->vector_fetch_2_loops[0][i] = 1;
    }
    vector_params->vector_fetch_2_loops[1][0] = 1;
    vector_params->vector_fetch_2_loops[1][1] = 1;
    vector_params->vector_fetch_2_loops[1][2] = output_dim / OC_DIMENSION;

    for (int i = 0; i < 2; i++) {
      vector_params->vector_fetch_2_y_loop_idx[i] = 0;
      vector_params->vector_fetch_2_x_loop_idx[i] = 1;
      vector_params->vector_fetch_2_k_loop_idx[i] = 2;
    }
  };

  // output
  vector_params->vector_output_offset = get_address(output);
  vector_params->output_dtype =
      get_index_from_type_name<OUTPUT_DATATYPES>(output.dtype);

  for (int i = 0; i < 3; i++) {
    vector_params->output_loops[0][i] = 1;
  }
  vector_params->output_loops[1][0] = 1;
  vector_params->output_loops[1][1] = 1;
  vector_params->output_loops[1][2] = output_dim / OC_DIMENSION;

  for (int i = 0; i < 2; i++) {
    vector_params->output_y_loop_idx[i] = 0;
    vector_params->output_x_loop_idx[i] = 1;
    vector_params->output_k_loop_idx[i] = 2;
  }

  // Start reduction engine
  VectorInstructions inst0;
  inst0.op_type = VectorInstructions::reduction;
  inst0.inst_loop_count = VECTOR_UNIT_WIDTH / REDUCER_WIDTH;
  inst0.reduce_count = reduction_dim / REDUCER_WIDTH;
  inst0.reduce_op = VectorInstructions::radd;
  inst0.rdest = has_vector_pipeline ? VectorInstructions::to_pipeline
                                    : VectorInstructions::to_memory;
  vector_instruction_config->inst[0] = inst0;

  // Multiply input vector with weight matrix. Reducer takes
  // reduction_dim / OC_DIMENSION cycles to do the complete reduction,
  // and OC_DIMENSION cycles to fill up the entire vector
  VectorInstructions inst1;
  inst1.op_type = VectorInstructions::vector;
  inst1.inst_loop_count = reduction_dim;
  inst1.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  inst1.vector_op0_src1 = VectorInstructions::from_vector_fetch_1;
  inst1.vector_op0 = VectorInstructions::op0_mul;
  inst1.vdest = VectorInstructions::to_reduce;
  vector_instruction_config->inst[1] = inst1;

  if (has_vector_pipeline) {
    VectorInstructions pipeline_inst;
    pipeline_inst.op_type = VectorInstructions::vector;
    pipeline_inst.inst_loop_count = 1;
    pipeline_inst.vector_op0_src0 = VectorInstructions::from_vector_reducer;
    pipeline_inst.vdest = VectorInstructions::to_output;

    bool vector_fetch_2_in_use = false;
    int initial_stage = 0;
    if (has_bias) {
      configure_vector_fetch_2(resolve(matrix_op, "bias", env));
      vector_fetch_2_in_use = true;
      pipeline_inst.vector_op2_src1 = VectorInstructions::from_vector_fetch_2;
      pipeline_inst.vector_op2 = VectorInstructions::op2_add;
      initial_stage = 3;
    }

    auto map_tensor_operand =
        [&](const voyager::PrimOp& op, const ScalarEnv& env,
            const std::string& other_key, int stage, VectorInstructions& inst) {
          const Tensor& self = resolve(op, "input", env);
          const Tensor& tensor = resolve(op, other_key, env);
          const Tensor& tensor_to_load = tensor.materialized ? tensor : self;

          if (stage == 0) {
            throw std::runtime_error(
                "The MVM dot product occupies vector fetch 1, so a fused "
                "non-scalar side operand cannot use pipeline stage 0.");
          }
          if (vector_fetch_2_in_use) {
            throw std::runtime_error(
                "Fused MVM operations require more than one vector fetch 2 "
                "operand.");
          }

          vector_fetch_2_in_use = true;
          configure_vector_fetch_2(tensor_to_load);
          if (stage == 2) {
            inst.vector_op2_src1 = VectorInstructions::from_vector_fetch_2;
          } else {
            inst.vector_op3_src1 = VectorInstructions::from_vector_fetch_2;
          }
        };

    auto is_stage_available = [](const voyager::PrimOp& op,
                                 const ScalarEnv& env, int stage) {
      if (stage != 0 || (!has_arg(op, "other") &&
                         strip_namespace(op.target()) != "quantize")) {
        return true;
      }

      const std::string other_key =
          strip_namespace(op.target()) == "quantize" ? "scale" : "other";
      if (!has_tensor(op, other_key)) return true;
      if (strip_namespace(op.target()) == "quantize" &&
          !resolve(op, "scale", env).materialized) {
        return true;
      }
      return get_size(resolve(op, other_key, env)) == 1;
    };

    map_vector_pipeline_ops(operation, env, vector_ops, vector_params,
                            vector_instruction_config, pipeline_inst,
                            mapped_params, map_tensor_operand,
                            is_stage_available, initial_stage);
    vector_instruction_config->inst[2] = pipeline_inst;
  }

  vector_instruction_config->num_inst = has_vector_pipeline ? 3 : 2;
  vector_instruction_config->config_loop_count = output_dim / VECTOR_UNIT_WIDTH;

  mapped_params.push_back(vector_params);
  mapped_params.push_back(vector_instruction_config);
}
