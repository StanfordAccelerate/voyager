#pragma once

#include "test/common/Tiling.h"
#include "test/toolchain/ApproximationConstants.h"
#include "test/toolchain/Common.h"

inline bool are_broadcastable(const std::vector<int>& shape1,
                              const std::vector<int>& shape2) {
  size_t len1 = shape1.size();
  size_t len2 = shape2.size();
  size_t min_len = std::min(len1, len2);

  for (size_t i = 0; i < min_len; ++i) {
    int dim1 = shape1[len1 - 1 - i];
    int dim2 = shape2[len2 - 1 - i];
    if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
      return false;
    }
  }
  return true;
}

inline std::vector<int> broadcast_shape(std::vector<int>& shape1,
                                        std::vector<int>& shape2) {
  if (!are_broadcastable(shape1, shape2)) {
    throw std::invalid_argument("Shapes are not broadcastable");
  }

  int n1 = shape1.size();
  int n2 = shape2.size();
  int max_size = std::max(n1, n2);
  std::vector<int> result_shape(max_size);

  for (int i = 1; i <= max_size; i++) {
    int dim1 = n1 - i >= 0 ? shape1[n1 - i] : 1;
    int dim2 = n2 - i >= 0 ? shape2[n2 - i] : 1;
    result_shape[max_size - i] = std::max(dim1, dim2);
  }

  for (int i = max_size - 1; i >= 0; --i) {
    if (result_shape[i] == 1) {
      result_shape.erase(result_shape.begin() + i);
      if (i >= max_size - n1)
        shape1.erase(shape1.begin() + (i - (max_size - n1)));
      if (i >= max_size - n2)
        shape2.erase(shape2.begin() + (i - (max_size - n2)));
    }
  }

  return result_shape;
}

void set_vector_fetch_1(const codegen::Tensor& tensor,
                        std::vector<int> output_shape,
                        VectorParams* vector_params) {
  vector_params->vector_fetch_1_offset = get_address(tensor);
  vector_params->vector_fetch_1_mode = true;

  auto input_shape = get_shape(tensor);
  squeeze_front_ones(input_shape);
  pad_shape_to_ndim(input_shape, 3);

  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_1_broadcast[i] =
        input_shape[i] == 1 && output_shape[i] != 1;
  }

  vector_params->vector_fetch_1_dtype =
      get_index_from_type_name<VU_INPUT_TYPES>(tensor.dtype());
  const int dtype_width =
      get_type_width<VU_INPUT_TYPES>(vector_params->vector_fetch_1_dtype);
  int fetch_width = OC_DIMENSION * dtype_width;
  vector_params->vector_fetch_1_burst_size = fetch_width / 8;
  vector_params->vector_fetch_1_num_beats =
      (fetch_width + OC_PORT_WIDTH - 1) / OC_PORT_WIDTH;
  vector_params->vector_fetch_1_packing_factor =
      OC_DIMENSION / VECTOR_UNIT_WIDTH;

  pad_shape_to_ndim(output_shape, 6);
  output_shape = adjust_loop_indices(output_shape, OC_DIMENSION);

  vector_params->vector_fetch_1_loops[0][0] = output_shape[0];
  vector_params->vector_fetch_1_loops[0][1] = output_shape[1];
  vector_params->vector_fetch_1_loops[0][2] = output_shape[2];
  vector_params->vector_fetch_1_loops[1][0] = output_shape[3];
  vector_params->vector_fetch_1_loops[1][1] = output_shape[4];
  vector_params->vector_fetch_1_loops[1][2] = output_shape[5] / OC_DIMENSION;

  for (int i = 0; i < 2; i++) {
    vector_params->vector_fetch_1_y_loop_idx[i] = 0;
    vector_params->vector_fetch_1_x_loop_idx[i] = 1;
    vector_params->vector_fetch_1_k_loop_idx[i] = 2;
  }

  VECTOR_DATATYPE scale = get_tensor_scalar_scale(tensor);
  vector_params->vector_fetch_1_dq_scale = scale.bits_rep();
}

void set_vector_fetch_2(const codegen::Tensor& tensor,
                        std::vector<int> output_shape,
                        VectorParams* vector_params) {
  vector_params->vector_fetch_2_offset = get_address(tensor);
  vector_params->vector_fetch_2_mode = true;

  auto input_shape = get_shape(tensor);
  squeeze_front_ones(input_shape);
  pad_shape_to_ndim(input_shape, 3);

  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_2_broadcast[i] = input_shape[i] == 1;
  }

  vector_params->vector_fetch_2_dtype =
      get_index_from_type_name<VU_INPUT_TYPES>(tensor.dtype());
  const int dtype_width =
      get_type_width<VU_INPUT_TYPES>(vector_params->vector_fetch_2_dtype);
  int fetch_width = OC_DIMENSION * dtype_width;
  vector_params->vector_fetch_2_burst_size = fetch_width / 8;
  vector_params->vector_fetch_2_num_beats =
      (fetch_width + OC_PORT_WIDTH - 1) / OC_PORT_WIDTH;
  vector_params->vector_fetch_2_packing_factor =
      OC_DIMENSION / VECTOR_UNIT_WIDTH;

  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_2_loops[0][i] = 1;
  }

  pad_shape_to_ndim(output_shape, 3);
  vector_params->vector_fetch_2_loops[1][0] = output_shape[0];
  vector_params->vector_fetch_2_loops[1][1] = output_shape[1];
  vector_params->vector_fetch_2_loops[1][2] =
      output_shape.back() / OC_DIMENSION;

  for (int i = 0; i < 2; i++) {
    vector_params->vector_fetch_2_y_loop_idx[i] = 0;
    vector_params->vector_fetch_2_x_loop_idx[i] = 1;
    vector_params->vector_fetch_2_k_loop_idx[i] = 2;
  }

  VECTOR_DATATYPE scale = get_tensor_scalar_scale(tensor);
  vector_params->vector_fetch_2_dq_scale = scale.bits_rep();
}

void set_vector_immediate(const float scalar, const int stage,
                          const std::string opcode, VectorInstructions& inst) {
  VECTOR_DATATYPE immediate = scalar;

  if (opcode == "div" || opcode == "div_") {
    immediate = 1.0 / immediate;
  }

  if (stage == 0) {
    inst.vector_op0_src1 = VectorInstructions::from_immediate_0;
    inst.immediate0 = immediate.bits_rep();
  } else if (stage == 2) {
    inst.vector_op2_src1 = VectorInstructions::from_immediate_1;
    inst.immediate1 = immediate.bits_rep();
  } else {
    inst.vector_op3_src1 = VectorInstructions::from_immediate_2;
    inst.immediate2 = immediate.bits_rep();
  }
}

void map_vector_operations(const codegen::Operation& param,
                           std::deque<BaseParams*>& mapped_params) {
  VectorParams* vector_params = new VectorParams;
  VectorInstructionConfig* vector_instruction_config =
      new VectorInstructionConfig;

  // Support a maximum buffer size of 1024
  constexpr int BUFSIZE =
      std::min({1024 / OC_DIMENSION, OC_DIMENSION, VECTOR_UNIT_WIDTH});

  auto op_list = get_op_list(param);

  const auto input = op_list[0].kwargs().at("input").tensor();
  vector_params->vector_fetch_0_offset = get_address(input);
  vector_params->vector_fetch_0_mode = 2;
  vector_params->vector_fetch_0_dtype =
      get_index_from_type_name<VU_INPUT_TYPES>(input.dtype());

  // Use the original shape without permute/slice
  auto input_shape = get_shape(input, false);
  const int input_ndim = input_shape.size();

  if (input.has_reshape() || is_dma_op(op_list[0].target())) {
    for (const auto dim : input_shape) {
      if (dim > MAX_LOOP_VALUE) {
        spdlog::error("ERROR: input shape dimension is greater than {}: ",
                      MAX_LOOP_VALUE);
        print_shape(input_shape);
        throw std::invalid_argument("Unsupported input shape dimension!");
      }
    }
  } else {
    input_shape = split_loops(input_shape, MAX_LOOP_VALUE);
    input_shape = adjust_loop_indices(input_shape, OC_DIMENSION);
  }

  // Pad the shape to 6 dimensions with 1s
  int padded_dims = pad_shape_to_ndim(input_shape, 6);

  vector_params->vector_fetch_0_loops[0][0] = input_shape[0];
  vector_params->vector_fetch_0_loops[0][1] = input_shape[1];
  vector_params->vector_fetch_0_loops[0][2] = input_shape[2];
  vector_params->vector_fetch_0_loops[1][0] = input_shape[3];
  vector_params->vector_fetch_0_loops[1][1] = input_shape[4];
  vector_params->vector_fetch_0_loops[1][2] = input_shape[5] / OC_DIMENSION;

  codegen::OpOverload reshape_op;
  if (input.has_reshape()) {
    reshape_op = input.reshape();
  } else if (is_dma_op(op_list[0].target())) {
    reshape_op = op_list[0];
    op_list.erase(op_list.begin());
  }

  const auto reshape_kwargs = reshape_op.kwargs();

  if (reshape_op.target() == "slice") {
    vector_params->has_slicing = true;

    auto start = reshape_kwargs.at("start").int_value();
    auto end = reshape_kwargs.at("end").int_value();
    auto step = reshape_kwargs.at("step").int_value();
    auto dim = reshape_kwargs.at("dim").int_value();

    auto shape = get_shape(input, false);

    dim = dim < 0 ? dim + shape.size() : dim;
    end = end > shape[dim] ? shape[dim] : end;

    vector_params->vector_fetch_0_slice_dim = dim + padded_dims;
    vector_params->vector_fetch_0_slice_start = start;
    vector_params->vector_fetch_0_slice_step = step;
    // vector_fetch_0_slice_end is the last index normalized by step
    vector_params->vector_fetch_0_slice_end = (end - start + step - 1) / step;

    // Last dimension needs to be scaled by OC_DIMENSION
    if (vector_params->vector_fetch_0_slice_dim == 5) {
      if (start % OC_DIMENSION != 0 || end % OC_DIMENSION != 0) {
        throw std::invalid_argument(
            "Slice indices for the last dimension must be multiples of "
            "OC_DIMENSION!");
      }

      vector_params->vector_fetch_0_slice_start /= OC_DIMENSION;
      vector_params->vector_fetch_0_slice_end /= OC_DIMENSION;
    }
  } else if (reshape_op.target() == "permute") {
    const auto int_list = reshape_kwargs.at("dims").int_list().values();
    std::vector<int> dims(int_list.begin(), int_list.end());
    const int ndim = input.shape_size();

    if (is_transpose(dims)) {
      vector_params->has_transpose = true;
    } else if (dims[ndim - 1] == ndim - 1) {
      vector_params->has_permute = true;
    } else {
      throw std::invalid_argument("Unsupported permute operation!");
    }
  } else if (reshape_op.target() == "transpose") {
    int dim0 = reshape_kwargs.at("dim0").int_value();
    int dim1 = reshape_kwargs.at("dim1").int_value();

    dim0 = dim0 < 0 ? dim0 + input_ndim : dim0;
    dim1 = dim1 < 0 ? dim1 + input_ndim : dim1;

    if (dim0 > dim1) {
      std::swap(dim0, dim1);
    }

    if (dim0 == input_ndim - 2 && dim1 == input_ndim - 1) {
      vector_params->has_transpose = true;
    } else if (dim1 != input_ndim - 1) {
      vector_params->has_permute = true;
    } else {
      throw std::invalid_argument("Unsupported transpose operation!");
    }
  }

  if (vector_params->has_permute) {
    if (input_shape[input_shape.size() - 1] % OC_DIMENSION != 0) {
      throw std::invalid_argument(
          "Last dimension of input shape must be a multiple of OC_DIMENSION!");
    }

    std::vector<int> dims;
    if (reshape_kwargs.contains("dims")) {
      auto int_list = reshape_op.kwargs().at("dims").int_list().values();
      dims = std::vector<int>(int_list.begin(), int_list.end());

      for (int i = 0; i < dims.size(); i++) {
        vector_params->vector_fetch_0_permute_dims[i + padded_dims] =
            dims[i] + padded_dims;
      }
    } else if (reshape_kwargs.contains("dim0") &&
               reshape_kwargs.contains("dim1")) {
      int dim0 = reshape_kwargs.at("dim0").int_value();
      int dim1 = reshape_kwargs.at("dim1").int_value();
      std::swap(vector_params->vector_fetch_0_permute_dims[dim0 + padded_dims],
                vector_params->vector_fetch_0_permute_dims[dim1 + padded_dims]);
    } else {
      throw std::invalid_argument("Invalid permute arguments!");
    }
  }

  // TODO: use tiling to set address generator
  Tiling tiling;
  if (vector_params->has_transpose) {
    auto input_shape = get_shape(input, false);
    input_shape = squeeze_shape(input_shape);
    int padded_dims = pad_shape_to_ndim(input_shape, 3);

    // Transpose the input shape
    std::swap(input_shape[1], input_shape[2]);

    if (input_shape[2] % OC_DIMENSION != 0) {
      throw std::invalid_argument(
          "Transposed dimension is not a multiple of OC_DIMENSION!");
    }

    // Tiled access
    vector_params->vector_fetch_0_mode = 1;

    int Y1 = input_shape[0];
    int K1 = input_shape[2] / OC_DIMENSION;
    int X1 = input_shape[1] / BUFSIZE;
    int X0 = OC_DIMENSION;

    tiling = {
        .loops = {{Y1, X1, K1, 1, 1, 1}, {1, 1, 1, 1, 1, X0}},
        .x_loop_idx = {1, 5},
        .y_loop_idx = {0, 4},
        .weight_loop_idx = {2, 1},
    };

    for (int i = 0; i < 3; i++) {
      vector_params->vector_fetch_0_loops[0][i] = tiling.loops[0][i];
    }
    vector_params->vector_fetch_0_y_loop_idx[0] = tiling.y_loop_idx[0];
    vector_params->vector_fetch_0_x_loop_idx[0] = tiling.x_loop_idx[0];
    vector_params->vector_fetch_0_k_loop_idx[0] = tiling.weight_loop_idx[0];

    int loop_index = 0;
    for (int i = 0; i < 6; i++) {
      // ignore the loops not present in outputs (reduction, fx, fy)
      if (i == tiling.y_loop_idx[1]) {
        vector_params->vector_fetch_0_loops[1][loop_index] = tiling.loops[1][i];
        vector_params->vector_fetch_0_y_loop_idx[1] = loop_index++;
      }
      if (i == tiling.x_loop_idx[1]) {
        vector_params->vector_fetch_0_loops[1][loop_index] = tiling.loops[1][i];
        vector_params->vector_fetch_0_x_loop_idx[1] = loop_index++;
      }
      if (i == tiling.weight_loop_idx[1]) {
        vector_params->vector_fetch_0_loops[1][loop_index] = tiling.loops[1][i];
        vector_params->vector_fetch_0_k_loop_idx[1] = loop_index++;
      }
    }

    // Adjust for output loops
    tiling.loops[1][5] = BUFSIZE;
  }

  int packing_factor = OC_DIMENSION / VECTOR_UNIT_WIDTH;
  int numel = OC_DIMENSION;

  if (vector_params->has_transpose) {
    packing_factor = 1;
    numel = BUFSIZE;
  }

  const int dtype_width =
      get_type_width<VU_INPUT_TYPES>(vector_params->vector_fetch_0_dtype);
  const int fetch_width = numel * dtype_width;
  vector_params->vector_fetch_0_burst_size = fetch_width / 8;
  vector_params->vector_fetch_0_num_beats =
      (fetch_width + OC_PORT_WIDTH - 1) / OC_PORT_WIDTH;
  vector_params->vector_fetch_0_packing_factor = packing_factor;

  VECTOR_DATATYPE scale = get_tensor_scalar_scale(input);
  vector_params->vector_fetch_0_dq_scale = scale.bits_rep();

  const auto output = get_op_outputs(param).back();
  vector_params->vector_output_offset = get_address(output);
  vector_params->output_mode = 2;

  auto output_shape = get_shape(output);
  output_shape = split_loops(output_shape, MAX_LOOP_VALUE);
  if (output_shape.size() > 6) {
    throw std::invalid_argument("Too many dimensions for vector operations!");
  }

  output_shape = adjust_loop_indices(output_shape, OC_DIMENSION);

  const int padding = 6 - output_shape.size();
  for (int i = 0; i < padding; i++) {
    output_shape.insert(output_shape.begin(), 1);
  }

  vector_params->output_loops[0][0] = output_shape[0];
  vector_params->output_loops[0][1] = output_shape[1];
  vector_params->output_loops[0][2] = output_shape[2];
  vector_params->output_loops[1][0] = output_shape[3];
  vector_params->output_loops[1][1] = output_shape[4];
  vector_params->output_loops[1][2] = output_shape[5] / OC_DIMENSION;

  if (vector_params->has_transpose) {
    vector_params->output_mode = 1;

    // Set outer loops
    for (int i = 0; i < 3; i++) {
      vector_params->output_loops[0][i] = tiling.loops[0][i];
    }
    vector_params->output_y_loop_idx[0] = tiling.y_loop_idx[0];
    vector_params->output_x_loop_idx[0] = tiling.x_loop_idx[0];
    vector_params->output_k_loop_idx[0] = tiling.weight_loop_idx[0];

    // Set inner loops
    int loop_index = 0;
    for (int i = 0; i < 6; i++) {
      if (i == tiling.y_loop_idx[1]) {
        vector_params->output_loops[1][loop_index] = tiling.loops[1][i];
        vector_params->output_y_loop_idx[1] = loop_index++;
      }
      if (i == tiling.x_loop_idx[1]) {
        vector_params->output_loops[1][loop_index] = tiling.loops[1][i];
        vector_params->output_x_loop_idx[1] = loop_index++;
      }
      if (i == tiling.weight_loop_idx[1]) {
        vector_params->output_loops[1][loop_index] = tiling.loops[1][i];
        vector_params->output_k_loop_idx[1] = loop_index++;
      }
    }
  }

  vector_params->output_dtype =
      get_index_from_type_name<OUTPUT_DATATYPES>(output.dtype());

  // Set input broadcasting based on output shape
  if (!vector_params->has_permute && !vector_params->has_slicing &&
      !vector_params->has_transpose) {
    for (int i = 0; i < 6; i++) {
      if (input_shape[i] != output_shape[i] && input_shape[i] == 1) {
        vector_params->vector_fetch_0_loops[i / 3][i % 3] = output_shape[i];
        vector_params->vector_fetch_0_broadcast[i] = 1;
      }
    }
  }

  VectorInstructions inst;
  inst.op_type = VectorInstructions::vector;
  inst.inst_loop_count = get_size(output) / VECTOR_UNIT_WIDTH;
  inst.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  inst.vdest = VectorInstructions::to_output;

  auto mapping = get_vector_instruction_mapping();
  int stage = 0;

  for (int i = 0; i < op_list.size(); i++) {
    const auto op = op_list[i];
    const std::string opcode = op.target();

    // Grab kwargs that are relevant for some activation functions
    std::map<std::string, VECTOR_DATATYPE> activation_kwargs;
    if (unary_ops_with_kwargs.find(op.target()) !=
        unary_ops_with_kwargs.end()) {
      for (const auto& [key, value] : op.kwargs()) {
        if (key != "input") {
          activation_kwargs[key] = VECTOR_DATATYPE(value.float_value());
        }
      }
    }

    // Dequantization doesn't take a stage in the pipeline
    if (opcode == "dequantize") {
      const auto other = op.kwargs().at("scale").tensor();
      assert(get_size(other) == 1);

      float* array = read_constant_param(other);
      VECTOR_DATATYPE immediate = array[0];
      vector_params->vector_fetch_0_dq_scale = immediate.bits_rep();

      delete[] array;
      continue;
    }

    for (; stage < vector_unit_stages.size(); stage++) {
      // Only the last stage has a true divider
      if (opcode == "div" && stage != 3) {
        const auto other = op.kwargs().at("other").tensor();
        if (get_size(other) > 1) {
          continue;
        }
      }

      if (vector_unit_stages[stage].find(opcode) !=
          vector_unit_stages[stage].end()) {
        break;
      }
    }

    if (stage == vector_unit_stages.size()) {
      throw std::runtime_error("Vector operation not supported!\n");
    }

    spdlog::debug("stage {} target: {}\n", stage, opcode);

    unsigned int vop = mapping[opcode];
    if (stage == 0) {
      inst.vector_op0 = opcode == "div" ? VectorInstructions::vmult : vop;
    } else if (stage == 1) {
      inst.vector_op1 = vop;
    } else if (stage == 2) {
      inst.vector_op2 = opcode == "div" ? VectorInstructions::vmult : vop;
    } else if (stage == 3) {
      inst.vector_op3 = vop;
    }

    if (opcode == "neg") {
      const auto self = op.kwargs().at("input").tensor();
      const auto output_shape = squeeze_shape(get_shape(self));

      VECTOR_DATATYPE immediate = 0;
      inst.immediate0 = immediate.bits_rep();
      inst.vector_op0_src0 = VectorInstructions::from_immediate_0;
      inst.vector_op0_src1 = VectorInstructions::from_vector_fetch_0;
    } else if (opcode == "quantize_mx" || opcode == "quantize_mx_outlier") {
      float quant_max = op.kwargs().at("quant_max").float_value();
      bool force_scale_power_of_two =
          op.kwargs().at("force_scale_power_of_two").bool_value();

      if (force_scale_power_of_two) {
        inst.immediate2 = floor(log2(quant_max));
      } else {
        VECTOR_DATATYPE scale = quant_max;
        inst.immediate2 = scale.bits_rep();
      }

      const auto outputs = get_op_outputs(param);
      const int num_outputs = outputs.size();

      vector_params->quantize_output_mx = true;
      vector_params->mx_scale_offset = get_address(outputs[num_outputs - 2]);

      if (opcode == "quantize_mx_outlier") {
        vector_params->has_sparse_output = true;
        vector_params->csr_data_offset = get_address(outputs[0]);
        vector_params->csr_indices_offset = get_address(outputs[1]);
        vector_params->csr_indptr_offset = get_address(outputs[2]);

        auto& config = vector_instruction_config->outlier_filter;

        VECTOR_DATATYPE threshold = op.kwargs().at("threshold").float_value();
        config.outlier_threshold = threshold.bits_rep();

        const auto quantize_input = op.kwargs().at("input").tensor();
        const auto quantize_shape = get_shape(quantize_input);
        config.dense_input_shape[1] = quantize_shape.back() / VECTOR_UNIT_WIDTH;
        config.dense_input_shape[0] =
            get_size(quantize_input) / quantize_shape.back();
      }

      // Copy coefficients from ApproximationConstants.h
    } else if (opcode == "gelu" || opcode == "gelu_") {
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = GELU_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] = GELU_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = GELU_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = GELU_CLAMP_MAX;
    } else if (opcode == "silu" || opcode == "silu_") {
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = SILU_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] = SILU_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = SILU_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = SILU_CLAMP_MAX;
    } else if (opcode == "elu" || opcode == "elu_") {
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = ELU_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] = ELU_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = ELU_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = ELU_CLAMP_MAX;
    } else if (opcode == "log_sigmoid" || opcode == "log_sigmoid_") {
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = LOGSIGMOID_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] =
              LOGSIGMOID_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = LOGSIGMOID_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = LOGSIGMOID_CLAMP_MAX;
    } else if (opcode == "tanh" || opcode == "tanh_") {
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = TANH_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] = TANH_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = TANH_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = TANH_CLAMP_MAX;
    } else if (opcode == "tanh_1" || opcode == "tanh_1_") {
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = TANHSHRINK_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] =
              TANHSHRINK_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = TANHSHRINK_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = TANHSHRINK_CLAMP_MAX;
    } else if (opcode == "softplus" || opcode == "softplus_") {
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = SOFTPLUS_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] =
              SOFTPLUS_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = SOFTPLUS_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = SOFTPLUS_CLAMP_MAX;
    } else if (opcode == "mish" || opcode == "mish_") {
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = MISH_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] = MISH_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = MISH_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = MISH_CLAMP_MAX;
    } else if (opcode == "sigmoid" || opcode == "sigmoid_") {
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = SIGMOID_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] = SIGMOID_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = SIGMOID_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = SIGMOID_CLAMP_MAX;
    } else if (opcode == "selu" || opcode == "selu_") {
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = SELU_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] = SELU_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = SELU_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = SELU_CLAMP_MAX;
    } else if (opcode == "celu" || opcode == "celu_") {
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = CELU_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] = CELU_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = CELU_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = CELU_CLAMP_MAX;
    } else if (opcode == "hardsigmoid" || opcode == "hardsigmoid_") {
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = HARDSIGMOID_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] =
              HARDSIGMOID_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = HARDSIGMOID_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = HARDSIGMOID_CLAMP_MAX;
    } else if (opcode == "hardswish" || opcode == "hardswish_") {
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = HARDSWISH_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] =
              HARDSWISH_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = HARDSWISH_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = HARDSWISH_CLAMP_MAX;

      // Start of activation functions with kwargs
    } else if (opcode == "hardshrink" || opcode == "hardshrink_") {
      const VECTOR_DATATYPE lambda = activation_kwargs.at("lambd");
      const VECTOR_DATATYPE HARDSHRINK_MAXES[NUM_MAXES] = {
          -lambda, 0.0, 0.0, 0.0, 0.0, lambda};
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = HARDSHRINK_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] =
              HARDSHRINK_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = HARDSHRINK_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = HARDSHRINK_CLAMP_MAX;
    } else if (opcode == "hardtanh" || opcode == "hardtanh_") {
      const VECTOR_DATATYPE min_val = activation_kwargs.at("min_val");
      const VECTOR_DATATYPE max_val = activation_kwargs.at("max_val");
      const VECTOR_DATATYPE HARDTANH_MAXES[NUM_MAXES] = {
          min_val, max_val, max_val, max_val, max_val, max_val};
      const VECTOR_DATATYPE HARDTANH_RANGES[NUM_RANGES][NUM_COEFFS] = {
          {min_val, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 1.0, 0.0},
          {0.0, 1.0, 0.0},     {0.0, 1.0, 0.0}, {0.0, 1.0, 0.0},
          {max_val, 0.0, 0.0}};
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = HARDTANH_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] =
              HARDTANH_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = HARDTANH_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = HARDTANH_CLAMP_MAX;
    } else if (opcode == "leaky_relu" || opcode == "leaky_relu_") {
      const VECTOR_DATATYPE negative_slope =
          activation_kwargs.at("negative_slope");
      const VECTOR_DATATYPE LEAKYRELU_RANGES[NUM_RANGES][NUM_COEFFS] = {
          {0.0, negative_slope, 0.0},
          {0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0},
          {0.0, 1.0, 0.0}};
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = LEAKYRELU_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] =
              LEAKYRELU_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = LEAKYRELU_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = LEAKYRELU_CLAMP_MAX;
    } else if (opcode == "rrelu" || opcode == "rrelu_") {
      const VECTOR_DATATYPE lower = activation_kwargs.at("lower");
      const VECTOR_DATATYPE upper = activation_kwargs.at("upper");
      const VECTOR_DATATYPE alpha = (lower + upper) / VECTOR_DATATYPE(2);
      const VECTOR_DATATYPE RRELU_RANGES[NUM_RANGES][NUM_COEFFS] = {
          {0.0, alpha, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0},   {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = RRELU_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] = RRELU_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = RRELU_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = RRELU_CLAMP_MAX;
    } else if (opcode == "softshrink" || opcode == "softshrink_") {
      const VECTOR_DATATYPE lambda = activation_kwargs.at("lambd");
      const VECTOR_DATATYPE SOFTSHRINK_MAXES[NUM_MAXES] = {
          -lambda, 0.0, 0.0, 0.0, 0.0, lambda};
      const VECTOR_DATATYPE SOFTSHRINK_RANGES[NUM_RANGES][NUM_COEFFS] = {
          {lambda, 1.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0},    {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
          {-lambda, 1.0, 0.0}};
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = SOFTSHRINK_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] =
              SOFTSHRINK_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = SOFTSHRINK_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = SOFTSHRINK_CLAMP_MAX;
    } else if (opcode == "threshold" || opcode == "threshold_") {
      const VECTOR_DATATYPE threshold = activation_kwargs.at("threshold");
      const VECTOR_DATATYPE value = activation_kwargs.at("value");
      const VECTOR_DATATYPE THRESHOLD_MAXES[NUM_MAXES] = {0.0, 0.0, 0.0,
                                                          0.0, 0.0, threshold};
      const VECTOR_DATATYPE THRESHOLD_RANGES[NUM_RANGES][NUM_COEFFS] = {
          {value, 0.0, 0.0}, {value, 0.0, 0.0}, {value, 0.0, 0.0},
          {value, 0.0, 0.0}, {value, 0.0, 0.0}, {value, 0.0, 0.0},
          {0.0, 1.0, 0.0}};
      for (int i = 0; i < NUM_MAXES; i++) {
        vector_instruction_config->approx.maxes[i] = THRESHOLD_MAXES[i];
      }
      for (int i = 0; i < NUM_RANGES; i++) {
        for (int j = 0; j < NUM_COEFFS; j++) {
          vector_instruction_config->approx.ranges[i][j] =
              THRESHOLD_RANGES[i][j];
        }
      }
      vector_instruction_config->approx.clamp_min = THRESHOLD_CLAMP_MIN;
      vector_instruction_config->approx.clamp_max = THRESHOLD_CLAMP_MAX;
    } else if (op.kwargs().contains("other") || opcode == "quantize") {
      std::string other_key = opcode == "quantize" ? "scale" : "other";
      const auto other = op.kwargs().at(other_key);

      if (other.has_float_value()) {
        float scalar = other.float_value();
        set_immediate(scalar, stage, opcode, inst);
      } else if (other.has_int_value()) {
        int scalar = other.int_value();
        set_immediate(scalar, stage, opcode, inst);
      } else if (other.has_tensor() && get_size(other.tensor()) == 1) {
        float* array = read_constant_param(other.tensor());
        set_immediate(array[0], stage, opcode, inst);
        delete[] array;
      } else {
        auto self = op.kwargs().at("input").tensor();
        auto tensor = other.tensor();

        auto input_shape = get_shape(self);
        auto other_shape = get_shape(tensor);

        if (opcode == "quantize") {
          auto result =
              factor_out_non_broadcastable_dim(input_shape, other_shape);
          input_shape = result.first;
          other_shape = result.second;
        }

        auto output_shape = broadcast_shape(input_shape, other_shape);
        update_tensor_shape(self, input_shape);
        update_tensor_shape(tensor, other_shape);
        auto tensor_to_load = tensor.has_memory() ? tensor : self;

        if (stage == 0) {
          inst.vector_op0_src1 = VectorInstructions::from_vector_fetch_1;
          set_vector_fetch_1(tensor_to_load, output_shape, vector_params);
        } else if (stage == 2) {
          inst.vector_op2_src1 = VectorInstructions::from_vector_fetch_2;
          set_vector_fetch_2(tensor_to_load, output_shape, vector_params);
        } else {
          assert(inst.vector_op2_src1 !=
                 VectorInstructions::from_vector_fetch_2);
          inst.vector_op3_src1 = VectorInstructions::from_vector_fetch_2;
          set_vector_fetch_2(tensor_to_load, output_shape, vector_params);
        }
      }
    }

    // For both quantize and quantize_mx, set output codebook if provided
    if (op.kwargs().contains("output_code")) {
      const auto code = op.kwargs().at("output_code").tensor();
      const int size = get_size(code);

      float* array = read_constant_param(code);

      for (int i = 0; i < size; i++) {
        vector_params->output_code[i] = array[i] * 2;
      }

      delete[] array;

      vector_params->use_output_codebook = true;
    }

    stage++;
  }

  // total output count
  vector_instruction_config->inst[0] = inst;
  vector_instruction_config->num_inst = 1;
  vector_instruction_config->config_loop_count = 1;

  mapped_params.push_back(vector_params);
  mapped_params.push_back(vector_instruction_config);
}
