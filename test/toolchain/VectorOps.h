#pragma once

#include "test/common/Tiling.h"
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

void set_vector_fetch_1(const Tensor& tensor, std::vector<int> output_shape,
                        VectorParams* vector_params) {
  if (tensor.window_pitch > 0) {
    throw std::runtime_error(
        "A pitched window cannot be fetched as a side operand.");
  }
  vector_params->vector_fetch_1_offset = get_address(tensor);
  vector_params->vector_fetch_1_mode = true;

  auto input_shape = get_shape(tensor);
  squeeze_front_ones(input_shape);
  pad_shape_to_ndim(input_shape, 3);

  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_1_broadcast[i] =
        input_shape[i] == 1 && output_shape[i] != 1;
  }

  const int dtype = get_index_from_type_name<VU_INPUT_TYPES>(tensor.dtype);
  const int dtype_width = get_type_width<VU_INPUT_TYPES>(dtype);
  const int fetch_width = OC_DIMENSION * dtype_width;

  vector_params->vector_fetch_1_dtype = dtype;
  vector_params->vector_fetch_1_stride = OC_DIMENSION;
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
}

void set_vector_fetch_2(const Tensor& tensor, std::vector<int> output_shape,
                        VectorParams* vector_params) {
  if (tensor.window_pitch > 0) {
    throw std::runtime_error(
        "A pitched window cannot be fetched as a side operand.");
  }
  vector_params->vector_fetch_2_offset = get_address(tensor);
  vector_params->vector_fetch_2_mode = true;

  auto input_shape = get_shape(tensor);
  squeeze_front_ones(input_shape);
  pad_shape_to_ndim(input_shape, 3);

  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_2_broadcast[i] = input_shape[i] == 1;
  }

  const int dtype = get_index_from_type_name<VU_INPUT_TYPES>(tensor.dtype);
  const int dtype_width = get_type_width<VU_INPUT_TYPES>(dtype);
  const int fetch_width = OC_DIMENSION * dtype_width;

  vector_params->vector_fetch_2_dtype = dtype;
  vector_params->vector_fetch_2_stride = OC_DIMENSION;
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
}

void map_vector_operations(const voyager::Operation& operation,
                           const ScalarEnv& env,
                           std::deque<BaseParams*>& mapped_params) {
  const auto& original_op_list = get_prim_ops(operation);

  VectorParams* vector_params = new VectorParams;
  VectorInstructionConfig* vector_instruction_config =
      new VectorInstructionConfig;

  // Support a maximum buffer size of 1024
  constexpr int BUFSIZE =
      std::min({1024 / OC_DIMENSION, OC_DIMENSION, VECTOR_UNIT_WIDTH});

  auto op_list = get_prim_ops(operation);

  // A leading side-operand dequantize is not the head of the chain: its
  // result rejoins the pipeline as a later stage's side operand, and that
  // consumer binds it through a side fetch. The value flowing down the
  // pipeline is the next op's input.
  const voyager::PrimOp* head = op_list[0];
  for (const auto* prim : op_list) {
    if (strip_namespace(prim->target()) == "dequantize" &&
        is_side_operand_dequantize(*prim, op_list)) {
      continue;
    }
    head = prim;
    break;
  }

  const auto input = resolve(*head, "input", env);
  vector_params->vector_fetch_0_offset = get_address(input);
  vector_params->vector_fetch_0_mode = 2;

  // Use the original shape without permute/slice
  auto input_shape = get_shape(input);
  const int input_ndim = input_shape.size();

  if (is_dma_op(strip_namespace(op_list[0]->target()))) {
    for (const auto dim : input_shape) {
      if (dim > MAX_LOOP_VALUE) {
        spdlog::error("ERROR: input shape dimension is greater than {}: ",
                      MAX_LOOP_VALUE);
        print_shape(input_shape);
        throw std::invalid_argument("Unsupported input shape dimension!");
      }
    }
  } else if (input.window_pitch > 0) {
    // A pitched window: the fetch walks the underlying rows, and the slice
    // below clamps each to the window's run.
    input_shape.back() = input.window_pitch;
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

  const voyager::PrimOp* reshape_op = nullptr;
  if (is_dma_op(strip_namespace(op_list[0]->target()))) {
    reshape_op = op_list[0];
    op_list.erase(op_list.begin());
  }

  const std::string reshape_target =
      reshape_op ? strip_namespace(reshape_op->target()) : "";

  if (reshape_target == "slice") {
    vector_params->has_slicing = true;

    auto start = arg_int(*reshape_op, "start", env);
    auto end = arg_int(*reshape_op, "end", env);
    auto step = arg_int(*reshape_op, "step", env);
    auto dim = arg_int(*reshape_op, "dim", env);

    auto shape = get_shape(input);

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
  } else if (reshape_target == "permute") {
    const std::vector<int> dims = arg_ints(*reshape_op, "dims", env);
    const int ndim = input.shape.size();

    if (is_transpose(dims)) {
      vector_params->has_transpose = true;
    } else if (dims[ndim - 1] == ndim - 1) {
      vector_params->has_permute = true;
    } else {
      throw std::invalid_argument("Unsupported permute operation!");
    }
  } else if (reshape_target == "transpose") {
    int dim0 = arg_int(*reshape_op, "dim0", env);
    int dim1 = arg_int(*reshape_op, "dim1", env);

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

  // A pitched window (a strided voyager.subview) fetches like a fused slice
  // of the last dimension: the loops walk the underlying rows and the slice
  // clamps each to the window's run.
  if (input.window_pitch > 0) {
    vector_params->has_slicing = true;

    const int64_t col = input.window_col;
    const int64_t run = input.shape.back();
    if (col % OC_DIMENSION != 0 || run % OC_DIMENSION != 0) {
      throw std::invalid_argument(
          "Window bounds for the last dimension must be multiples of "
          "OC_DIMENSION!");
    }

    vector_params->vector_fetch_0_slice_dim = 5;
    vector_params->vector_fetch_0_slice_start = col / OC_DIMENSION;
    vector_params->vector_fetch_0_slice_step = 1;
    vector_params->vector_fetch_0_slice_end = run / OC_DIMENSION;

    vector_params->vector_fetch_0_offset =
        get_address(input) - col * get_width(input) / 8;
  }

  if (vector_params->has_permute) {
    if (input_shape[input_shape.size() - 1] % OC_DIMENSION != 0) {
      throw std::invalid_argument(
          "Last dimension of input shape must be a multiple of OC_DIMENSION!");
    }

    if (has_arg(*reshape_op, "dims")) {
      const std::vector<int> dims = arg_ints(*reshape_op, "dims", env);

      for (int i = 0; i < dims.size(); i++) {
        vector_params->vector_fetch_0_permute_dims[i + padded_dims] =
            dims[i] + padded_dims;
      }
    } else if (has_arg(*reshape_op, "dim0") && has_arg(*reshape_op, "dim1")) {
      int dim0 = arg_int(*reshape_op, "dim0", env);
      int dim1 = arg_int(*reshape_op, "dim1", env);
      std::swap(vector_params->vector_fetch_0_permute_dims[dim0 + padded_dims],
                vector_params->vector_fetch_0_permute_dims[dim1 + padded_dims]);
    } else {
      throw std::invalid_argument("Invalid permute arguments!");
    }
  }

  // TODO: use tiling to set address generator
  Tiling tiling;
  if (vector_params->has_transpose) {
    auto input_shape = get_shape(input);
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

  int data_stride = OC_DIMENSION;
  int packing_factor = OC_DIMENSION / VECTOR_UNIT_WIDTH;

  if (vector_params->has_transpose) {
    data_stride = BUFSIZE;
    packing_factor = 1;
  }

  const int dtype = get_index_from_type_name<VU_INPUT_TYPES>(input.dtype);
  const int dtype_width = get_type_width<VU_INPUT_TYPES>(dtype);
  const int fetch_width = data_stride * dtype_width;

  vector_params->vector_fetch_0_dtype = dtype;
  vector_params->vector_fetch_0_stride = OC_DIMENSION;
  vector_params->vector_fetch_0_burst_size = fetch_width / 8;
  vector_params->vector_fetch_0_num_beats =
      (fetch_width + OC_PORT_WIDTH - 1) / OC_PORT_WIDTH;
  vector_params->vector_fetch_0_packing_factor = packing_factor;

  const auto output = resolve_outputs(operation, env).back();
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
      get_index_from_type_name<OUTPUT_DATATYPES>(output.dtype);

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

  auto map_tensor_operand = [&](const voyager::PrimOp& op, const ScalarEnv& env,
                                const std::string& other_key, int stage,
                                VectorInstructions& pipeline_inst) {
    Tensor self = resolve(op, "input", env);
    Tensor tensor = resolve(op, other_key, env);

    auto input_shape = get_shape(self);
    auto other_shape = get_shape(tensor);

    if (strip_namespace(op.target()) == "quantize") {
      if (input_shape.size() < 3) pad_shape_to_ndim(input_shape, 3);
      if (other_shape.size() < 3) pad_shape_to_ndim(other_shape, 3);
      auto result = factor_out_non_broadcastable_dim(input_shape, other_shape);
      input_shape = result.first;
      other_shape = result.second;
    }

    auto output_shape = broadcast_shape(input_shape, other_shape);
    update_tensor_shape(self, input_shape);
    update_tensor_shape(tensor, other_shape);

    // An operand a dequantize produced inside this fusion is not itself
    // materialized, so fetch what that dequantize reads; the shared pipeline
    // mapper applies the dequantize scale through the stage mac. Elementwise,
    // so the raw codes keep the side operand's shape.
    const voyager::PrimOp* producer =
        get_fused_producer(operation, env, tensor);
    Tensor tensor_to_load = tensor.materialized ? tensor : self;
    if (producer != nullptr &&
        strip_namespace(producer->target()) == "dequantize") {
      tensor_to_load = resolve(*producer, "input", env);
      update_tensor_shape(tensor_to_load, other_shape);
    }

    if (stage == 0) {
      pipeline_inst.vector_op0_src1 = VectorInstructions::from_vector_fetch_1;
      set_vector_fetch_1(tensor_to_load, output_shape, vector_params);
    } else if (stage == 2) {
      pipeline_inst.vector_op2_src1 = VectorInstructions::from_vector_fetch_2;
      set_vector_fetch_2(tensor_to_load, output_shape, vector_params);
    } else {
      if (pipeline_inst.vector_op2_src1 ==
          VectorInstructions::from_vector_fetch_2) {
        throw std::runtime_error(
            "Vector pipeline stages 2 and 3 cannot fetch different side "
            "operands.");
      }
      pipeline_inst.vector_op3_src1 = VectorInstructions::from_vector_fetch_2;
      set_vector_fetch_2(tensor_to_load, output_shape, vector_params);
    }
  };

  auto is_stage_available = [](const voyager::PrimOp&, const ScalarEnv&, int) {
    return true;
  };
  // A microscaled quantize whose block is strided leaves its scale to a pass
  // of its own; this pass divides by what that pass wrote, so bind it the way
  // a quantize with a tensor scale is bound.
  Tensor mx_scale_fetch;
  mx_scale_fetch.materialized = false;

  map_vector_pipeline_ops(operation, env, op_list, vector_params,
                          vector_instruction_config, inst, mapped_params,
                          map_tensor_operand, is_stage_available, 0,
                          &mx_scale_fetch);

  if (mx_scale_fetch.materialized) {
    // The scale broadcasts over the quantize's own view of the data. The
    // anchor's input is the shape before the relayout the quantize sits
    // behind, and against that shape the block count multiplies into the fetch
    // walk instead of collapsing into it.
    auto data_shape = get_shape(resolve(*op_list.back(), "input", env));
    auto scale_shape = get_shape(mx_scale_fetch);
    if (data_shape.size() < 3) pad_shape_to_ndim(data_shape, 3);
    if (scale_shape.size() < 3) pad_shape_to_ndim(scale_shape, 3);

    auto factored = factor_out_non_broadcastable_dim(data_shape, scale_shape);
    auto fetch_shape = broadcast_shape(factored.first, factored.second);

    update_tensor_shape(mx_scale_fetch, factored.second);
    set_vector_fetch_2(mx_scale_fetch, fetch_shape, vector_params);
  }

  // total output count
  vector_instruction_config->inst[0] = inst;
  vector_instruction_config->num_inst = 1;
  vector_instruction_config->config_loop_count = 1;

  mapped_params.push_back(vector_params);
  mapped_params.push_back(vector_instruction_config);
}
