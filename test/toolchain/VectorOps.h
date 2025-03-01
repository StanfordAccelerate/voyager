#pragma once

#include "test/toolchain/Common.h"

inline bool are_broadcastable(const std::vector<int> &shape1,
                              const std::vector<int> &shape2) {
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

inline std::vector<int> broadcast_shape(const std::vector<int> &shape1,
                                        const std::vector<int> &shape2) {
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

  return squeeze_shape(result_shape);
}

void set_vector_addr_gen1(const codegen::Tensor &tensor,
                          const std::vector<int> &output_shape,
                          AcceleratorMemoryMap &accelerator_memory_map,
                          VectorParams *vector_params) {
  const auto memory = tensor.memory();
  accelerator_memory_map["vector1"] = get_partition(memory.partition());
  vector_params->ADDRESS_GEN1_OFFSET = memory.address();

  int nonzero_dims = 0;
  for (const int &dim : tensor.shape()) {
    if (dim != 1) nonzero_dims++;
  }

  if (nonzero_dims == 1) {
    vector_params->addressGen1Mode = 3;
  } else if (nonzero_dims == 2) {
    vector_params->addressGen1Mode = 2;
  } else if (nonzero_dims == 3) {
    vector_params->addressGen1Mode = 1;
  }

  vector_params->fetch_vector_type_1 =
      DataTypes::TypeName<VECTOR_DATATYPE>::name() == tensor.dtype();

  for (int i = 0; i < 3; i++) {
    vector_params->addressGen1Loops[0][i] = 1;
  }

  const int total_dims = output_shape.size();
  if (total_dims == 1) {
    vector_params->addressGen1Loops[1][0] = 1;
    vector_params->addressGen1Loops[1][1] = 1;
  } else if (total_dims == 2) {
    vector_params->addressGen1Loops[1][0] = 1;
    vector_params->addressGen1Loops[1][1] = output_shape[0];
  } else if (total_dims == 3) {
    vector_params->addressGen1Loops[1][0] = output_shape[0];
    vector_params->addressGen1Loops[1][1] = output_shape[1];
  } else {
    throw std::invalid_argument(
        "Unsupported number of dimensions for vector operations!");
  }
  vector_params->addressGen1Loops[1][2] = output_shape.back() / OC_DIMENSION;

  for (int i = 0; i < 2; i++) {
    vector_params->addressGen1InputYLoopIndex[i] = 0;
    vector_params->addressGen1InputXLoopIndex[i] = 1;
    vector_params->addressGen1WeightLoopIndex[i] = 2;
  }

  DataTypes::bfloat16 scale = tensor.scale() != 0 ? tensor.scale() : 1.0;
  vector_params->vec1_dq_scale = scale.bits_rep();
}

void set_vector_addr_gen2(const codegen::Tensor &tensor,
                          const std::vector<int> &output_shape,
                          AcceleratorMemoryMap &accelerator_memory_map,
                          VectorParams *vector_params) {
  const auto memory = tensor.memory();
  accelerator_memory_map["vector2"] = get_partition(memory.partition());
  vector_params->ADDRESS_GEN2_OFFSET = memory.address();

  int nonzero_dims = 0;
  for (const int &dim : tensor.shape()) {
    if (dim != 1) nonzero_dims++;
  }

  if (nonzero_dims == 1) {
    vector_params->addressGen2Mode = 3;
  } else if (nonzero_dims == 2) {
    vector_params->addressGen2Mode = 2;
  } else if (nonzero_dims == 3) {
    vector_params->addressGen2Mode = 1;
  }

  vector_params->fetch_vector_type_2 =
      DataTypes::TypeName<VECTOR_DATATYPE>::name() == tensor.dtype();

  for (int i = 0; i < 3; i++) {
    vector_params->addressGen2Loops[0][i] = 1;
  }

  const int total_dims = output_shape.size();
  if (total_dims == 1) {
    vector_params->addressGen2Loops[1][0] = 1;
    vector_params->addressGen2Loops[1][1] = 1;
  } else if (total_dims == 2) {
    vector_params->addressGen2Loops[1][0] = 1;
    vector_params->addressGen2Loops[1][1] = output_shape[0];
  } else if (total_dims == 3) {
    vector_params->addressGen2Loops[1][0] = output_shape[0];
    vector_params->addressGen2Loops[1][1] = output_shape[1];
  } else {
    throw std::invalid_argument(
        "Unsupported number of dimensions for vector operations!");
  }
  vector_params->addressGen2Loops[1][2] = output_shape.back() / OC_DIMENSION;

  for (int i = 0; i < 2; i++) {
    vector_params->addressGen2InputYLoopIndex[i] = 0;
    vector_params->addressGen2InputXLoopIndex[i] = 1;
    vector_params->addressGen2WeightLoopIndex[i] = 2;
  }

  DataTypes::bfloat16 scale = tensor.scale() != 0 ? tensor.scale() : 1.0;
  vector_params->vec2_dq_scale = scale.bits_rep();
}

void set_vector_immediate(const float scalar, const int stage,
                          const std::string opcode, VectorInstructions &inst) {
  std::cerr << "immediate: " << scalar << std::endl;

  VECTOR_DATATYPE immediate = scalar;

  if (opcode == "div" || opcode == "div_") {
    immediate = 1.0 / immediate;
  }

  if (stage == 0) {
    inst.vector_op0_src1 = VectorInstructions::from_immediate_0;
    inst.immediate0 = immediate.bits_rep();
  } else if (stage == 3) {
    inst.vector_op2_src1 = VectorInstructions::from_immediate_1;
    inst.immediate1 = immediate.bits_rep();
  } else if (stage == 5) {
    inst.immediate1 = immediate.bits_rep();
  }
}

void MapVectorOperations(const codegen::Operation &param,
                         std::deque<BaseParams *> &mappedParams,
                         std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  VectorParams *vector_params = new VectorParams;
  AcceleratorMemoryMap accelerator_memory_map;

  const auto op_list = get_op_list(param);

  const auto input = op_list[0].kwargs().at("input").tensor();

  codegen::Tensor output;
  if (param.has_output()) {
    output = param.output();
  } else {
    assert(op_list.back().target() == "quantize_mx");
    output = param.outputs().tensors(1);
  }

  const auto input_memory = input.memory();
  accelerator_memory_map["vector0"] = get_partition(input_memory.partition());
  vector_params->VECTOR_OFFSET = input_memory.address();
  vector_params->addressGen0Mode = 2;

  // Use the original shape without permute/slice
  auto input_shape = get_shape(input, false);

  if (input.has_reshape() ||
      MEMORY_OPS.find(op_list[0].target()) != MEMORY_OPS.end()) {
    for (const auto dim : input_shape) {
      if (dim > 1024) {
        std::cerr << "ERROR: input shape dimension is greater than 1024: ";
        print_shape(input_shape);
        throw std::invalid_argument("Invalid input shape dimension!");
      }
    }

    if (input_shape.back() % OC_DIMENSION != 0) {
      std::cerr << "ERROR: input last dimension is not a multiple of "
                   "OC_DIMENSION: ";
      print_shape(input_shape);
      throw std::invalid_argument("Invalid input shape dimension!");
    }
  } else {
    input_shape = split_loops(input_shape, 1024);
    input_shape = adjust_loop_indices(input_shape, OC_DIMENSION);
  }

  // Pad the shape to 6 dimensions with 1s
  int padded_dims = pad_shape_to_ndim(input_shape, 6);

  vector_params->addressGen0Loop[0][0] = input_shape[0];
  vector_params->addressGen0Loop[0][1] = input_shape[1];
  vector_params->addressGen0Loop[0][2] = input_shape[2];
  vector_params->addressGen0Loop[1][0] = input_shape[3];
  vector_params->addressGen0Loop[1][1] = input_shape[4];
  vector_params->addressGen0Loop[1][2] = input_shape[5] / OC_DIMENSION;

  auto reshape_op = op_list[0];
  if (input.has_reshape()) {
    reshape_op = input.reshape();
  }

  const auto reshape_kwargs = reshape_op.kwargs();

  if (reshape_op.target() == "slice") {
    vector_params->has_slicing = true;

    uint64_t start = reshape_kwargs.at("start").int_value();
    uint64_t end = reshape_kwargs.at("end").int_value();
    uint64_t step = reshape_kwargs.at("step").int_value();
    uint64_t dim = reshape_kwargs.at("dim").int_value();

    std::vector<int> shape(input.shape().begin(), input.shape().end());

    dim = dim < 0 ? dim + shape.size() : dim;
    end = end > shape[dim] ? shape[dim] : end;

    vector_params->vec0_dim = dim + padded_dims;
    vector_params->vec0_start = start;
    vector_params->vec0_end = end;
    vector_params->vec0_step = step;

    // Last dimension needs to be scaled by OC_DIMENSION
    if (vector_params->vec0_dim == 5) {
      if (start % OC_DIMENSION != 0 || end % OC_DIMENSION != 0) {
        throw std::invalid_argument(
            "Slice start and end must be multiples of OC_DIMENSION!");
      }
      vector_params->vec0_start /= OC_DIMENSION;
      vector_params->vec0_end /= OC_DIMENSION;
    }
  }

  if (reshape_op.target() == "permute") {
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
    if (dim0 > dim1) {
      std::swap(dim0, dim1);
    }

    if (dim0 == input.shape_size() - 2 && dim1 == input.shape_size() - 1) {
      vector_params->has_transpose = true;
    } else if (dim1 != input.shape_size() - 1) {
      vector_params->has_permute = true;
    } else {
      throw std::invalid_argument("Unsupported transpose operation!");
    }
  }

  if (vector_params->has_permute) {
    std::vector<int> dims;
    if (reshape_kwargs.contains("dims")) {
      auto int_list = reshape_op.kwargs().at("dims").int_list().values();
      dims = std::vector<int>(int_list.begin(), int_list.end());

      for (int i = 0; i < dims.size(); i++) {
        vector_params->vec0_dim_order[i + padded_dims] = dims[i] + padded_dims;
      }
    } else if (reshape_kwargs.contains("dim0") &&
               reshape_kwargs.contains("dim1")) {
      int dim0 = reshape_kwargs.at("dim0").int_value();
      int dim1 = reshape_kwargs.at("dim1").int_value();
      std::swap(vector_params->vec0_dim_order[dim0 + padded_dims],
                vector_params->vec0_dim_order[dim1 + padded_dims]);
    } else {
      throw std::invalid_argument("Invalid permute arguments!");
    }
  }

  // TODO: use tiling to set address generator
  Tiling tiling;

  if (vector_params->has_transpose) {
    // Only support a buffer size of 32
    const int BUFSIZE = OC_DIMENSION < 32 ? OC_DIMENSION : 32;

    std::vector<int> input_shape(input.shape().begin(), input.shape().end());
    input_shape = squeeze_shape(input_shape);
    int padded_dims = pad_shape_to_ndim(input_shape, 3);

    // Transpose the input shape
    std::swap(input_shape[1], input_shape[2]);

    if (input_shape[2] % OC_DIMENSION != 0) {
      throw std::invalid_argument(
          "Transposed dimension is not a multiple of OC_DIMENSION!");
    }

    // Tiled access
    vector_params->addressGen0Mode = 1;

    int Y1 = input_shape[0];
    int K1 = input_shape[2] / OC_DIMENSION;
    int X1 = input_shape[1] / BUFSIZE;
    int X0 = OC_DIMENSION;

    tiling = {
        .loops = {{Y1, X1, K1, 1, 1, 1}, {1, 1, 1, 1, 1, X0}},
        .x_loop_index = {1, 5},
        .y_loop_index = {0, 4},
        .weight_loop_index = {2, 1},
    };

    for (int i = 0; i < 3; i++) {
      vector_params->addressGen0Loop[0][i] = tiling.loops[0][i];
    }
    vector_params->addressGen0InputYLoopIndex[0] = tiling.y_loop_index[0];
    vector_params->addressGen0InputXLoopIndex[0] = tiling.x_loop_index[0];
    vector_params->addressGen0WeightLoopIndex[0] = tiling.weight_loop_index[0];

    int loop_index = 0;
    for (int i = 0; i < 6; i++) {
      // ignore the loops not present in outputs (reduction, fx, fy)
      if (i == tiling.y_loop_index[1]) {
        vector_params->addressGen0Loop[1][loop_index] = tiling.loops[1][i];
        vector_params->addressGen0InputYLoopIndex[1] = loop_index++;
      }
      if (i == tiling.x_loop_index[1]) {
        vector_params->addressGen0Loop[1][loop_index] = tiling.loops[1][i];
        vector_params->addressGen0InputXLoopIndex[1] = loop_index++;
      }
      if (i == tiling.weight_loop_index[1]) {
        vector_params->addressGen0Loop[1][loop_index] = tiling.loops[1][i];
        vector_params->addressGen0WeightLoopIndex[1] = loop_index++;
      }
    }

    // Adjust for output loops
    tiling.loops[1][5] = BUFSIZE;
  }

  VECTOR_DATATYPE scale = 1.0;
  vector_params->vec0_dq_scale = scale.bits_rep();

  // set double precision if the datatype is not the same as the input
  // datatype
  vector_params->fetch_vector_type_0 =
      input.dtype() == DataTypes::TypeName<VECTOR_DATATYPE>::name();

  const auto output_memory = output.memory();
  accelerator_memory_map["outputs"] = get_partition(output_memory.partition());
  vector_params->VECTOR_OUTPUT_OFFSET = output_memory.address();
  vector_params->outputAddressMode = 2;

  auto output_shape = get_shape(output);
  output_shape = split_loops(output_shape, 1024);
  if (output_shape.size() > 6) {
    throw std::invalid_argument("Too many dimensions for vector operations!");
  }

  output_shape = adjust_loop_indices(output_shape, OC_DIMENSION);

  const int padding = 6 - output_shape.size();
  for (int i = 0; i < padding; i++) {
    output_shape.insert(output_shape.begin(), 1);
  }

  vector_params->outputLoops[0][0] = output_shape[0];
  vector_params->outputLoops[0][1] = output_shape[1];
  vector_params->outputLoops[0][2] = output_shape[2];
  vector_params->outputLoops[1][0] = output_shape[3];
  vector_params->outputLoops[1][1] = output_shape[4];
  vector_params->outputLoops[1][2] = output_shape[5] / OC_DIMENSION;

  if (vector_params->has_transpose) {
    vector_params->outputAddressMode = 1;

    // Set outer loops
    for (int i = 0; i < 3; i++) {
      vector_params->outputLoops[0][i] = tiling.loops[0][i];
    }
    vector_params->outputYLoopIndex[0] = tiling.y_loop_index[0];
    vector_params->outputXLoopIndex[0] = tiling.x_loop_index[0];
    vector_params->outputWeightLoopIndex[0] = tiling.weight_loop_index[0];

    // Set inner loops
    int loop_index = 0;
    for (int i = 0; i < 6; i++) {
      if (i == tiling.y_loop_index[1]) {
        vector_params->outputLoops[1][loop_index] = tiling.loops[1][i];
        vector_params->outputYLoopIndex[1] = loop_index++;
      }
      if (i == tiling.x_loop_index[1]) {
        vector_params->outputLoops[1][loop_index] = tiling.loops[1][i];
        vector_params->outputXLoopIndex[1] = loop_index++;
      }
      if (i == tiling.weight_loop_index[1]) {
        vector_params->outputLoops[1][loop_index] = tiling.loops[1][i];
        vector_params->outputWeightLoopIndex[1] = loop_index++;
      }
    }
  }

  vector_params->output_vector_type =
      output.dtype() == DataTypes::TypeName<VECTOR_DATATYPE>::name();

  if (output.has_reshape()) {
    vector_params->has_attn_head_permute = output.shape(1) < output.shape(2);
    vector_params->CONCAT_OUTPUT = output.shape(1) > output.shape(2);

    // if we have permutation, we need to configure the address generators
    // accordingly we need to make sure the output is split into 32x32 blocks
    vector_params->outputLoops[1][1] = output.shape(1);
    vector_params->outputLoops[1][2] =
        output.shape(2) * output.shape(3) / OC_DIMENSION;
  }

  VectorInstructions inst;
  memset(&inst, 0, sizeof(inst));
  inst.instType = VectorInstructions::vector;
  inst.vector_op0_src0 = VectorInstructions::from_vector_fetch_0;
  inst.vdest = VectorInstructions::to_output;

  auto inst_map = get_vector_instruction_mapping();

  int curr_stage = 0;

  for (int i = 0; i < op_list.size(); i++) {
    const auto op = op_list[i];
    const std::string opcode = op.target();

    if (opcode == "quantize" || opcode == "dequantize") {
      const auto scale = op.kwargs().at("scale").tensor();

      assert(get_size(scale) == 1);

      // scalar scale factor
      VECTOR_DATATYPE immediate = read_constant_param(scale);

      if (opcode == "quantize") {
        vector_params->quantize_output = true;
        vector_params->output_scale = immediate.bits_rep();
      } else {
        inst.vdequantize = true;
        inst.immediate0 = immediate.bits_rep();
      }
    } else if (opcode == "quantize_mx") {
      vector_params->quantize_output_mx = true;
      vector_params->SCALE_OFFSET =
          param.outputs().tensors(0).memory().address();
    } else {
      if (curr_stage == 4) {
        // we have already processed all the stages
        assert(i == op_list.size() - 1);
        break;
      }

      for (int stage = curr_stage; stage < 4; stage++) {
        if (vector_unit_stages[stage].find(opcode) ==
            vector_unit_stages[stage].end()) {
          continue;
        }

        std::cerr << "stage " << stage << " target: " << opcode << std::endl;

        unsigned int vop = inst_map[opcode];
        if (stage == 0) {
          inst.vector_op0 = vop;
        } else if (stage == 1) {
          inst.vector_op1 = vop;
        } else if (stage == 2) {
          inst.vector_op2 = vop;
        } else if (stage == 3) {
          inst.vector_op3 = vop;
        }

        if (opcode == "vmap") {
          const auto other = op.kwargs().at("other").tensor();
          inst.VMAP_OFFSET = other.memory().address();
        } else if (opcode == "neg") {
          const auto self = op.kwargs().at("input").tensor();
          const auto output_shape = squeeze_shape(get_shape(self));

          inst.vector_op0_src0 = VectorInstructions::from_immediate_0;

          VECTOR_DATATYPE immediate = 0;
          inst.immediate0 = immediate.bits_rep();

          inst.vector_op0_src1 = VectorInstructions::from_vector_fetch_0;

          set_vector_addr_gen1(self, output_shape, accelerator_memory_map,
                               vector_params);
        } else if (op.kwargs().contains("other")) {
          const auto other = op.kwargs().at("other");

          if (other.has_float_value() || other.has_int_value()) {
            float scalar = other.has_float_value() ? other.float_value()
                                                   : other.int_value();
            set_vector_immediate(scalar, stage, opcode, inst);
          } else if (other.has_tensor()) {
            const auto tensor = other.tensor();

            if (get_size(tensor) == 1) {
              float scalar = read_constant_param(tensor);
              set_vector_immediate(scalar, stage, opcode, inst);
            } else {
              const auto self = op.kwargs().at("input").tensor();
              const auto tensor_to_load = tensor.has_memory() ? tensor : self;

              const auto input_shape = get_shape(self);
              const auto other_shape = get_shape(tensor);
              const auto output_shape =
                  squeeze_shape(broadcast_shape(input_shape, other_shape));

              if (stage == 0) {
                inst.vector_op0_src1 = VectorInstructions::from_vector_fetch_1;
                set_vector_addr_gen1(tensor_to_load, output_shape,
                                     accelerator_memory_map, vector_params);
              } else if (stage == 2) {
                inst.vector_op2_src1 = VectorInstructions::from_vector_fetch_2;
                set_vector_addr_gen2(tensor_to_load, output_shape,
                                     accelerator_memory_map, vector_params);
              }
            }
          }
        }

        curr_stage = stage + 1;

        break;
      }
    }
  }

  // total output count
  VectorInstructionConfig *vector_instruction_config =
      new VectorInstructionConfig;
  vector_instruction_config->inst[0] = inst;
  vector_instruction_config->instCount[0] = get_size(output) / OC_DIMENSION;
  vector_instruction_config->instLen = 1;
  vector_instruction_config->instLoopCount = 1;

  mappedParams.push_back(vector_params);
  mappedParams.push_back(vector_instruction_config);
  opMemoryMaps.push_back(accelerator_memory_map);
}