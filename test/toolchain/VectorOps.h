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
    inst.vOp0Src1 = VectorInstructions::op0immediate;
    inst.immediate0 = immediate.bits_rep();
  } else if (stage == 3) {
    inst.vOp3Src1 = VectorInstructions::op3immediate;
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
  std::vector<int> input_shape(input.shape().begin(), input.shape().end());

  // TODO: how to handle reshape operation loop bound adjustment?
  if (!input.has_reshape()) {
    input_shape = split_loops(input_shape, 1024);
    input_shape = adjust_loop_indices(input_shape, OC_DIMENSION);
  }

  // Pad the shape to 6 dimensions with 1s
  const int num_extra_dims = pad_shape_to_ndim(input_shape, 6);

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

  if (reshape_op.target() == "slice") {
    vector_params->has_slicing = true;

    const auto kwargs = reshape_op.kwargs();
    uint64_t start = kwargs.at("start").int_value();
    uint64_t end = kwargs.at("end").int_value();
    uint64_t step = kwargs.at("step").int_value();
    uint64_t dim = kwargs.at("dim").int_value();

    std::vector<int> shape(input.shape().begin(), input.shape().end());

    dim = dim < 0 ? dim + shape.size() : dim;
    end = end > shape[dim] ? shape[dim] : end;

    vector_params->vec0_dim = dim + num_extra_dims;
    vector_params->vec0_start = start;
    vector_params->vec0_end = end;
    vector_params->vec0_step = step;

    // Last dimension needs to be scaled by OC_DIMENSION
    if (vector_params->vec0_dim == 5) {
      vector_params->vec0_start /= OC_DIMENSION;
      vector_params->vec0_end /= OC_DIMENSION;
    }
  }

  if (reshape_op.target() == "permute") {
    vector_params->has_reshape = true;

    const auto dims = reshape_op.kwargs().at("dims").int_list().values();

    for (int i = 0; i < dims.size(); i++) {
      vector_params->vec0_dim_order[i + num_extra_dims] =
          dims[i] + num_extra_dims;
    }
  }

  Tiling tiling;
  if (reshape_op.target() == "transpose") {
    vector_params->has_transpose = true;

    // Only support a buffer size of 32
    const int BUFFER_SIZE = 32;
    std::vector<int> input_shape(input.shape().begin(), input.shape().end());
    int last_dim = input_shape.back();
    input_shape.pop_back();
    input_shape.push_back(last_dim / BUFFER_SIZE);
  }

  VECTOR_DATATYPE scale = 1.0;
  vector_params->vec0_dq_scale = scale.bits_rep();

  // set double precision if the datatype is not the same as the input datatype
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
  inst.vInput = VectorInstructions::readFromVectorFetch;
  inst.vAccumulatePush = VectorInstructions::nop;
  inst.vDest = VectorInstructions::vWriteOut;

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
        inst.vDequantize = true;
        inst.immediate0 = immediate.bits_rep();
      }
    } else if (opcode == "quantize_mx") {
      vector_params->quantize_output_mx = true;
      vector_params->SCALE_OFFSET =
          param.outputs().tensors(0).memory().address();
    } else {
      if (curr_stage == 5) {
        // we have already processed all the stages
        assert(i == op_list.size() - 1);
        break;
      }

      for (int stage = curr_stage; stage < 5; stage++) {
        bool matched = vector_unit_stages[stage].find(opcode) !=
                       vector_unit_stages[stage].end();

        std::cerr << "stage: " << stage << "  opcode: " << opcode
                  << "  matched: " << matched << std::endl;

        if (matched) {
          unsigned int vop = inst_map[opcode];
          if (stage == 0) {
            inst.vOp0 = vop;
          } else if (stage == 1) {
            inst.vOp1 = vop;
          } else if (stage == 2) {
            inst.vOp2 = vop;
          } else if (stage == 3) {
            inst.vOp3 = vop;
          } else if (stage == 4) {
            inst.vOp4 = vop;
          }

          curr_stage = stage + 1;
          if (opcode == "vmap") {
            const auto other = op.kwargs().at("other").tensor();
            inst.vmapOffset = other.memory().address();
          } else if (op.kwargs().contains("other")) {
            const auto other = op.kwargs().at("other");

            if (other.has_float_value() || other.has_int_value()) {
              float scalar = other.has_float_value() ? other.float_value()
                                                     : other.int_value();
              set_immediate(scalar, stage, opcode, inst);
            } else if (other.has_tensor()) {
              const auto tensor = other.tensor();

              if (get_size(tensor) == 1) {
                float scalar = read_constant_param(tensor);
                set_immediate(scalar, stage, opcode, inst);
              } else {
                const auto self = op.kwargs().at("input").tensor();
                const auto tensor_to_load = tensor.has_memory() ? tensor : self;

                const auto input_shape = get_shape(self);
                const auto other_shape = get_shape(tensor);
                const auto output_shape =
                    squeeze_shape(broadcast_shape(input_shape, other_shape));

                if (stage == 0) {
                  inst.vOp0Src1 = VectorInstructions::readInterface;
                  set_vector_addr_gen1(tensor_to_load, output_shape,
                                       accelerator_memory_map, vector_params);
                } else if (stage == 3) {
                  inst.vOp3Src1 = VectorInstructions::readNormalInterface;
                  set_vector_addr_gen2(tensor_to_load, output_shape,
                                       accelerator_memory_map, vector_params);
                }
              }
            }
          }

          break;
        }
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