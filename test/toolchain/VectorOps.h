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
  vector_params->ADDRESS_GEN1_OFFSET = memory.offset();

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

  vector_params->DP_VEC1 =
      (DataTypes::TypeName<INPUT_DATATYPE>::name() != tensor.dtype()) &&
      ((DataTypes::TypeName<MX_DATATYPE>::name() != tensor.dtype()));

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
  vector_params->vec1DequantizeScale = scale.bits_rep();
}

void set_vector_addr_gen2(const codegen::Tensor &tensor,
                          const std::vector<int> &output_shape,
                          AcceleratorMemoryMap &accelerator_memory_map,
                          VectorParams *vector_params) {
  const auto memory = tensor.memory();
  accelerator_memory_map["vector2"] = get_partition(memory.partition());
  vector_params->ADDRESS_GEN2_OFFSET = memory.offset();

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

  vector_params->DP_VEC2 =
      (DataTypes::TypeName<INPUT_DATATYPE>::name() != tensor.dtype()) &&
      ((DataTypes::TypeName<MX_DATATYPE>::name() != tensor.dtype()));

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
  vector_params->vec2DequantizeScale = scale.bits_rep();
}

void MapVectorOperations(const codegen::Operator &param,
                         std::deque<BaseParams *> &mappedParams,
                         std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  VectorParams *vector_params = new VectorParams;
  AcceleratorMemoryMap accelerator_memory_map;

  codegen::Tensor vector_input;
  if (param.has_slicing_op()) {
    vector_input = param.slicing_op().input();
  } else if (param.has_reshape_op()) {
    vector_input = param.reshape_op().input();
  } else {
    vector_input = param.vector_ops(0).input();
  }

  const auto vector_output = param.output();
  const int numel = get_size(vector_output);

  const auto input_memory = vector_input.memory();
  accelerator_memory_map["vector0"] = get_partition(input_memory.partition());
  vector_params->VECTOR_OFFSET = input_memory.offset();
  vector_params->addressGen0Mode = 2;

  auto loop_bounds = get_tensor_shape(vector_input);

  // TODO: how to handle reshape operation loop bound adjustment?
  if (!vector_input.has_reshape()) {
    loop_bounds = split_loops(loop_bounds, 1024);
    loop_bounds = adjust_loop_indices(loop_bounds, OC_DIMENSION);
  }

  // Pad the shape to 6 dimensions with 1s
  const int num_extra_dims = pad_shape_to_ndim(loop_bounds, 6);

  vector_params->addressGen0Loop[0][0] = loop_bounds[0];
  vector_params->addressGen0Loop[0][1] = loop_bounds[1];
  vector_params->addressGen0Loop[0][2] = loop_bounds[2];
  vector_params->addressGen0Loop[1][0] = loop_bounds[3];
  vector_params->addressGen0Loop[1][1] = loop_bounds[4];
  vector_params->addressGen0Loop[1][2] = loop_bounds[5] / OC_DIMENSION;

  if (param.has_slicing_op() || vector_input.has_slicing()) {
    const auto &slicing =
        param.has_slicing_op() ? param.slicing_op() : vector_input.slicing();
    vector_params->VECTOR_INPUT0_SLICING = true;
    vector_params->addressGen0Dim = slicing.dim() + num_extra_dims;
    vector_params->addressGen0Start = slicing.start();
    vector_params->addressGen0End = slicing.end();
    vector_params->addressGen0Stride = slicing.step();

    // Last dimension needs to be scaled by OC_DIMENSION
    if (vector_params->addressGen0Dim == 5) {
      vector_params->addressGen0Start /= OC_DIMENSION;
      vector_params->addressGen0End /= OC_DIMENSION;
    }
  } else if (param.has_reshape_op() || vector_input.has_reshape()) {
    const auto &reshape =
        param.has_reshape_op() ? param.reshape_op() : vector_input.reshape();
    vector_params->VECTOR_INPUT0_RESHAPE = true;
    if (reshape.opcode() == "permute") {
      for (int i = 0; i < reshape.dims_size(); i++) {
        vector_params->addressGen0AxisOrder[i + num_extra_dims] =
            reshape.dims(i) + num_extra_dims;
      }
    } else {
      throw std::invalid_argument("Unsupported reshape operation!");
    }
  }

  // set double precision if the datatype is not the same as the input datatype
  vector_params->DP_VEC0 =
      DataTypes::TypeName<INPUT_DATATYPE>::name() != vector_input.dtype();

  const auto output_memory = vector_output.memory();
  accelerator_memory_map["outputs"] = get_partition(output_memory.partition());
  vector_params->VECTOR_OUTPUT_OFFSET = output_memory.offset();
  vector_params->outputAddressMode = 2;

  auto output_shape = get_tensor_shape(vector_output);
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

  vector_params->DP_OUTPUT =
      DataTypes::TypeName<INPUT_DATATYPE>::name() != param.output().dtype();

  if (param.output().has_reshape()) {
    vector_params->SPLIT_OUTPUT =
        param.output().shape(1) < param.output().shape(2);
    vector_params->CONCAT_OUTPUT =
        param.output().shape(1) > param.output().shape(2);

    // if we have permutation, we need to configure the address generators
    // accordingly we need to make sure the output is split into 32x32 blocks
    vector_params->outputLoops[1][1] = param.output().shape(1);
    vector_params->outputLoops[1][2] =
        param.output().shape(2) * param.output().shape(3) / OC_DIMENSION;
  }

  VectorInstructions vinst;
  memset(&vinst, 0, sizeof(vinst));
  vinst.instType = VectorInstructions::vector;
  vinst.vInput = VectorInstructions::readFromVectorFetch;
  vinst.vAccumulatePush = VectorInstructions::nop;
  vinst.vDest = VectorInstructions::vWriteOut;

  auto vinst_mappings = get_vector_instruction_mapping();
  auto it = param.vector_ops().begin();
  bool has_vector_params = it != param.vector_ops().end();
  std::string output_node = "";

  int vectorStage = 0;
  while (has_vector_params) {
    const auto opcode = it->opcode();

    if (opcode.rfind("dequantize", 0) == 0 ||
        opcode.rfind("quantize", 0) == 0) {
      const auto tensor_to_load =
          output_node == it->other().node() ? it->input() : it->other();
      const int size = get_size(tensor_to_load);

      if (size == 1) {  // scalar scale factor
        VECTOR_DATATYPE immediate = read_constant_param(tensor_to_load);

        if (opcode.rfind("dequantize", 0) == 0) {
          vinst.immediate0 = immediate.bits_rep();
        } else if (opcode.rfind("quantize", 0) == 0) {
          vector_params->OUTPUT_QUANTIZE = true;
          vector_params->outputQuantizeScale = immediate.bits_rep();
        }
      } else {  // microscaling
        auto other_shape = get_tensor_shape(it->other());

        // change other_shape to 2D tensor
        int total_size = 1;
        for (int i = 0; i < other_shape.size(); i++) {
          total_size *= other_shape[i];
        }

        int other_dim_factors[2];
        factorizeForAddressGen(total_size / OC_DIMENSION, other_dim_factors);

        other_dim_factors[1] *= OC_DIMENSION;
        other_shape = {other_dim_factors[0], other_dim_factors[1]};

        set_vector_addr_gen1(tensor_to_load, other_shape,
                             accelerator_memory_map, vector_params);

        vector_params->BROADCAST_VEC1_SCALE = true;
        vector_params->vec1BroadcastCount = numel / size / OC_DIMENSION;
        vector_params->OUTPUT_QUANTIZE_MX = true;
      }

    } else {
      if (vectorStage == 5) {
        // we have already processed all the stages
        break;
      }

      for (int stage = vectorStage; stage < 5; stage++) {
        bool matched = vector_unit_stages[stage].find(opcode) !=
                       vector_unit_stages[stage].end();
        std::cerr << "stage: " << stage << "  opcode: " << opcode
                  << "  matched: " << matched << std::endl;
        if (matched) {
          unsigned int vop = vinst_mappings[opcode];

          if (stage == 0) {
            vinst.vOp0 = vop;
          } else if (stage == 1) {
            vinst.vOp1 = vop;
          } else if (stage == 2) {
            vinst.vOp2 = vop;
          } else if (stage == 3) {
            vinst.vOp3 = vop;
          } else if (stage == 4) {
            vinst.vOp4 = vop;
          }

          vectorStage = stage + 1;
          if (it->opcode() == "vmap") {
            vinst.vmapOffset = it->other().memory().offset();
          } else if (it->has_other_scalar()) {
            std::cerr << "immediate: " << it->other_scalar() << std::endl;
            VECTOR_DATATYPE immediate = it->other_scalar();

            if (it->opcode() == "div" || it->opcode() == "div_") {
              immediate = 1.0 / immediate;
            }

            if (stage == 0) {
              vinst.vOp0Src1 = VectorInstructions::op0immediate;
              vinst.immediate0 = immediate.bits_rep();
            } else if (stage == 3) {
              vinst.vOp3Src1 = VectorInstructions::op3immediate;
              vinst.immediate1 = immediate.bits_rep();
            } else if (stage == 5) {
              vinst.immediate1 = immediate.bits_rep();
            }
          } else if (it->has_other()) {
            const auto tensor_to_load =
                output_node == it->other().node() ? it->input() : it->other();
            const int size = get_size(tensor_to_load);
            if (size == 1) {
              VECTOR_DATATYPE immediate = read_constant_param(tensor_to_load);

              if (it->opcode() == "div" || it->opcode() == "div_") {
                immediate = 1.0 / immediate;
              }

              if (stage == 0) {
                vinst.vOp0Src1 = VectorInstructions::op0immediate;
                vinst.immediate0 = immediate.bits_rep();
              } else if (stage == 3) {
                vinst.vOp3Src1 = VectorInstructions::op3immediate;
                vinst.immediate1 = immediate.bits_rep();
              } else if (stage == 5) {
                vinst.immediate1 = immediate.bits_rep();
              }
            } else {
              const auto input_shape = get_input_shape(it->input());
              const auto other_shape = get_input_shape(it->other());
              const auto output_shape =
                  squeeze_shape(broadcast_shape(input_shape, other_shape));

              if (stage == 0) {
                vinst.vOp0Src1 = VectorInstructions::readInterface;
                set_vector_addr_gen1(tensor_to_load, output_shape,
                                     accelerator_memory_map, vector_params);
              } else if (stage == 3) {
                vinst.vOp3Src1 = VectorInstructions::readNormalInterface;
                set_vector_addr_gen2(tensor_to_load, output_shape,
                                     accelerator_memory_map, vector_params);
              }
            }
          }

          break;
        }
      }
    }

    output_node = it->name();
    ++it;
    has_vector_params = it != param.vector_ops().end();
  }

  // check that no more vector instructions are present
  if (it != param.vector_ops().end()) {
    std::cerr << "Error: unsupported vector fusion pattern" << std::endl;
    exit(1);
  }

  // total output count
  VectorInstructionConfig *vector_instruction_config =
      new VectorInstructionConfig;
  vector_instruction_config->inst[0] = vinst;
  vector_instruction_config->instCount[0] = numel / OC_DIMENSION;
  vector_instruction_config->instLen = 1;
  vector_instruction_config->instLoopCount = 1;

  mappedParams.push_back(vector_params);
  mappedParams.push_back(vector_instruction_config);
  opMemoryMaps.push_back(accelerator_memory_map);
}