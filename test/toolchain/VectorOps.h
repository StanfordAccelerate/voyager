#pragma once

#include "src/AccelTypes.h"
#include "src/Params.h"
#include "test/common/GoldModel.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"
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

  return result_shape;
}

void set_vector_addr_gen1(const codegen::Tensor &tensor,
                          const std::vector<int> &output_shape,
                          AcceleratorMemoryMap &accelerator_memory_map,
                          VectorParams *vector_params) {
  int nonzero_dims = 0;
  for (const int &dim : tensor.shape()) {
    if (dim != 1) nonzero_dims++;
  }

  int input_dim = 1;
  for (int i = 0; i < output_shape.size() - 1; i++) {
    input_dim *= output_shape[i];
  }
  int output_dim = output_shape[output_shape.size() - 1];

  const auto memory = tensor.memory();
  accelerator_memory_map["vector1"] = get_partition(memory.partition());
  vector_params->ADDRESS_GEN1_OFFSET = memory.offset();
  vector_params->addressGen1Mode = nonzero_dims == 1 ? 3 : 1;
  // TODO: double precision
  vector_params->DP_VEC1 =
      DataTypes::TypeName<INPUT_DATATYPE>::name() != tensor.dtype();

  for (int i = 0; i < 3; i++) {
    vector_params->addressGen1Loops[0][i] = 1;
  }
  vector_params->addressGen1Loops[1][0] = 1;
  vector_params->addressGen1Loops[1][1] = input_dim;
  vector_params->addressGen1Loops[1][2] = output_dim / (OC_DIMENSION);

  for (int i = 0; i < 2; i++) {
    vector_params->addressGen1InputXLoopIndex[i] = 0;
    vector_params->addressGen1InputYLoopIndex[i] = 1;
    vector_params->addressGen1WeightLoopIndex[i] = 2;
  }

  DataTypes::bfloat16 scale = tensor.scale() != 0 ? tensor.scale() : 1.0;
  vector_params->vec1DequantizeScale = scale.bits_rep();
}

void set_vector_addr_gen2(const codegen::Tensor &tensor,
                          const std::vector<int> &output_shape,
                          AcceleratorMemoryMap &accelerator_memory_map,
                          VectorParams *vector_params) {
  int nonzero_dims = 0;
  for (const int &dim : tensor.shape()) {
    if (dim != 1) nonzero_dims++;
  }

  int input_dim = 1;
  for (int i = 0; i < output_shape.size() - 1; i++) {
    input_dim *= output_shape[i];
  }
  int output_dim = output_shape[output_shape.size() - 1];

  const auto memory = tensor.memory();
  accelerator_memory_map["vector2"] = get_partition(memory.partition());
  vector_params->ADDRESS_GEN2_OFFSET = memory.offset();
  vector_params->addressGen2Mode = nonzero_dims == 1 ? 3 : 1;
  // TODO: double precision
  vector_params->DP_VEC2 = false;

  // copy loop values and indices
  for (int i = 0; i < 3; i++) {
    vector_params->addressGen2Loops[0][i] = 1;
  }
  vector_params->addressGen2Loops[1][0] = 1;
  vector_params->addressGen2Loops[1][1] = input_dim;
  vector_params->addressGen2Loops[1][2] = output_dim / (OC_DIMENSION);

  for (int i = 0; i < 2; i++) {
    vector_params->addressGen2InputXLoopIndex[i] = 0;
    vector_params->addressGen2InputYLoopIndex[i] = 1;
    vector_params->addressGen2WeightLoopIndex[i] = 2;
  }

  DataTypes::bfloat16 scale = tensor.scale() != 0 ? tensor.scale() : 1.0;
  vector_params->vec2DequantizeScale = scale.bits_rep();
}

void MapVectorOperations(const codegen::AcceleratorParam &param,
                         std::deque<BaseParams *> &mappedParams,
                         std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  VectorParams *vector_params = new VectorParams;
  AcceleratorMemoryMap accelerator_memory_map;

  const auto vector_input = param.vector_params(0).input();
  int dim = 1;
  for (int i = 0; i < vector_input.shape_size(); i++) {
    dim *= vector_input.shape(i);
  }

  int dim_factors[2];
  factorizeForAddressGen(dim / OC_DIMENSION, dim_factors);

  const auto input_memory = vector_input.memory();
  accelerator_memory_map["vector0"] = get_partition(input_memory.partition());
  vector_params->VECTOR_OFFSET = input_memory.offset();
  vector_params->addressGen0Mode = 1;
  vector_params->addressGen0Broadcast = false;
  for (int i = 0; i < 3; i++) {
    vector_params->addressGen0Loop[0][i] = 1;
  }
  vector_params->addressGen0Loop[1][0] = 1;
  vector_params->addressGen0Loop[1][1] = dim_factors[0];
  vector_params->addressGen0Loop[1][2] = dim_factors[1];

  for (int i = 0; i < 2; i++) {
    vector_params->addressGen0InputXLoopIndex[i] = 0;
    vector_params->addressGen0InputYLoopIndex[i] = 1;
    vector_params->addressGen0WeightLoopIndex[i] = 2;
  }

  // set double precision if the datatype is not the same as the input datatype
  vector_params->DP_VEC0 =
      DataTypes::TypeName<INPUT_DATATYPE>::name() != vector_input.dtype();

  const auto output_memory = param.output().memory();
  accelerator_memory_map["outputs"] = get_partition(output_memory.partition());
  vector_params->VECTOR_OUTPUT_OFFSET = output_memory.offset();

  for (int i = 0; i < 3; i++) {
    vector_params->outputLoops[0][i] = 1;
  }
  vector_params->outputLoops[1][0] = 1;
  vector_params->outputLoops[1][1] = dim_factors[0];
  vector_params->outputLoops[1][2] = dim_factors[1];

  for (int i = 0; i < 2; i++) {
    vector_params->outputXLoopIndex[i] = 0;
    vector_params->outputYLoopIndex[i] = 1;
    vector_params->outputWeightLoopIndex[i] = 2;
  }

  vector_params->DP_OUTPUT =
      DataTypes::TypeName<INPUT_DATATYPE>::name() != param.output().dtype();
  if (param.output().has_permutation()) {
    vector_params->SPLIT_OUTPUT =
        param.output().shape(1) < param.output().shape(2);
    vector_params->CONCAT_OUTPUT =
        param.output().shape(1) > param.output().shape(2);
  }

  if (param.output().has_permutation()) {
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
  auto it = param.vector_params().begin();
  bool has_vector_params = it != param.vector_params().end();
  std::string output_node = "";

  int vectorStage = 0;
  while (has_vector_params) {
    const auto opcode = it->opcode();

    if (opcode.rfind("dequantize", 0) == 0 ||
        opcode.rfind("quantize", 0) == 0) {
      const int size = get_size(it->other());
      // support only scalar scale factor
      assert(size == 1);
      VECTOR_DATATYPE immediate = read_constant_param(it->other());
      if (opcode.rfind("dequantize", 0) == 0) {
        vinst.immediate0 = immediate.bits_rep();
      } else if (opcode.rfind("quantize", 0) == 0) {
        vector_params->outputQuantizeScale = immediate.bits_rep();
      }

    } else {
      if (vectorStage == 5) {
        // we have already processed all the stages
        break;
      }

      for (int stage = vectorStage; stage < 5; stage++) {
        bool matched =
            vector_ops[stage].find(opcode) != vector_ops[stage].end();
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
          if (it->has_other()) {
            const auto tensor_to_load =
                output_node == it->other().node() ? it->input() : it->other();
            const int size = get_size(tensor_to_load);
            if (size == 1) {
              // TODO: Ideally this should be stroed in the vector param.
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
              const auto input_shape = get_shape(it->input());
              const auto other_shape = get_shape(it->other());
              const auto output_shape =
                  broadcast_shape(input_shape, other_shape);

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
    has_vector_params = it != param.vector_params().end();
  }

  // check that no more vector instructions are present
  if (it != param.vector_params().end()) {
    std::cerr << "Error: unsupported vector fusion pattern" << std::endl;
    exit(1);
  }

  // total output count
  VectorInstructionConfig *vector_instruction_config =
      new VectorInstructionConfig;
  vector_instruction_config->inst[0] = vinst;
  vector_instruction_config->instCount[0] = dim / OC_DIMENSION;
  vector_instruction_config->instLen = 1;
  vector_instruction_config->instLoopCount = 1;

  mappedParams.push_back(vector_params);
  mappedParams.push_back(vector_instruction_config);
  opMemoryMaps.push_back(accelerator_memory_map);
}