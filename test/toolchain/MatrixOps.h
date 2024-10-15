#pragma once

#include "src/AccelTypes.h"
#include "src/Params.h"
#include "test/common/GoldModel.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"
#include "test/toolchain/Common.h"

inline std::vector<int> get_shape(const codegen::Tensor &tensor) {
  auto repeated_field = tensor.shape();
  return std::vector<int>(repeated_field.begin(), repeated_field.end());
}

inline int get_size(const std::vector<int> &shape) {
  int size = 1;
  for (const auto &dim : shape) size *= dim;
  return size;
}

inline int get_size(const codegen::Tensor &tensor) {
  const auto shape = get_shape(tensor);
  return get_size(shape);
}

void set_addr_gen1(const codegen::Tensor &tensor, const Tiling &tiling,
                   AcceleratorMemoryMap &accelerator_memory_map,
                   VectorParams *vector_params) {
  int nonzero_dims = 0;
  for (const int &dim : tensor.shape()) {
    if (dim != 1) nonzero_dims++;
  }

  const auto memory = tensor.memory();
  accelerator_memory_map["vector1"] = get_partition(memory.partition());
  vector_params->ADDRESS_GEN1_OFFSET = memory.offset();
  vector_params->addressGen1Mode = nonzero_dims == 1 ? 3 : 1;
  // TODO: double precision
  vector_params->DP_VEC1 =
      DataTypes::TypeName<INPUT_DATATYPE>::name() != tensor.dtype();

  // copy loop values and indices
  for (int i = 0; i < 3; i++) {
    vector_params->addressGen1Loops[0][i] = tiling.loops[0][i];
  }
  vector_params->addressGen1InputXLoopIndex[0] = tiling.x_loop_index[0];
  vector_params->addressGen1InputYLoopIndex[0] = tiling.y_loop_index[0];
  vector_params->addressGen1WeightLoopIndex[0] = tiling.weight_loop_index[0];

  int loop_index = 0;
  for (int i = 0; i < 6; i++) {
    // ignore the loops not present in outputs (reduction, fx, fy)
    if (i == tiling.weight_loop_index[1] || i == tiling.x_loop_index[1] ||
        i == tiling.y_loop_index[1]) {
      vector_params->addressGen1Loops[1][loop_index] = tiling.loops[1][i];
      if (i == tiling.x_loop_index[1]) {
        vector_params->addressGen1InputXLoopIndex[1] = loop_index;
      }
      if (i == tiling.y_loop_index[1]) {
        vector_params->addressGen1InputYLoopIndex[1] = loop_index;
      }
      if (i == tiling.weight_loop_index[1]) {
        vector_params->addressGen1WeightLoopIndex[1] = loop_index;
      }
      loop_index++;
    }
  }

  DataTypes::bfloat16 scale = tensor.scale() != 0 ? tensor.scale() : 1.0;
  vector_params->vec1DequantizeScale = scale.bits_rep();
}

void set_addr_gen2(const codegen::Tensor &tensor, const Tiling &tiling,
                   AcceleratorMemoryMap &accelerator_memory_map,
                   VectorParams *vector_params) {
  int nonzero_dims = 0;
  for (const int &dim : tensor.shape()) {
    if (dim != 1) nonzero_dims++;
  }

  const auto memory = tensor.memory();
  accelerator_memory_map["vector2"] = get_partition(memory.partition());
  vector_params->ADDRESS_GEN2_OFFSET = memory.offset();
  vector_params->addressGen2Mode = nonzero_dims == 1 ? 3 : 1;
  // TODO: double precision
  vector_params->DP_VEC2 =
      DataTypes::TypeName<INPUT_DATATYPE>::name() != tensor.dtype();

  // copy loop values and indices
  for (int i = 0; i < 3; i++) {
    vector_params->addressGen2Loops[0][i] = tiling.loops[0][i];
  }
  vector_params->addressGen2InputXLoopIndex[0] = tiling.x_loop_index[0];
  vector_params->addressGen2InputYLoopIndex[0] = tiling.y_loop_index[0];
  vector_params->addressGen2WeightLoopIndex[0] = tiling.weight_loop_index[0];

  int loop_index = 0;
  for (int i = 0; i < 6; i++) {
    // ignore the loops not present in outputs (reduction, fx, fy)
    if (i == tiling.weight_loop_index[1] || i == tiling.x_loop_index[1] ||
        i == tiling.y_loop_index[1]) {
      vector_params->addressGen2Loops[1][loop_index] = tiling.loops[1][i];
      if (i == tiling.x_loop_index[1]) {
        vector_params->addressGen2InputXLoopIndex[1] = loop_index;
      }
      if (i == tiling.y_loop_index[1]) {
        vector_params->addressGen2InputYLoopIndex[1] = loop_index;
      }
      if (i == tiling.weight_loop_index[1]) {
        vector_params->addressGen2WeightLoopIndex[1] = loop_index;
      }
      loop_index++;
    }
  }

  DataTypes::bfloat16 scale = tensor.scale() != 0 ? tensor.scale() : 1.0;
  vector_params->vec2DequantizeScale = scale.bits_rep();
}

void MapMatrixOperation(const codegen::AcceleratorParam &param,
                        std::deque<BaseParams *> &mappedParams,
                        std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  Tiling tiling;
  const auto matrix_param = param.matrix_param();
  if (matrix_param.opcode() == "conv2d") {
    tiling = get_conv2d_tiling(param);
  } else {
    tiling = get_linear_tiling(param);
  }

  int X = tiling.loops[0][tiling.x_loop_index[0]] *
          tiling.loops[1][tiling.x_loop_index[1]];
  int Y = tiling.loops[0][tiling.y_loop_index[0]] *
          tiling.loops[1][tiling.y_loop_index[1]];
  int C = tiling.loops[1][tiling.reduction_loop_index[1]] * (16);
  int K = tiling.loops[0][tiling.weight_loop_index[0]] *
          tiling.loops[1][tiling.weight_loop_index[1]] * (16);
  int FX = tiling.loops[1][tiling.fx_index];
  int FY = tiling.loops[1][tiling.fy_index];
  int STRIDE = tiling.stride;

  adjust_tiling_for_dimension(tiling);

  MatrixParams *matrix_params = new MatrixParams;
  AcceleratorMemoryMap accelerator_memory_map;

  // matrix tiling
  const auto input_memory = matrix_param.input().memory();
  accelerator_memory_map["inputs"] = get_partition(input_memory.partition());
  matrix_params->INPUT_OFFSET = input_memory.offset();
  const auto weight_memory = matrix_param.weight().memory();
  accelerator_memory_map["weights"] = get_partition(weight_memory.partition());
  matrix_params->WEIGHT_OFFSET = weight_memory.offset();

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 6; j++) {
      matrix_params->loops[i][j] = tiling.loops[i][j];
    }
    matrix_params->inputXLoopIndex[i] = tiling.x_loop_index[i];
    matrix_params->inputYLoopIndex[i] = tiling.y_loop_index[i];
    matrix_params->reductionLoopIndex[i] = tiling.reduction_loop_index[i];
    matrix_params->weightLoopIndex[i] = tiling.weight_loop_index[i];
    matrix_params->weightReuseIndex[i] = tiling.weight_reuse_index[i];
  }
  matrix_params->fxIndex = tiling.fx_index;
  matrix_params->fyIndex = tiling.fy_index;

  // set outer loop values
  for (int j = 0; j < 5; j++) {
    matrix_params->weightAddressGenLoops[0][j] = tiling.loops[0][j];
  }
  // matrix_params->weightAddressGenInputXLoopIndex = tiling.x_loop_index[0];
  // matrix_params->weightAddressGenInputYLoopIndex = tiling.y_loop_index[0];
  matrix_params->weightAddressGenWeightLoopIndex[0] =
      tiling.weight_loop_index[0];

  // set outer loop values
  const auto weight = matrix_param.weight();
  matrix_params->WEIGHT_TRANSPOSE =
      weight.permutation().opcode() == "transpose";
  if (matrix_params->WEIGHT_TRANSPOSE) {
    // for tranpose, we need to enforce that the innermost loop is the
    // unrolled reduction loop
    // we can just use the following loop nest:
    // C1, K, FY, FX, C0
    matrix_params->weightAddressGenLoops[1][4] = OC_DIMENSION;
    matrix_params->weightAddressGenReductionLoopIndex[1] = 4;
    matrix_params->weightAddressGenLoops[1][3] =
        tiling.loops[1][tiling.fx_index];
    matrix_params->weightAddressGenFxIndex = 3;
    matrix_params->weightAddressGenLoops[1][2] =
        tiling.loops[1][tiling.fy_index];
    matrix_params->weightAddressGenFyIndex = 2;
    matrix_params->weightAddressGenLoops[1][1] =
        tiling.loops[1][tiling.weight_loop_index[1]];

    if (OC_DIMENSION > IC_DIMENSION) {
      // matrix_params->weightAddressGenLoops[1][1] =
      //     tiling.loops[1][tiling.weight_loop_index[1]] /
      //     (OC_DIMENSION / IC_DIMENSION);
    }

    matrix_params->weightAddressGenWeightLoopIndex[1] = 1;
    matrix_params->weightAddressGenLoops[1][0] =
        tiling.loops[1][tiling.reduction_loop_index[1]];

    if (OC_DIMENSION > IC_DIMENSION) {
      // we can reduce the number of iterations, since we have already fetched
      // the values
      matrix_params->weightAddressGenLoops[1][0] =
          tiling.loops[1][tiling.reduction_loop_index[1]] /
          (OC_DIMENSION / IC_DIMENSION);
    }
    matrix_params->weightAddressGenReductionLoopIndex[0] = 0;
  } else {  // if not tranpose, then we have freedom to pick any loop order
    // for efficient memory accesses, addresses should be consecutive
    // or least, not multiples of 4, due to interleaving.
    // given that weights are arranged as: FY,FX,C,K
    // the following loop nest should work:
    // C1, C0, FX, FY, K
    // int index = 0;
    // for (int j = 0; j < 6; j++) {
    //   if (j == matrix_params->inputXLoopIndex[1] ||
    //       j == matrix_params->inputYLoopIndex[1]) {
    //     continue;
    //   }
    //   matrix_params->weightAddressGenLoops[1][index] = tiling.loops[1][j];

    //   if (j == matrix_params->reductionLoopIndex[1]) {
    //     matrix_params->weightAddressGenReductionLoopIndex[0] = index;
    //   }
    //   if (j == matrix_params->fxIndex) {
    //     matrix_params->weightAddressGenFxIndex = index;
    //   }
    //   if (j == matrix_params->fyIndex) {
    //     matrix_params->weightAddressGenFyIndex = index;
    //   }
    //   if (j == matrix_params->weightLoopIndex[1]) {
    //     matrix_params->weightAddressGenWeightLoopIndex[1] = index;
    //   }

    //   index++;
    // }
    // matrix_params->weightAddressGenLoops[1][4] = DIMENSION;
    // matrix_params->weightAddressGenReductionLoopIndex[1] = 4;

    matrix_params->weightAddressGenLoops[1][4] =
        tiling.loops[1][tiling.weight_loop_index[1]];
    matrix_params->weightAddressGenWeightLoopIndex[1] = 4;

    matrix_params->weightAddressGenLoops[1][3] =
        tiling.loops[1][tiling.fy_index];
    matrix_params->weightAddressGenFyIndex = 3;

    matrix_params->weightAddressGenLoops[1][2] =
        tiling.loops[1][tiling.fx_index];
    if (tiling.replication) {
      matrix_params->weightAddressGenLoops[1][2] = 7;
    }
    matrix_params->weightAddressGenFxIndex = 2;

    if (tiling.replication) {
      matrix_params->weightAddressGenLoops[1][1] = 3;
      matrix_params->weightAddressGenReductionLoopIndex[1] = 1;
    } else {
      matrix_params->weightAddressGenLoops[1][1] = IC_DIMENSION;
      matrix_params->weightAddressGenReductionLoopIndex[1] = 1;
    }
    matrix_params->weightAddressGenLoops[1][0] =
        tiling.loops[1][tiling.reduction_loop_index[1]];
    matrix_params->weightAddressGenReductionLoopIndex[0] = 0;
  }

  matrix_params->STRIDE = tiling.stride;
  matrix_params->REPLICATION = tiling.replication;
  // TODO:
  matrix_params->STORE_IN_ACC = false;
  matrix_params->ACC_FROM_ACC = false;

  // Permute input for transformer attention outputs
  const auto input = matrix_param.input();
  const auto permutation = input.permutation();
  matrix_params->CONCAT_INPUT = permutation.opcode() == "permute";
  // matrix_params->CONCAT_HEAD_WEIGHTS = false;
  matrix_params->TRANPOSE_INPUTS = permutation.opcode() == "transpose";

  // bias
  const auto bias_memory = matrix_param.bias().memory();
  matrix_params->BIAS = matrix_param.has_bias();
  matrix_params->BIAS_OFFSET = bias_memory.offset();
  accelerator_memory_map["bias"] = get_partition(bias_memory.partition());

  // vector instructions
  VectorParams *vector_params = new VectorParams;
  vector_params->addressGen0Mode = 0;  // use matrix unit outputs

  const auto output_memory = param.output().memory();
  accelerator_memory_map["outputs"] = get_partition(output_memory.partition());
  vector_params->VECTOR_OUTPUT_OFFSET = output_memory.offset();

  for (int i = 0; i < 3; i++) {
    vector_params->outputLoops[0][i] = tiling.loops[0][i];
  }
  vector_params->outputXLoopIndex[0] = tiling.x_loop_index[0];
  vector_params->outputYLoopIndex[0] = tiling.y_loop_index[0];
  vector_params->outputWeightLoopIndex[0] = tiling.weight_loop_index[0];

  int outputLoopIndex = 0;
  for (int i = 0; i < 6; i++) {
    // ignore the loops not present in outputs (reduction, fx, fy)
    if (i == tiling.weight_loop_index[1] || i == tiling.x_loop_index[1] ||
        i == tiling.y_loop_index[1]) {
      vector_params->outputLoops[1][outputLoopIndex] = tiling.loops[1][i];
      if (i == tiling.x_loop_index[1]) {
        vector_params->outputXLoopIndex[1] = outputLoopIndex;
      }
      if (i == tiling.y_loop_index[1]) {
        vector_params->outputYLoopIndex[1] = outputLoopIndex;
      }
      if (i == tiling.weight_loop_index[1]) {
        vector_params->outputWeightLoopIndex[1] = outputLoopIndex;
      }
      outputLoopIndex++;
    }
  }

  vector_params->DP_OUTPUT =
      DataTypes::TypeName<INPUT_DATATYPE>::name() != param.output().dtype();

  // TODO: Transformer qkv output permutation
  vector_params->SPLIT_OUTPUT = param.output().has_permutation();

  VectorInstructions vinst;
  memset(&vinst, 0, sizeof(vinst));
  vinst.instType = VectorInstructions::vector;
  vinst.vInput = VectorInstructions::readFromSystolicArray;
  vinst.vAccumulatePush = VectorInstructions::nop;
  vinst.vDest = VectorInstructions::vWriteOut;

  auto vinst_mappings = get_vector_instruction_mapping();
  auto it = param.vector_params().begin();
  bool has_vector_params = it != param.vector_params().end();
  std::string output_node = matrix_param.name();

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
        vinst.vDequantizeScale = immediate.bits_rep();
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
          // set the opcode for the vector stage
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

          // increment the stage for the next operation
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
              if (stage == 0) {
                vinst.vOp0Src1 = VectorInstructions::readInterface;
                set_addr_gen1(tensor_to_load, tiling, accelerator_memory_map,
                              vector_params);
              } else if (stage == 3) {
                vinst.vOp3Src1 = VectorInstructions::readNormalInterface;
                set_addr_gen2(tensor_to_load, tiling, accelerator_memory_map,
                              vector_params);
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
  vector_instruction_config->instCount[0] =
      tiling.loops[0][tiling.x_loop_index[0]] *
      tiling.loops[1][tiling.x_loop_index[1]] *
      tiling.loops[0][tiling.y_loop_index[0]] *
      tiling.loops[1][tiling.y_loop_index[1]] *
      tiling.loops[0][tiling.weight_loop_index[0]] *
      tiling.loops[1][tiling.weight_loop_index[1]];
  vector_instruction_config->instLen = 1;
  vector_instruction_config->instLoopCount = 1;

  mappedParams.push_back(matrix_params);
  mappedParams.push_back(vector_params);
  mappedParams.push_back(vector_instruction_config);
  opMemoryMaps.push_back(accelerator_memory_map);
}
