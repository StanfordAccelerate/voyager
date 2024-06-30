#pragma once

#include "src/AccelTypes.h"
#include "src/Params.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"

std::map<int, std::set<std::string>> vector_ops = {
    {0, {"add", "add_", "sub", "sub_", "mult", "mult_"}},
    {1, {"exp"}},
    {2, {}},
    {3, {"add", "add_", "mult", "mult_", "square"}},
    {4, {"relu", "relu_"}},
};

std::map<std::string, unsigned int> get_vector_instruction_mapping() {
  std::map<std::string, unsigned int> mapping;
  mapping["add"] = VectorInstructions::vadd;
  mapping["add_"] = VectorInstructions::vadd;
  mapping["sub"] = VectorInstructions::vsub;
  mapping["sub_"] = VectorInstructions::vsub;
  mapping["mult"] = VectorInstructions::vmult;
  mapping["mult_"] = VectorInstructions::vmult;
  mapping["relu"] = VectorInstructions::vrelu;
  mapping["relu_"] = VectorInstructions::vrelu;
  mapping["exp"] = VectorInstructions::vexp;
  mapping["square"] = VectorInstructions::vsquare;
  return mapping;
}

inline MemorySource get_partition(const int &partition) {
  return partition == 0 ? SRAM : RRAM;
}

void set_addr_gen1(const codegen::Tensor &tensor, const Tiling &tiling,
                   AcceleratorMemoryMap &accelerator_memory_map,
                   VectorParams *accelerator_vector_param) {
  const auto memory = tensor.memory();
  accelerator_memory_map["vector1"] = get_partition(memory.partition());
  accelerator_vector_param->ADDRESS_GEN1_OFFSET = memory.offset();
  // TODO: broadcasting
  accelerator_vector_param->addressGen1Mode = 1;
  // TODO: double precision
  accelerator_vector_param->DP_VEC1 = false;

  // copy loop values and indices
  for (int i = 0; i < 3; i++) {
    accelerator_vector_param->addressGen1Loops[0][i] = tiling.loops[0][i];
  }
  accelerator_vector_param->addressGen1InputXLoopIndex[0] =
      tiling.x_loop_index[0];
  accelerator_vector_param->addressGen1InputYLoopIndex[0] =
      tiling.y_loop_index[0];
  accelerator_vector_param->addressGen1WeightLoopIndex[0] =
      tiling.weight_loop_index[0];

  int loop_index = 0;
  for (int i = 0; i < 6; i++) {
    // ignore the loops not present in outputs (reduction, fx, fy)
    if (i == tiling.weight_loop_index[1] || i == tiling.x_loop_index[1] ||
        i == tiling.y_loop_index[1]) {
      accelerator_vector_param->addressGen1Loops[1][loop_index] =
          tiling.loops[1][i];
      if (i == tiling.x_loop_index[1]) {
        accelerator_vector_param->addressGen1InputXLoopIndex[1] = loop_index;
      }
      if (i == tiling.y_loop_index[1]) {
        accelerator_vector_param->addressGen1InputYLoopIndex[1] = loop_index;
      }
      if (i == tiling.weight_loop_index[1]) {
        accelerator_vector_param->addressGen1WeightLoopIndex[1] = loop_index;
      }
      loop_index++;
    }
  }
}

void set_addr_gen2(const codegen::Tensor &tensor, const Tiling &tiling,
                   AcceleratorMemoryMap &accelerator_memory_map,
                   VectorParams *accelerator_vector_param) {
  const auto memory = tensor.memory();
  accelerator_memory_map["vector2"] = get_partition(memory.partition());
  accelerator_vector_param->ADDRESS_GEN2_OFFSET = memory.offset();
  // TODO: broadcasting
  accelerator_vector_param->addressGen2Mode = 1;
  // TODO: double precision
  accelerator_vector_param->DP_VEC2 = false;

  // copy loop values and indices
  for (int i = 0; i < 3; i++) {
    accelerator_vector_param->addressGen2Loops[0][i] = tiling.loops[0][i];
  }
  accelerator_vector_param->addressGen2InputXLoopIndex[0] =
      tiling.x_loop_index[0];
  accelerator_vector_param->addressGen2InputYLoopIndex[0] =
      tiling.y_loop_index[0];
  accelerator_vector_param->addressGen2WeightLoopIndex[0] =
      tiling.weight_loop_index[0];

  int loop_index = 0;
  for (int i = 0; i < 6; i++) {
    // ignore the loops not present in outputs (reduction, fx, fy)
    if (i == tiling.weight_loop_index[1] || i == tiling.x_loop_index[1] ||
        i == tiling.y_loop_index[1]) {
      accelerator_vector_param->addressGen2Loops[1][loop_index] =
          tiling.loops[1][i];
      if (i == tiling.x_loop_index[1]) {
        accelerator_vector_param->addressGen2InputXLoopIndex[1] = loop_index;
      }
      if (i == tiling.y_loop_index[1]) {
        accelerator_vector_param->addressGen2InputYLoopIndex[1] = loop_index;
      }
      if (i == tiling.weight_loop_index[1]) {
        accelerator_vector_param->addressGen2WeightLoopIndex[1] = loop_index;
      }
      loop_index++;
    }
  }
}

void MapMatrixOperation(const codegen::AcceleratorParam &param,
                        std::deque<BaseParams *> &mappedParams,
                        std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  Tiling tiling;
  const auto matrix_param = param.matrix_param();
  if (matrix_param.opcode() == "conv2d") {
    tiling = get_conv_tiling(matrix_param);
  } else if (matrix_param.opcode() == "linear") {
    tiling = get_linear_tiling(matrix_param);
  } else if (matrix_param.opcode() == "matmul") {
    tiling = get_matmul_tiling(matrix_param);
  } else {
    std::cerr << "Unsupported matrix instruction: " << matrix_param.opcode()
              << std::endl;
    exit(1);
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

  if (IC_DIMENSION < 16) {
    tiling.loops[1][tiling.reduction_loop_index[1]] *= (16 / IC_DIMENSION);
  } else if (IC_DIMENSION > 16) {
    if (!tiling.replication) {
      tiling.loops[1][tiling.reduction_loop_index[1]] /= (IC_DIMENSION / 16);
    }
  }

  if (OC_DIMENSION < 16) {
    tiling.loops[0][tiling.weight_loop_index[0]] *= (16 / OC_DIMENSION);
  } else if (OC_DIMENSION > 16) {
    if ((tiling.loops[1][tiling.weight_loop_index[1]] >= 4 &&
         tiling.loops[1][tiling.fx_index] > 1 &&
         tiling.loops[1][tiling.fy_index] > 1)) {
      tiling.loops[1][tiling.weight_loop_index[1]] /= (OC_DIMENSION / 16);
    } else if (tiling.loops[0][tiling.weight_loop_index[0]] <
                   (OC_DIMENSION / 16) &&
               tiling.loops[0][tiling.weight_loop_index[0]] != 1) {
      int reductionFactor = OC_DIMENSION / 16;
      reductionFactor =
          reductionFactor / tiling.loops[0][tiling.weight_loop_index[0]];
      tiling.loops[0][tiling.weight_loop_index[0]] = 1;
      tiling.loops[1][tiling.weight_loop_index[1]] /= reductionFactor;
    } else if (tiling.loops[0][tiling.weight_loop_index[0]] == 1) {
      tiling.loops[1][tiling.weight_loop_index[1]] /= (OC_DIMENSION / 16);
    } else {
      tiling.loops[0][tiling.weight_loop_index[0]] /= (OC_DIMENSION / 16);
    }
  }

  MatrixParams *matrixParams = new MatrixParams;
  AcceleratorMemoryMap accelerator_memory_map;

  // matrix tiling
  const auto input_memory = matrix_param.input().memory();
  accelerator_memory_map["inputs"] = get_partition(input_memory.partition());
  matrixParams->INPUT_OFFSET = input_memory.offset();
  const auto weight_memory = matrix_param.weight().memory();
  accelerator_memory_map["weights"] = get_partition(weight_memory.partition());
  matrixParams->WEIGHT_OFFSET = weight_memory.offset();
  // TODO: PT2E IR does not have this field
  matrixParams->WEIGHT_TRANSPOSE = false;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 6; j++) {
      matrixParams->loops[i][j] = tiling.loops[i][j];
    }
    matrixParams->inputXLoopIndex[i] = tiling.x_loop_index[i];
    matrixParams->inputYLoopIndex[i] = tiling.y_loop_index[i];
    matrixParams->reductionLoopIndex[i] = tiling.reduction_loop_index[i];
    matrixParams->weightLoopIndex[i] = tiling.weight_loop_index[i];
    matrixParams->weightReuseIndex[i] = tiling.weight_reuse_index[i];
  }
  matrixParams->fxIndex = tiling.fx_index;
  matrixParams->fyIndex = tiling.fy_index;

  // set outer loop values
  for (int j = 0; j < 5; j++) {
    matrixParams->weightAddressGenLoops[0][j] = tiling.loops[0][j];
  }
  // matrixParams->weightAddressGenInputXLoopIndex = tiling.x_loop_index[0];
  // matrixParams->weightAddressGenInputYLoopIndex = tiling.y_loop_index[0];
  matrixParams->weightAddressGenWeightLoopIndex[0] =
      tiling.weight_loop_index[0];

  // set outer loop values
  // if (params.WEIGHT_TRANSPOSE) {
  if (false) {
    // for tranpose, we need to enforce that the innermost loop is the
    // unrolled reduction loop
    // we can just use the following loop nest:
    // C1, K, FY, FX, C0
    matrixParams->weightAddressGenLoops[1][4] = OC_DIMENSION;
    matrixParams->weightAddressGenReductionLoopIndex[1] = 4;
    matrixParams->weightAddressGenLoops[1][3] =
        tiling.loops[1][tiling.fx_index];
    matrixParams->weightAddressGenFxIndex = 3;
    matrixParams->weightAddressGenLoops[1][2] =
        tiling.loops[1][tiling.fy_index];
    matrixParams->weightAddressGenFyIndex = 2;
    matrixParams->weightAddressGenLoops[1][1] =
        tiling.loops[1][tiling.weight_loop_index[1]];

    if (OC_DIMENSION > IC_DIMENSION) {
      // matrixParams->weightAddressGenLoops[1][1] =
      //     tiling.loops[1][tiling.weight_loop_index[1]] /
      //     (OC_DIMENSION / IC_DIMENSION);
    }

    matrixParams->weightAddressGenWeightLoopIndex[1] = 1;
    matrixParams->weightAddressGenLoops[1][0] =
        tiling.loops[1][tiling.reduction_loop_index[1]];

    if (OC_DIMENSION > IC_DIMENSION) {
      // we can reduce the number of iterations, since we have already fetched
      // the values
      matrixParams->weightAddressGenLoops[1][0] =
          tiling.loops[1][tiling.reduction_loop_index[1]] /
          (OC_DIMENSION / IC_DIMENSION);
    }
    matrixParams->weightAddressGenReductionLoopIndex[0] = 0;
  } else {  // if not tranpose, then we have freedom to pick any loop order
    // for efficient memory accesses, addresses should be consecutive
    // or least, not multiples of 4, due to interleaving.
    // given that weights are arranged as: FY,FX,C,K
    // the following loop nest should work:
    // C1, C0, FX, FY, K
    // int index = 0;
    // for (int j = 0; j < 6; j++) {
    //   if (j == matrixParams->inputXLoopIndex[1] ||
    //       j == matrixParams->inputYLoopIndex[1]) {
    //     continue;
    //   }
    //   matrixParams->weightAddressGenLoops[1][index] = tiling.loops[1][j];

    //   if (j == matrixParams->reductionLoopIndex[1]) {
    //     matrixParams->weightAddressGenReductionLoopIndex[0] = index;
    //   }
    //   if (j == matrixParams->fxIndex) {
    //     matrixParams->weightAddressGenFxIndex = index;
    //   }
    //   if (j == matrixParams->fyIndex) {
    //     matrixParams->weightAddressGenFyIndex = index;
    //   }
    //   if (j == matrixParams->weightLoopIndex[1]) {
    //     matrixParams->weightAddressGenWeightLoopIndex[1] = index;
    //   }

    //   index++;
    // }
    // matrixParams->weightAddressGenLoops[1][4] = DIMENSION;
    // matrixParams->weightAddressGenReductionLoopIndex[1] = 4;

    matrixParams->weightAddressGenLoops[1][4] =
        tiling.loops[1][tiling.weight_loop_index[1]];
    matrixParams->weightAddressGenWeightLoopIndex[1] = 4;

    matrixParams->weightAddressGenLoops[1][3] =
        tiling.loops[1][tiling.fy_index];
    matrixParams->weightAddressGenFyIndex = 3;

    matrixParams->weightAddressGenLoops[1][2] =
        tiling.loops[1][tiling.fx_index];
    if (tiling.replication) {
      matrixParams->weightAddressGenLoops[1][2] = 7;
    }
    matrixParams->weightAddressGenFxIndex = 2;

    if (tiling.replication) {
      matrixParams->weightAddressGenLoops[1][1] = 3;
      matrixParams->weightAddressGenReductionLoopIndex[1] = 1;
    } else {
      matrixParams->weightAddressGenLoops[1][1] = IC_DIMENSION;
      matrixParams->weightAddressGenReductionLoopIndex[1] = 1;
    }
    matrixParams->weightAddressGenLoops[1][0] =
        tiling.loops[1][tiling.reduction_loop_index[1]];
    matrixParams->weightAddressGenReductionLoopIndex[0] = 0;
  }

  matrixParams->STRIDE = tiling.stride;
  // matrixParams->HEAD_SIZE_LG2 = 0;
  matrixParams->REPLICATION = tiling.replication;
  // TODO: PT2E IR does not have these fields
  matrixParams->STORE_IN_ACC = false;
  matrixParams->ACC_FROM_ACC = false;
  matrixParams->CONCAT_INPUT = false;
  matrixParams->CONCAT_HEAD_WEIGHTS = false;
  matrixParams->TRANPOSE_INPUTS = false;

  // bias
  const auto bias_memory = matrix_param.bias().memory();
  matrixParams->BIAS = matrix_param.has_bias();
  matrixParams->BIAS_OFFSET = bias_memory.offset();
  accelerator_memory_map["bias"] = get_partition(bias_memory.partition());

  // vector instructions
  VectorParams *accelerator_vector_param = new VectorParams;
  accelerator_vector_param->VECTOR_OFFSET = input_memory.offset();
  accelerator_vector_param->addressGen0Mode = false;  // use matrix unit outputs

  VectorInstructions vector_instr;
  memset(&vector_instr, 0, sizeof(vector_instr));
  vector_instr.instType = VectorInstructions::vector;
  vector_instr.vInput = VectorInstructions::readFromSystolicArray;
  vector_instr.vAccumulatePush = 0;

  int interface_count = 0;
  int immediate_count = 0;
  int stage = 0;
  auto vector_instruction_mapping = get_vector_instruction_mapping();
  auto it = param.vector_params().begin();
  bool has_vector_params = it != param.vector_params().end();
  std::string output_node = matrix_param.name();
  for (int stage = 0; stage < 5; stage++) {
    const auto opcode = has_vector_params ? it->opcode() : "nop";
    bool matched = vector_ops[stage].find(opcode) != vector_ops[stage].end();

    std::cerr << "stage: " << stage << " opcode: " << opcode
              << " matched: " << matched << std::endl;

    switch (stage) {
      case 0: {
        vector_instr.vOp0 = matched ? vector_instruction_mapping[opcode]
                                    : VectorInstructions::nop;
        break;
      }
      case 1: {
        vector_instr.vOp1 = matched ? vector_instruction_mapping[opcode]
                                    : VectorInstructions::nop;
        break;
      }
      case 2: {
        vector_instr.vOp2 = matched ? vector_instruction_mapping[opcode]
                                    : VectorInstructions::nop;
        break;
      }
      case 3: {
        vector_instr.vOp3 = matched ? vector_instruction_mapping[opcode]
                                    : VectorInstructions::nop;
        break;
      }
      case 4: {
        vector_instr.vOp4 = matched ? vector_instruction_mapping[opcode]
                                    : VectorInstructions::nop;
        break;
      }
    }

    if (matched) {
      if (it->has_other()) {
        const auto tensor_to_load =
            it->other().node() == output_node ? it->input() : it->other();
        const auto shape = tensor_to_load.shape();
        int size = 0;
        for (const int &dim : shape) size *= dim;

        if (size == 1) {
          // Currently only support two immediates
          if (immediate_count == 0) {
            vector_instr.vOp0Src1 = VectorInstructions::op0immediate0;
            // TODO:
            // vector_instr.immediate0 = ;
          } else if (immediate_count == 1) {
            vector_instr.vOp1Src1 = VectorInstructions::op1immediate0;
            // TODO:
            // vector_instr.immediate1 = ;
          } else {
            std::cerr << "Error: more than 2 immediate values\n";
            exit(1);
          }
          immediate_count++;
        } else {
          // Currently only support two address generators
          vector_instr.vOp0Src1 = VectorInstructions::readInterface;
          if (interface_count == 0) {
            set_addr_gen1(tensor_to_load, tiling, accelerator_memory_map,
                          accelerator_vector_param);
          } else if (interface_count == 1) {
            set_addr_gen2(tensor_to_load, tiling, accelerator_memory_map,
                          accelerator_vector_param);
          } else {
            std::cerr << "Error: more than 2 address generators\n";
            exit(1);
          }
          interface_count++;
        }
      }
      ++it;
      if (it != param.vector_params().end()) {
        output_node = it->name();
      }
    }
  }

  if (it != param.vector_params().end()) {
    std::cerr << "Error: unsupported vector fusion pattern" << std::endl;
    exit(1);
  }

  const auto output_tensor = param.output();
  const auto output_memory = output_tensor.memory();

  accelerator_memory_map["outputs"] = get_partition(output_memory.partition());
  accelerator_vector_param->VECTOR_OUTPUT_OFFSET = output_memory.offset();
  // TODO: double precision
  accelerator_vector_param->DP_OUTPUT = false;
  // TODO: Transformer qkv output permutation
  accelerator_vector_param->SPLIT_OUTPUT = false;

  for (int i = 0; i < 3; i++) {
    accelerator_vector_param->outputLoops[0][i] = tiling.loops[0][i];
  }
  accelerator_vector_param->outputXLoopIndex[0] = tiling.x_loop_index[0];
  accelerator_vector_param->outputYLoopIndex[0] = tiling.y_loop_index[0];
  accelerator_vector_param->outputWeightLoopIndex[0] =
      tiling.weight_loop_index[0];

  int outputLoopIndex = 0;
  for (int i = 0; i < 6; i++) {
    // ignore the loops not present in outputs (reduction, fx, fy)
    if (i == tiling.weight_loop_index[1] || i == tiling.x_loop_index[1] ||
        i == tiling.y_loop_index[1]) {
      accelerator_vector_param->outputLoops[1][outputLoopIndex] =
          tiling.loops[1][i];
      if (i == tiling.x_loop_index[1]) {
        accelerator_vector_param->outputXLoopIndex[1] = outputLoopIndex;
      }
      if (i == tiling.y_loop_index[1]) {
        accelerator_vector_param->outputYLoopIndex[1] = outputLoopIndex;
      }
      if (i == tiling.weight_loop_index[1]) {
        accelerator_vector_param->outputWeightLoopIndex[1] = outputLoopIndex;
      }
      outputLoopIndex++;
    }
  }

  vector_instr.vDest = VectorInstructions::vWriteOut;

  // total output count
  VectorInstructionConfig *vector_instruction_config =
      new VectorInstructionConfig;
  vector_instruction_config->inst[0] = vector_instr;
  vector_instruction_config->instCount[0] =
      tiling.loops[0][tiling.x_loop_index[0]] *
      tiling.loops[1][tiling.x_loop_index[1]] *
      tiling.loops[0][tiling.y_loop_index[0]] *
      tiling.loops[1][tiling.y_loop_index[1]] *
      tiling.loops[0][tiling.weight_loop_index[0]] *
      tiling.loops[1][tiling.weight_loop_index[1]];
  vector_instruction_config->instLen = 1;
  vector_instruction_config->instLoopCount = 1;

  mappedParams.push_back(matrixParams);
  mappedParams.push_back(accelerator_vector_param);
  mappedParams.push_back(vector_instruction_config);
  opMemoryMaps.push_back(accelerator_memory_map);
}
