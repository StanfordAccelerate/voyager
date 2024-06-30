#pragma once

#include "src/AccelTypes.h"
#include "src/Params.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"

void MapVectorOperations(const codegen::AcceleratorParam &param,
                         std::deque<BaseParams *> &mappedParams,
                         std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  const auto vector_input = param.vector_params(0).input();
  const auto input_memory = vector_input.memory();
  int input_dim = 1;
  for (int i = 0; i < vector_input.shape_size() - 1; i++) {
    input_dim *= vector_input.shape(i);
  }
  int output_dim = vector_input.shape(vector_input.shape_size() - 1);

  VectorParams *accelerator_vector_param = new VectorParams;
  accelerator_vector_param->VECTOR_OFFSET = input_memory.offset();
  accelerator_vector_param->addressGen0Mode = true;
  accelerator_vector_param->addressGen0Broadcast = false;
  for (int i = 0; i < 3; i++) {
    accelerator_vector_param->addressGen0Loop[0][i] = 1;
  }
  accelerator_vector_param->addressGen0Loop[1][0] = 1;
  accelerator_vector_param->addressGen0Loop[1][1] = input_dim;
  accelerator_vector_param->addressGen0Loop[1][2] = output_dim / (OC_DIMENSION);
  accelerator_vector_param->DP_VEC0 = false;

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
  std::string output_node;
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

  mappedParams.push_back(accelerator_vector_param);
  mappedParams.push_back(vector_instruction_config);
  opMemoryMaps.push_back(accelerator_memory_map);
}