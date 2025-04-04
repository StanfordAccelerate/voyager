#pragma once

#include "spdlog/spdlog.h"
#include "src/AccelTypes.h"
#include "src/Params.h"
#include "test/common/GoldModel.h"
#include "test/common/Network.h"
#include "test/common/Tiling.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"
#include "test/toolchain/Common.h"

void set_addr_gen1(const codegen::Tensor &tensor, const Tiling &tiling,
                   AcceleratorMemoryMap &accelerator_memory_map,
                   VectorParams *vector_params) {
  int nonzero_dims = 0;
  for (const int &dim : tensor.shape()) {
    if (dim != 1) nonzero_dims++;
  }

  const auto memory = tensor.memory();
  accelerator_memory_map["vector1"] = get_partition(memory.partition());
  vector_params->ADDRESS_GEN1_OFFSET = memory.address();
  vector_params->addr_gen1_mode = true;
  vector_params->addr_gen1_broadcast = nonzero_dims == 1 ? 0b011 : 0b000;
  vector_params->addr_gen1_dtype =
      get_index_from_type_name<VECTOR_INPUT_DATATYPES>(tensor.dtype());

  // copy loop values and indices
  for (int i = 0; i < 3; i++) {
    vector_params->addr_gen1_loops[0][i] = tiling.loops[0][i];
  }
  vector_params->addr_gen1_x_loop_idx[0] = tiling.x_loop_index[0];
  vector_params->addr_gen1_y_loop_idx[0] = tiling.y_loop_index[0];
  vector_params->addr_gen1_k_loop_idx[0] = tiling.weight_loop_index[0];

  int loop_index = 0;
  for (int i = 0; i < 6; i++) {
    // ignore the loops not present in outputs (reduction, fx, fy)
    if (i == tiling.weight_loop_index[1] || i == tiling.x_loop_index[1] ||
        i == tiling.y_loop_index[1]) {
      vector_params->addr_gen1_loops[1][loop_index] = tiling.loops[1][i];
      if (i == tiling.x_loop_index[1]) {
        vector_params->addr_gen1_x_loop_idx[1] = loop_index;
      }
      if (i == tiling.y_loop_index[1]) {
        vector_params->addr_gen1_y_loop_idx[1] = loop_index;
      }
      if (i == tiling.weight_loop_index[1]) {
        vector_params->addr_gen1_k_loop_idx[1] = loop_index;
      }
      loop_index++;
    }
  }

  DataTypes::bfloat16 scale = tensor.scale() != 0 ? tensor.scale() : 1.0;
  vector_params->addr_gen1_dq_scale = scale.bits_rep();
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
  vector_params->ADDRESS_GEN2_OFFSET = memory.address();
  vector_params->addr_gen2_mode = true;
  vector_params->addr_gen2_broadcast = nonzero_dims == 1 ? 0b011 : 0b000;

  vector_params->addr_gen2_dtype =
      get_index_from_type_name<VECTOR_INPUT_DATATYPES>(tensor.dtype());

  // copy loop values and indices
  for (int i = 0; i < 3; i++) {
    vector_params->addr_gen2_loops[0][i] = tiling.loops[0][i];
  }
  vector_params->addr_gen2_x_loop_idx[0] = tiling.x_loop_index[0];
  vector_params->addr_gen2_y_loop_idx[0] = tiling.y_loop_index[0];
  vector_params->addr_gen2_k_loop_idx[0] = tiling.weight_loop_index[0];

  int loop_index = 0;
  for (int i = 0; i < 6; i++) {
    // ignore the loops not present in outputs (reduction, fx, fy)
    if (i == tiling.weight_loop_index[1] || i == tiling.x_loop_index[1] ||
        i == tiling.y_loop_index[1]) {
      vector_params->addr_gen2_loops[1][loop_index] = tiling.loops[1][i];
      if (i == tiling.x_loop_index[1]) {
        vector_params->addr_gen2_x_loop_idx[1] = loop_index;
      }
      if (i == tiling.y_loop_index[1]) {
        vector_params->addr_gen2_y_loop_idx[1] = loop_index;
      }
      if (i == tiling.weight_loop_index[1]) {
        vector_params->addr_gen2_k_loop_idx[1] = loop_index;
      }
      loop_index++;
    }
  }

  DataTypes::bfloat16 scale = tensor.scale() != 0 ? tensor.scale() : 1.0;
  vector_params->addr_gen2_dq_scale = scale.bits_rep();
}

void set_immediate(const float scalar, const int stage,
                   const std::string opcode, VectorInstructions &inst) {
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

void MapMatrixOperation(const Operation &operation,
                        std::deque<BaseParams *> &mappedParams,
                        std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  MatrixParams *matrix_params = new MatrixParams;
  AcceleratorMemoryMap accelerator_memory_map;

  const auto param = operation.param;
  const auto op_list = get_op_list(param);
  const auto matrix_op = op_list[0];

  Tiling tiling = get_tiling(operation);

  int X = tiling.loops[0][tiling.x_loop_index[0]] *
          tiling.loops[1][tiling.x_loop_index[1]];
  int Y = tiling.loops[0][tiling.y_loop_index[0]] *
          tiling.loops[1][tiling.y_loop_index[1]];
  int C = tiling.loops[0][tiling.reduction_loop_index[0]] *
          tiling.loops[1][tiling.reduction_loop_index[1]] * IC_DIMENSION;
  int K = tiling.loops[0][tiling.weight_loop_index[0]] *
          tiling.loops[1][tiling.weight_loop_index[1]] * OC_DIMENSION;
  int FX = tiling.loops[1][tiling.fx_index];
  int FY = tiling.loops[1][tiling.fy_index];
  int STRIDE = tiling.stride;

  std::ostringstream oss;
  oss << tiling;
  spdlog::info("Using tiling: \n{}\n", oss.str());

  const auto input = matrix_op.kwargs().at("input").tensor();

  bool is_matmul = matrix_op.target().find("matmul") != std::string::npos;
  std::string weight_key = is_matmul ? "other" : "weight";
  const auto weight = matrix_op.kwargs().at(weight_key).tensor();

  const auto input_memory = input.memory();
  accelerator_memory_map["inputs"] = get_partition(input_memory.partition());
  matrix_params->INPUT_OFFSET = input_memory.address();

  const auto weight_memory = weight.memory();
  accelerator_memory_map["weights"] = get_partition(weight_memory.partition());
  matrix_params->WEIGHT_OFFSET = weight_memory.address();

  matrix_params->is_mx_op = matrix_op.target().find("mx") != std::string::npos;

  if (matrix_params->is_mx_op) {
    const int block_size = matrix_op.kwargs().at("block_size").int_value();
    assert(block_size == std::max(IC_DIMENSION, OC_DIMENSION));

    const auto input_scale = matrix_op.kwargs().at("input_scale").tensor();
    matrix_params->INPUT_SCALE_OFFSET = input_scale.memory().address();

    const auto weight_scale = matrix_op.kwargs().at("weight_scale").tensor();
    matrix_params->WEIGHT_SCALE_OFFSET = weight_scale.memory().address();
  }

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
  matrix_params->weightAddressGenReductionLoopIndex[0] =
      tiling.reduction_loop_index[0];

  // if OX and OY loops are the innermost L2 loops, they are irrelevant for
  // weight address generation
  if (tiling.loops[0][tiling.reduction_loop_index[0]] == 1) {
    if (tiling.weight_loop_index[0] < tiling.x_loop_index[0]) {
      matrix_params->weightAddressGenLoops[0][tiling.x_loop_index[0]] = 1;
    }
    if (tiling.weight_loop_index[0] < tiling.y_loop_index[0]) {
      matrix_params->weightAddressGenLoops[0][tiling.y_loop_index[0]] = 1;
    }
  }

  // set outer loop values
  matrix_params->has_weight_transpose =
      weight.reshape().target() == "transpose";
  if (matrix_params->has_weight_transpose) {
    // for transpose, we need to enforce that the innermost loop is the unrolled
    // reduction loop we can just use the following loop nest: C1, K, FY, FX, C0

    // C0 loop
    matrix_params->weightAddressGenLoops[1][4] = OC_DIMENSION;
    matrix_params->weightAddressGenReductionLoopIndex[2] = 4;

    // FX loop
    matrix_params->weightAddressGenLoops[1][3] =
        tiling.loops[1][tiling.fx_index];
    matrix_params->weightAddressGenFxIndex = 3;

    // FY loop
    matrix_params->weightAddressGenLoops[1][2] =
        tiling.loops[1][tiling.fy_index];
    matrix_params->weightAddressGenFyIndex = 2;

    // K loop
    matrix_params->weightAddressGenLoops[1][1] =
        tiling.loops[1][tiling.weight_loop_index[1]];
    matrix_params->weightAddressGenWeightLoopIndex[1] = 1;

    // C1 loop
    matrix_params->weightAddressGenLoops[1][0] =
        tiling.loops[1][tiling.reduction_loop_index[1]];
    if (OC_DIMENSION > IC_DIMENSION) {
      // we can reduce the number of iterations, since we have already fetched
      // the values
      if (tiling.loops[0][tiling.reduction_loop_index[0]] >=
          (OC_DIMENSION / IC_DIMENSION)) {
        matrix_params
            ->weightAddressGenLoops[0][tiling.reduction_loop_index[0]] =
            tiling.loops[0][tiling.reduction_loop_index[0]] /
            (OC_DIMENSION / IC_DIMENSION);
      } else {
        matrix_params
            ->weightAddressGenLoops[1][tiling.reduction_loop_index[1]] =
            tiling.loops[1][tiling.reduction_loop_index[1]] /
            (OC_DIMENSION / IC_DIMENSION);
      }
    }
    matrix_params->weightAddressGenReductionLoopIndex[1] = 0;
  } else {
    // if not transpose, then we have freedom to pick any loop order

    // K1 loop
    matrix_params->weightAddressGenLoops[1][4] =
        tiling.loops[1][tiling.weight_loop_index[1]];
    matrix_params->weightAddressGenWeightLoopIndex[1] = 4;

    // FY loop
    matrix_params->weightAddressGenLoops[1][3] =
        tiling.loops[1][tiling.fy_index];
    matrix_params->weightAddressGenFyIndex = 3;

    // FX loop
    matrix_params->weightAddressGenLoops[1][2] =
        tiling.loops[1][tiling.fx_index];
    if (tiling.replication) {
      matrix_params->weightAddressGenLoops[1][2] = 7;
    }
    matrix_params->weightAddressGenFxIndex = 2;

    // C0 loop
    if (tiling.replication) {
      matrix_params->weightAddressGenLoops[1][1] = 3;
    } else {
      matrix_params->weightAddressGenLoops[1][1] = IC_DIMENSION;
    }
    matrix_params->weightAddressGenReductionLoopIndex[2] = 1;

    // C1 loop
    matrix_params->weightAddressGenLoops[1][0] =
        tiling.loops[1][tiling.reduction_loop_index[1]];
    matrix_params->weightAddressGenReductionLoopIndex[1] = 0;
  }

  matrix_params->STRIDE = tiling.stride;
  matrix_params->is_replication = tiling.replication;

  // Permute input for transformer attention outputs
  if (input.has_reshape()) {
    const auto reshape_op = input.reshape();
    const auto reshape_kwargs = reshape_op.kwargs();

    if (reshape_op.target() == "permute") {
      const auto int_list = reshape_kwargs.at("dims").int_list().values();
      std::vector<int> dims(int_list.begin(), int_list.end());
      const int ndim = input.shape_size();

      if (is_transpose(dims)) {
        matrix_params->has_input_transpose = true;
      } else if (dims[ndim - 1] == ndim - 1) {
        matrix_params->has_attn_output_permute = true;
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
        matrix_params->has_input_transpose = true;
      } else if (dim1 != input.shape_size() - 1) {
        matrix_params->has_attn_output_permute = true;
      } else {
        throw std::invalid_argument("Unsupported transpose operation!");
      }
    }

    if (matrix_params->has_attn_output_permute) {
      const auto input_shape = input.shape();
      double result = std::log2(input_shape[input_shape.size() - 1]);
      if (std::fmod(result, 1.0) != 0.0) {
        throw std::runtime_error("Result is not an integer!");
      }
      matrix_params->head_size_power_of_two = result;
    }
  }

  // bias
  if (matrix_op.kwargs().contains("bias")) {
    const auto bias = matrix_op.kwargs().at("bias").tensor();
    const auto bias_memory = bias.memory();
    matrix_params->has_bias = true;
    matrix_params->BIAS_OFFSET = bias_memory.address();
    accelerator_memory_map["bias"] = get_partition(bias_memory.partition());
  }

  // vector instructions
  VectorParams *vector_params = new VectorParams;
  vector_params->addr_gen0_mode = 0;  // use matrix unit outputs

  codegen::Tensor output;
  if (param.has_output()) {
    output = param.output();
  } else {
    assert(op_list.back().target() == "quantize_mx");
    output = param.outputs().tensors(1);
  }

  const auto output_memory = output.memory();
  accelerator_memory_map["outputs"] = get_partition(output_memory.partition());
  vector_params->VECTOR_OUTPUT_OFFSET = output_memory.address();

  // Set outer loops
  for (int i = 0; i < 3; i++) {
    vector_params->output_loops[0][i] = tiling.loops[0][i];
  }
  vector_params->output_y_loop_idx[0] = tiling.y_loop_index[0];
  vector_params->output_x_loop_idx[0] = tiling.x_loop_index[0];
  vector_params->output_k_loop_idx[0] = tiling.weight_loop_index[0];

  // Set inner loops
  int outputLoopIndex = 0;
  for (int i = 0; i < 6; i++) {
    // ignore the loops not present in outputs (reduction, fx, fy)
    if (i == tiling.weight_loop_index[1] || i == tiling.x_loop_index[1] ||
        i == tiling.y_loop_index[1]) {
      vector_params->output_loops[1][outputLoopIndex] = tiling.loops[1][i];
      if (i == tiling.y_loop_index[1]) {
        vector_params->output_y_loop_idx[1] = outputLoopIndex;
      }
      if (i == tiling.x_loop_index[1]) {
        vector_params->output_x_loop_idx[1] = outputLoopIndex;
      }
      if (i == tiling.weight_loop_index[1]) {
        vector_params->output_k_loop_idx[1] = outputLoopIndex;
      }
      outputLoopIndex++;
    }
  }

  vector_params->output_dtype =
      get_index_from_type_name<OUTPUT_DATATYPES>(output.dtype());

  // Transformer head permutation
  if (output.has_reshape()) {
    vector_params->has_attn_head_permute = true;
    const auto permuted_shape =
        output.reshape().kwargs().at("output_shape").int_list().values();
    double result = std::log2(permuted_shape[permuted_shape.size() - 1]);
    if (std::fmod(result, 1.0) != 0.0) {
      throw std::runtime_error("Result is not an integer!");
    }
    vector_params->head_size_power_of_two = result;
  }

  VectorInstructions inst;
  memset(&inst, 0, sizeof(inst));
  inst.op_type = VectorInstructions::vector;
  inst.vector_op0_src0 = VectorInstructions::from_matrix_unit;
  inst.vdest = VectorInstructions::to_output;

  auto inst_map = get_vector_instruction_mapping();

  int curr_stage = 0;

  for (int i = 1; i < op_list.size(); i++) {
    const auto op = op_list[i];
    const std::string opcode = op.target();

    if (opcode == "dequantize") {
      const auto other = op.kwargs().at("scale").tensor();

      assert(get_size(other) == 1);

      VECTOR_DATATYPE immediate = read_constant_param(other);
      inst.vdequantize = true;
      inst.vector_dq_scale = immediate.bits_rep();
    } else {
      if (curr_stage == vector_unit_stages.size()) {
        // we have already processed all the stages
        assert(i == op_list.size() - 1);
        break;
      }

      for (int stage = curr_stage; stage < vector_unit_stages.size(); stage++) {
        if (vector_unit_stages[stage].find(opcode) ==
            vector_unit_stages[stage].end()) {
          continue;
        }

        // Only the last stage has a true divider
        if (opcode == "div" && stage != 3) {
          const auto other = op.kwargs().at("other").tensor();
          if (get_size(other) > 1) {
            continue;
          }
        }

        spdlog::debug("stage {} target: {}\n", stage, opcode);

        unsigned int vop = inst_map[opcode];
        if (stage == 0) {
          inst.vector_op0 = opcode == "div" ? VectorInstructions::vmult : vop;
        } else if (stage == 1) {
          inst.vector_op1 = vop;
        } else if (stage == 2) {
          inst.vector_op2 = opcode == "div" ? VectorInstructions::vmult : vop;
        } else if (stage == 3) {
          inst.vector_op3 = vop;
        }

        if (opcode == "vmap") {
          const auto other = op.kwargs().at("code").tensor();
          inst.VMAP_OFFSET = other.memory().address();
        } else if (opcode == "quantize_mx") {
          float quant_max = op.kwargs().at("quant_max").float_value();
          bool force_scale_power_of_two =
              op.kwargs().at("force_scale_power_of_two").int_value();

          if (force_scale_power_of_two) {
            inst.immediate2 = floor(log2(quant_max));
          } else {
            VECTOR_DATATYPE scale = quant_max;
            inst.immediate2 = scale.bits_rep();
          }

          vector_params->quantize_output_mx = true;
          vector_params->SCALE_OFFSET =
              param.outputs().tensors(0).memory().address();
        } else if (op.kwargs().contains("other") || opcode == "quantize") {
          std::string other_key = opcode == "quantize" ? "scale" : "other";
          const auto other = op.kwargs().at(other_key);

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

              if (stage == 0) {
                inst.vector_op0_src1 = VectorInstructions::from_vector_fetch_1;
                set_addr_gen1(tensor_to_load, tiling, accelerator_memory_map,
                              vector_params);
              } else if (stage == 2) {
                inst.vector_op2_src1 = VectorInstructions::from_vector_fetch_2;
                set_addr_gen2(tensor_to_load, tiling, accelerator_memory_map,
                              vector_params);
              } else if (inst.vector_op2_src1 !=
                         VectorInstructions::from_vector_fetch_2) {
                inst.vector_op3_src1 = VectorInstructions::from_vector_fetch_2;
                set_addr_gen2(tensor_to_load, tiling, accelerator_memory_map,
                              vector_params);
              } else {
                throw std::invalid_argument(
                    "Unsupported number of operands for vector operations!");
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
