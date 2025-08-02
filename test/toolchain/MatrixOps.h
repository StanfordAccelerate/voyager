#pragma once

#include "spdlog/spdlog.h"
#include "src/AccelTypes.h"
#include "src/Params.h"
#include "test/common/GoldModel.h"
#include "test/common/MemoryInterface.h"
#include "test/common/Network.h"
#include "test/common/Tiling.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"
#include "test/toolchain/ApproximationConstants.h"
#include "test/toolchain/Common.h"

void set_vector_fetch_1(const codegen::Tensor &tensor, const Tiling &tiling,
                        AcceleratorMemoryMap &accelerator_memory_map,
                        VectorParams *vector_params) {
  int nonzero_dims = 0;
  for (const int &dim : tensor.shape()) {
    if (dim != 1) nonzero_dims++;
  }

  const auto memory = tensor.memory();
  accelerator_memory_map["vector1"] = get_partition(memory.partition());
  vector_params->vector_fetch_1_offset = get_address(tensor);
  vector_params->vector_fetch_1_mode = true;
  vector_params->vector_fetch_1_broadcast = nonzero_dims == 1 ? 0b011 : 0b000;

  vector_params->vector_fetch_1_dtype =
      get_index_from_type_name<VU_INPUT_TYPES>(tensor.dtype());
  int fetch_width = OC_DIMENSION * get_type_width<VU_INPUT_TYPES>(
                                       vector_params->vector_fetch_1_dtype);
  vector_params->vector_fetch_1_burst_size = fetch_width / 8;
  vector_params->vector_fetch_1_num_beats = fetch_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_1_packing_factor =
      OC_DIMENSION / VECTOR_UNIT_WIDTH;

  // copy loop values and indices
  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_1_loops[0][i] = tiling.loops[0][i];
  }
  vector_params->vector_fetch_1_x_loop_idx[0] = tiling.x_loop_index[0];
  vector_params->vector_fetch_1_y_loop_idx[0] = tiling.y_loop_index[0];
  vector_params->vector_fetch_1_k_loop_idx[0] = tiling.weight_loop_index[0];

  int loop_index = 0;
  for (int i = 0; i < 6; i++) {
    // ignore the loops not present in outputs (reduction, fx, fy)
    if (i == tiling.weight_loop_index[1] || i == tiling.x_loop_index[1] ||
        i == tiling.y_loop_index[1]) {
      vector_params->vector_fetch_1_loops[1][loop_index] = tiling.loops[1][i];
      if (i == tiling.x_loop_index[1]) {
        vector_params->vector_fetch_1_x_loop_idx[1] = loop_index;
      }
      if (i == tiling.y_loop_index[1]) {
        vector_params->vector_fetch_1_y_loop_idx[1] = loop_index;
      }
      if (i == tiling.weight_loop_index[1]) {
        vector_params->vector_fetch_1_k_loop_idx[1] = loop_index;
      }
      loop_index++;
    }
  }

  DataTypes::bfloat16 scale = tensor.scale() != 0 ? tensor.scale() : 1.0;
  vector_params->vector_fetch_1_dq_scale = scale.bits_rep();
}

void set_vector_fetch_2(const codegen::Tensor &tensor, const Tiling &tiling,
                        AcceleratorMemoryMap &accelerator_memory_map,
                        VectorParams *vector_params) {
  int nonzero_dims = 0;
  for (const int &dim : tensor.shape()) {
    if (dim != 1) nonzero_dims++;
  }

  const auto memory = tensor.memory();
  accelerator_memory_map["vector2"] = get_partition(memory.partition());
  vector_params->vector_fetch_2_offset = get_address(tensor);
  vector_params->vector_fetch_2_mode = true;
  vector_params->vector_fetch_2_broadcast = nonzero_dims == 1 ? 0b011 : 0b000;

  vector_params->vector_fetch_2_dtype =
      get_index_from_type_name<VU_INPUT_TYPES>(tensor.dtype());
  int fetch_width = OC_DIMENSION * get_type_width<VU_INPUT_TYPES>(
                                       vector_params->vector_fetch_2_dtype);
  vector_params->vector_fetch_2_burst_size = fetch_width / 8;
  vector_params->vector_fetch_2_num_beats = fetch_width / OC_PORT_WIDTH;
  vector_params->vector_fetch_2_packing_factor =
      OC_DIMENSION / VECTOR_UNIT_WIDTH;

  // copy loop values and indices
  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_2_loops[0][i] = tiling.loops[0][i];
  }
  vector_params->vector_fetch_2_x_loop_idx[0] = tiling.x_loop_index[0];
  vector_params->vector_fetch_2_y_loop_idx[0] = tiling.y_loop_index[0];
  vector_params->vector_fetch_2_k_loop_idx[0] = tiling.weight_loop_index[0];

  int loop_index = 0;
  for (int i = 0; i < 6; i++) {
    // ignore the loops not present in outputs (reduction, fx, fy)
    if (i == tiling.weight_loop_index[1] || i == tiling.x_loop_index[1] ||
        i == tiling.y_loop_index[1]) {
      vector_params->vector_fetch_2_loops[1][loop_index] = tiling.loops[1][i];
      if (i == tiling.x_loop_index[1]) {
        vector_params->vector_fetch_2_x_loop_idx[1] = loop_index;
      }
      if (i == tiling.y_loop_index[1]) {
        vector_params->vector_fetch_2_y_loop_idx[1] = loop_index;
      }
      if (i == tiling.weight_loop_index[1]) {
        vector_params->vector_fetch_2_k_loop_idx[1] = loop_index;
      }
      loop_index++;
    }
  }

  DataTypes::bfloat16 scale = tensor.scale() != 0 ? tensor.scale() : 1.0;
  vector_params->vector_fetch_2_dq_scale = scale.bits_rep();
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

/**
 * \brief Determine whether we should use the direct path between the matrix
 * unit and the vector unit.
 *
 * Even if we have a double-buffered accumulation buffer, it may still be
 * profitable to use the direct path. This way, we save on the latency of having
 * to fully fill the accumulation buffer before even starting to drain it.
 *
 * However, the vector unit may not be able to keep up with the matrix unit. The
 * matrix unit produces `OC_DIMENSION` elements per cycle. If the inputs take
 * more than one cycle to fetch that many elements, the vector unit (and then
 * the matrix unit) will stall. The same happens if we can't write that many
 * elements to the output per cycle.
 *
 * This function determines whether it is profitable to use the direct path for
 * this particular instruction, given the accelerator's configuration. It should
 * only be called if we have a double-buffered accumulation buffer.
 */
static bool should_use_direct_path(const VectorParams *vector_params) {
  assert(DOUBLE_BUFFERED_ACCUM_BUFFER);

  // This is how much bandwidth we have available, in bits per cycle. It
  // happens that this value is the same for both inputs and the output.
  //
  // If we want to use the direct path without stalling, it had better be the
  // case that the available bandwidth exceeds the required bandwidth of
  // producing `OC_DIMENSION` elements per cycle.
  const size_t available_bandwidth = OC_PORT_WIDTH;

  // How much bandwidth we need for each of the ports, in bits per cycle. If the
  // address generation is inactive, we don't need any bandwidth. Otherwise, it
  // depends on the type we're fetching/writing.
  //
  // Remember, we need `OC_DIMENSION` elements per cycle on each (active) port
  // to keep up with the matrix unit.
  const size_t vector_fetch_1_bw =
      (vector_params->vector_fetch_1_mode == 0)
          ? 0
          : get_type_width<VU_INPUT_TYPES>(
                vector_params->vector_fetch_1_dtype) *
                OC_DIMENSION;
  const size_t vector_fetch_2_bw =
      (vector_params->vector_fetch_2_mode == 0)
          ? 0
          : get_type_width<VU_INPUT_TYPES>(
                vector_params->vector_fetch_2_dtype) *
                OC_DIMENSION;
  const size_t output_bw =
      (vector_params->output_mode == 0)
          ? 0
          : get_type_width<OUTPUT_DATATYPES>(vector_params->output_dtype) *
                OC_DIMENSION;

  bool should_use_direct_path = true;
  should_use_direct_path &= vector_fetch_1_bw <= available_bandwidth;
  should_use_direct_path &= vector_fetch_2_bw <= available_bandwidth;
  should_use_direct_path &= output_bw <= available_bandwidth;

  return should_use_direct_path;
}

void MapMatrixOperation(const Operation &operation,
                        std::deque<BaseParams *> &mappedParams,
                        std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  MatrixParams *matrix_params = new MatrixParams;
  AcceleratorMemoryMap accelerator_memory_map;
  VectorInstructionConfig *vector_instruction_config =
      new VectorInstructionConfig;

  const auto param = operation.param;
  const auto op_list = get_op_list(param);
  const auto matrix_op = op_list[0];

  const auto input = matrix_op.kwargs().at("input").tensor();

  bool is_matmul = matrix_op.target().find("matmul") != std::string::npos;
  std::string weight_key = is_matmul ? "other" : "weight";
  const auto weight = matrix_op.kwargs().at(weight_key).tensor();

  Tiling tiling;

  matrix_params->is_fc = is_fc(matrix_op);

  if (matrix_params->is_fc) {
    bool is_mx_op = matrix_op.target().find("mx") != std::string::npos;

    int K = is_matmul ? weight.shape(1) : weight.shape(0);
    int C = is_matmul ? weight.shape(0) : weight.shape(1);
    int C1 = is_mx_op ? matrix_op.kwargs().at("block_size").int_value() : 1;
    int C2 = C / C1;

    auto k_loops = split_loops({K}, pow(2, 16) - 1);
    k_loops = adjust_loop_indices(k_loops, OC_DIMENSION);
    pad_shape_to_ndim(k_loops, 2);

    tiling = {
        .loops = {{1, 1, k_loops[0], C2, 1, 1}, {C1, k_loops[1], 1, 1, 1, 1}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = {4, 2},
        .weight_reuse_index = {4, 5},
        .stride = 1,
        .resnet_replication = false,
        .generic_replication = false,
    };
  } else {
    tiling = get_tiling(operation);
  }

  std::ostringstream oss;
  oss << tiling;
  spdlog::info("Using tiling: \n{}\n", oss.str());

  const auto input_memory = input.memory();
  accelerator_memory_map["inputs"] = get_partition(input_memory.partition());
  matrix_params->INPUT_OFFSET = get_address(input);
  matrix_params->input_dtype =
      get_index_from_type_name<INPUT_DATATYPE>(input.dtype());
  matrix_params->use_input_codebook = matrix_op.kwargs().contains("input_code");

  if (matrix_params->use_input_codebook) {
    const auto code = matrix_op.kwargs().at("input_code").tensor();
    const auto size = get_size(code);

    float *input_code = read_constant_param(code);
    for (int i = 0; i < size; i++) {
      SA_INPUT_TYPE value = input_code[i];
      matrix_params->input_code[i] = value.bits_rep();
    }

    delete[] input_code;
  }

  const auto weight_memory = weight.memory();
  accelerator_memory_map["weights"] = get_partition(weight_memory.partition());
  matrix_params->WEIGHT_OFFSET = get_address(weight);
  matrix_params->weight_dtype =
      get_index_from_type_name<WEIGHT_DATATYPE>(weight.dtype());
  matrix_params->use_weight_codebook =
      matrix_op.kwargs().contains("weight_code");

  if (matrix_params->use_weight_codebook) {
    const auto code = matrix_op.kwargs().at("weight_code").tensor();
    const auto size = get_size(code);

    float *weight_code = read_constant_param(code);
    for (int i = 0; i < size; i++) {
      SA_WEIGHT_TYPE value = weight_code[i];
      matrix_params->weight_code[i] = value.bits_rep();
    }

    delete[] weight_code;
  }

  matrix_params->is_mx_op = matrix_op.target().find("mx") != std::string::npos;

  if (matrix_params->is_mx_op) {
    const int block_size = matrix_op.kwargs().at("block_size").int_value();
    assert(block_size == std::max(IC_DIMENSION, OC_DIMENSION));

    const auto input_scale = matrix_op.kwargs().at("input_scale").tensor();
    matrix_params->INPUT_SCALE_OFFSET = get_address(input_scale);

    const auto weight_scale = matrix_op.kwargs().at("weight_scale").tensor();
    matrix_params->WEIGHT_SCALE_OFFSET = get_address(weight_scale);
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
    matrix_params->fyIndex[i] = tiling.fy_index[i];
  }
  matrix_params->fxIndex = tiling.fx_index;

  // set outer loop values
  for (int j = 0; j < 5; j++) {
    matrix_params->weightAddressGenLoops[0][j] = tiling.loops[0][j];
  }
  matrix_params->weightAddressGenWeightLoopIndex[0] =
      tiling.weight_loop_index[0];
  matrix_params->weightAddressGenReductionLoopIndex[0] =
      tiling.reduction_loop_index[0];
  matrix_params->weightAddressGenFyIndex[0] = tiling.fy_index[0];

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
        tiling.loops[1][tiling.fy_index[1]];
    matrix_params->weightAddressGenFyIndex[1] = 2;

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

  } else {  // if not transpose, then we have freedom to pick any loop order
    // K1 loop
    matrix_params->weightAddressGenLoops[1][4] =
        tiling.loops[1][tiling.weight_loop_index[1]];
    matrix_params->weightAddressGenWeightLoopIndex[1] = 4;

    // C0 loop
    if (tiling.resnet_replication) {
      matrix_params->weightAddressGenLoops[1][3] = 3;
    } else if (tiling.generic_replication) {
      matrix_params->weightAddressGenLoops[1][3] = tiling.num_channels;
    } else {
      matrix_params->weightAddressGenLoops[1][3] = IC_DIMENSION;
    }
    matrix_params->weightAddressGenReductionLoopIndex[2] = 3;

    // C1 loop
    matrix_params->weightAddressGenLoops[1][2] =
        tiling.loops[1][tiling.reduction_loop_index[1]];
    matrix_params->weightAddressGenReductionLoopIndex[1] = 2;

    // FX loop
    matrix_params->weightAddressGenLoops[1][1] =
        tiling.loops[1][tiling.fx_index];
    if (tiling.resnet_replication) {
      matrix_params->weightAddressGenLoops[1][1] = 7;
    } else if (tiling.generic_replication) {
      matrix_params->weightAddressGenLoops[1][1] *= tiling.fx_unrolling;
    }
    matrix_params->weightAddressGenFxIndex = 1;

    // FY loop
    matrix_params->weightAddressGenLoops[1][0] =
        tiling.loops[1][tiling.fy_index[1]];
    matrix_params->weightAddressGenFyIndex[1] = 0;
  }

  matrix_params->stride = tiling.stride;
  matrix_params->padding = tiling.padding;
  matrix_params->is_resnet_replication = tiling.resnet_replication;
  matrix_params->is_generic_replication = tiling.generic_replication;
  matrix_params->num_channels = tiling.num_channels;
  matrix_params->fx_unrolling_lg2 = std::log2(tiling.fx_unrolling);

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

  int c_bound = tiling.resnet_replication
                    ? 1
                    : tiling.loops[1][tiling.reduction_loop_index[1]];
  int input_effective_fw;
  int input_pf =
      get_packing_factor<IC_DIMENSION, IC_PORT_WIDTH, INPUT_DATATYPE>(
          matrix_params->input_dtype, c_bound, input_effective_fw);

  matrix_params->input_burst_size = input_effective_fw / 8;
  matrix_params->input_num_beats = input_effective_fw / IC_PORT_WIDTH;
  matrix_params->input_packing_factor_power = std::log2(input_pf);

  int k_bound = matrix_params->has_weight_transpose
                    ? 1
                    : tiling.loops[1][tiling.weight_loop_index[1]];
  int weight_effective_fw;
  int weight_pf =
      get_packing_factor<OC_DIMENSION, OC_PORT_WIDTH, WEIGHT_DATATYPE>(
          matrix_params->weight_dtype, k_bound, weight_effective_fw);

  matrix_params->weight_burst_size = weight_effective_fw / 8;
  matrix_params->weight_num_beats = weight_effective_fw / OC_PORT_WIDTH;
  matrix_params->weight_packing_factor_power = std::log2(weight_pf);

  // bias
  if (matrix_op.kwargs().contains("bias")) {
    const auto bias = matrix_op.kwargs().at("bias").tensor();
    const auto bias_memory = bias.memory();
    matrix_params->has_bias = true;
    matrix_params->BIAS_OFFSET = get_address(bias);
    accelerator_memory_map["bias"] = get_partition(bias_memory.partition());
  }

  // vector instructions
  VectorParams *vector_params = new VectorParams;
#if DOUBLE_BUFFERED_ACCUM_BUFFER
  vector_params->vector_fetch_0_mode = 3;  // read from accumulation buffer
#else
  vector_params->vector_fetch_0_mode = 0;  // read from matrix unit
#endif
  // Set outer loops
  for (int i = 0; i < 3; i++) {
    vector_params->vector_fetch_0_loops[0][i] = tiling.loops[0][i];
  }
  vector_params->vector_fetch_0_y_loop_idx[0] = tiling.y_loop_index[0];
  vector_params->vector_fetch_0_x_loop_idx[0] = tiling.x_loop_index[0];
  vector_params->vector_fetch_0_k_loop_idx[0] = tiling.weight_loop_index[0];

  // Rescale tiling for vector instructions
  if (matrix_params->is_fc) {
    tiling.loops[1][1] /= OC_DIMENSION;
  }

  if (!matrix_params->is_fc) {
    vector_params->vector_fetch_0_mode = 3;  // read from accumulation buffer
    // Set outer loops
    for (int i = 0; i < 3; i++) {
      vector_params->vector_fetch_0_loops[0][i] = tiling.loops[0][i];
    }
    vector_params->vector_fetch_0_y_loop_idx[0] = tiling.y_loop_index[0];
    vector_params->vector_fetch_0_x_loop_idx[0] = tiling.x_loop_index[0];
    vector_params->vector_fetch_0_k_loop_idx[0] = tiling.weight_loop_index[0];

    // Set inner loops
    int addressGen0LoopIndex = 0;
    for (int i = 0; i < 6; i++) {
      // ignore the loops not present in outputs (reduction, fx, fy)
      if (i == tiling.weight_loop_index[1] || i == tiling.x_loop_index[1] ||
          i == tiling.y_loop_index[1]) {
        vector_params->vector_fetch_0_loops[1][addressGen0LoopIndex] =
            tiling.loops[1][i];
        if (i == tiling.y_loop_index[1]) {
          vector_params->vector_fetch_0_y_loop_idx[1] = addressGen0LoopIndex;
        }
        if (i == tiling.x_loop_index[1]) {
          vector_params->vector_fetch_0_x_loop_idx[1] = addressGen0LoopIndex;
        }
        if (i == tiling.weight_loop_index[1]) {
          vector_params->vector_fetch_0_k_loop_idx[1] = addressGen0LoopIndex;
        }
        addressGen0LoopIndex++;
      }
    }
  }

  const auto output = get_op_outputs(param).back();
  const auto output_memory = output.memory();
  accelerator_memory_map["outputs"] = get_partition(output_memory.partition());
  vector_params->VECTOR_OUTPUT_OFFSET = get_address(output);

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

  const int packing_factor = OC_DIMENSION / VECTOR_UNIT_WIDTH;

  VectorInstructions inst;
  inst.op_type = VectorInstructions::vector;
  inst.inst_count = tiling.loops[0][tiling.x_loop_index[0]] *
                    tiling.loops[1][tiling.x_loop_index[1]] *
                    tiling.loops[0][tiling.y_loop_index[0]] *
                    tiling.loops[1][tiling.y_loop_index[1]] *
                    tiling.loops[0][tiling.weight_loop_index[0]] *
                    tiling.loops[1][tiling.weight_loop_index[1]] *
                    packing_factor;

  if (matrix_params->is_fc) {
    inst.vector_op0_src0 = VectorInstructions::from_matrix_vector_unit;
  } else {
#if DOUBLE_BUFFERED_ACCUM_BUFFER
    inst.vector_op0_src0 = VectorInstructions::from_accumulation_buffer;
    matrix_params->write_output_to_accum_buffer = true;
#else
    inst.vector_op0_src0 = VectorInstructions::from_matrix_unit;
#endif
  }

  inst.vdest = VectorInstructions::to_output;

  auto inst_map = get_vector_instruction_mapping();

  int stage = 0;

  for (int i = 1; i < op_list.size(); i++) {
    const auto op = op_list[i];
    const std::string opcode = op.target();

    // Grab kwargs that are relevant for some activation functions
    std::map<std::string, VECTOR_DATATYPE> activation_kwargs;
    if (unary_ops_with_kwargs.find(op.target()) !=
        unary_ops_with_kwargs.end()) {
      for (const auto &[key, value] : op.kwargs()) {
        if (key != "input") {
          activation_kwargs[key] = VECTOR_DATATYPE(value.float_value());
        }
      }
    }

    // Dequantization doesn't take a stage in the pipeline
    if (opcode == "dequantize") {
      inst.vdequantize = true;

      const auto other = op.kwargs().at("scale").tensor();

      assert(get_size(other) == 1);

      float *array = read_constant_param(other);
      VECTOR_DATATYPE immediate = array[0];
      inst.vector_dq_scale = immediate.bits_rep();

      delete[] array;

      continue;
    }

    // Find the stage of the operation
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

    if (opcode == "quantize_mx") {
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
      vector_params->SCALE_OFFSET = get_address(param.outputs().tensors(0));

      if (op.kwargs().contains("quant_code")) {
        const auto code = op.kwargs().at("quant_code").tensor();
        const int size = get_size(code);

        float *array = read_constant_param(code);

        for (int i = 0; i < size; i++) {
          vector_params->output_code[i] = array[i] * 2;
        }

        delete[] array;

        vector_params->use_output_codebook = true;
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
        float *array = read_constant_param(other.tensor());
        set_immediate(array[0], stage, opcode, inst);
        delete[] array;
      } else {
        auto self = op.kwargs().at("input").tensor();
        auto tensor = other.tensor();
        auto tensor_to_load = tensor.has_memory() ? tensor : self;

        if (stage == 0) {
          inst.vector_op0_src1 = VectorInstructions::from_vector_fetch_1;
          set_vector_fetch_1(tensor_to_load, tiling, accelerator_memory_map,
                             vector_params);
        } else if (stage == 2) {
          inst.vector_op2_src1 = VectorInstructions::from_vector_fetch_2;
          set_vector_fetch_2(tensor_to_load, tiling, accelerator_memory_map,
                             vector_params);
        } else {
          assert(inst.vector_op2_src1 !=
                 VectorInstructions::from_vector_fetch_2);
          inst.vector_op3_src1 = VectorInstructions::from_vector_fetch_2;
          set_vector_fetch_2(tensor_to_load, tiling, accelerator_memory_map,
                             vector_params);
        }
      }
    }

    stage++;
  }

#if DOUBLE_BUFFERED_ACCUM_BUFFER
  if (should_use_direct_path(vector_params)) {
    inst.vector_op0_src0 = VectorInstructions::from_matrix_unit;
    vector_params->vector_fetch_0_mode = 0;
    matrix_params->write_output_to_accum_buffer = false;
  }
#endif

  // total output count
  vector_instruction_config->inst[0] = inst;
  vector_instruction_config->instLen = 1;
  vector_instruction_config->instLoopCount = 1;

  mappedParams.push_back(matrix_params);
  mappedParams.push_back(vector_params);
  mappedParams.push_back(vector_instruction_config);
  opMemoryMaps.push_back(accelerator_memory_map);
}
