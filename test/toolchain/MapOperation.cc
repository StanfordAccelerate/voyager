#include "spdlog/spdlog.h"
#include "src/AccelTypes.h"
#include "src/Params.h"
#include "test/common/Utils.h"
#include "test/compiler/proto/param.pb.h"
#include "test/toolchain/LayerNorm.h"
#include "test/toolchain/MatrixOps.h"
#include "test/toolchain/MatrixVectorMultiply.h"
#include "test/toolchain/Microscaling.h"
#include "test/toolchain/Pooling.h"
#include "test/toolchain/Reduction.h"
#include "test/toolchain/Softmax.h"
#include "test/toolchain/VectorOps.h"
#if SUPPORT_SPMM
#include "test/toolchain/SpMM.h"
#endif

void map_operation(const Operation& operation,
                   std::deque<BaseParams*>& mapped_params) {
  const auto param = operation.param;
  const auto op_list = get_op_list(param);
  const auto first_op = op_list[0];

  if (is_gemm_op(first_op.target())) {
    auto input = first_op.kwargs().at("input").tensor();
    if (is_fc_layer(first_op) && input.dtype() == "bfloat16") {
      map_matrix_vector_multiply(param, mapped_params);
    } else {
      map_matrix_operation(operation, mapped_params);
    }
  } else if (first_op.target() == "layer_norm") {
    map_layer_norm(param, mapped_params);
  } else if (first_op.target() == "softmax") {
    map_softmax(param, mapped_params);
  } else if (first_op.target() == "max_pool2d" ||
             first_op.target() == "adaptive_avg_pool2d") {
    map_pool2d(param, mapped_params);
  } else if (first_op.target() == "calculate_mx_qparam") {
    map_microscaling(param, mapped_params);
#if SUPPORT_SPMM
  } else if (first_op.target() == "spmm_csr") {
    map_spmm(operation, mapped_params, false);
#endif
  } else if (is_reduction_op(first_op.target())) {
    map_reduction(param, mapped_params);
  } else {
    map_vector_operations(param, mapped_params);
  }
}

std::deque<BaseParams*> get_ping_pong_params(std::deque<BaseParams*> params,
                                             int offset) {
  std::deque<BaseParams*> new_params;
  for (const auto base_param : params) {
    if (auto* mp = dynamic_cast<MatrixParams*>(base_param)) {
      auto* param = new MatrixParams(*mp);
      param->input_offset += offset;
      param->weight_offset += offset;
      param->bias_offset += offset;
      param->input_scale_offset += offset;
      param->weight_scale_offset += offset;
      param->output_offset += offset;
      param->dq_scale_offset += offset;
      param->dq_zero_point_offset += offset;
#if SUPPORT_SPMM
      param->spmm_indices_offset += offset;
      param->spmm_indptr_offset += offset;
      param->spmm_data_offset += offset;
#endif
      new_params.push_back(param);
    } else if (auto* vp = dynamic_cast<VectorParams*>(base_param)) {
      auto* param = new VectorParams(*vp);
      param->vector_fetch_0_offset += offset;
      param->vector_fetch_1_offset += offset;
      param->vector_fetch_2_offset += offset;
      param->vector_output_offset += offset;
      param->mx_scale_offset += offset;
      param->csr_data_offset += offset;
      param->csr_indices_offset += offset;
      param->csr_indptr_offset += offset;
      new_params.push_back(param);
    } else {
      new_params.push_back(base_param);
    }
  }
  return new_params;
}
