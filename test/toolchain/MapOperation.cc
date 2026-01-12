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
#include "test/toolchain/Softmax.h"
#if SUPPORT_SPMM
#include "test/toolchain/SpMM.h"
#endif
#include "test/toolchain/VectorOps.h"

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
    map_spmm(param, mapped_params);
#endif
  } else {
    map_vector_operations(param, mapped_params);
  }
}
