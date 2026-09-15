#include "spdlog/spdlog.h"
#include "src/Params.h"
#include "test/common/GraphUtils.h"
#include "test/common/Utils.h"
#include "test/toolchain/LayerNorm.h"
#include "test/toolchain/MatrixOps.h"
#include "test/toolchain/MatrixVectorMultiply.h"
#include "test/toolchain/Pooling.h"
#include "test/toolchain/Reduction.h"
#include "test/toolchain/Softmax.h"
#include "test/toolchain/VectorOps.h"
#if SUPPORT_SPMM
#include "test/toolchain/SpMM.h"
#endif

void map_operation(const voyager::Operation& operation, const ScalarEnv& env,
                   std::deque<BaseParams*>& mapped_params) {
  const auto& op_list = get_prim_ops(operation);
  const auto& anchor_op = get_anchor_op(operation);

  if (is_gemm_op(strip_namespace(anchor_op.target()))) {
    auto input = resolve(anchor_op, "input", env);
    if (is_fc_layer(anchor_op) && input.dtype == "bfloat16") {
      map_matrix_vector_multiply(operation, env, mapped_params);
    } else {
      map_matrix_operation(operation, env, mapped_params);
    }
  } else if (strip_namespace(anchor_op.target()) == "layer_norm") {
    map_layer_norm(operation, env, mapped_params);
  } else if (strip_namespace(anchor_op.target()) == "softmax") {
    map_softmax(operation, env, mapped_params);
  } else if (strip_namespace(anchor_op.target()) == "max_pool2d" ||
             strip_namespace(anchor_op.target()) == "adaptive_avg_pool2d") {
    map_pool2d(operation, env, mapped_params);
#if SUPPORT_SPMM
  } else if (strip_namespace(anchor_op.target()) == "spmm_csr") {
    map_spmm(operation, env, mapped_params, false);
#endif
  } else if (is_reduction_op(strip_namespace(anchor_op.target()))) {
    map_reduction(operation, env, mapped_params);
  } else {
    map_vector_operations(operation, env, mapped_params);
  }
}
