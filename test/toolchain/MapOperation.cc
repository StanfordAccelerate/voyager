#include "spdlog/spdlog.h"
#include "src/AccelTypes.h"
#include "src/Params.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"
#include "test/toolchain/LayerNorm.h"
#include "test/toolchain/MatrixOps.h"
#include "test/toolchain/MatrixVectorMultiply.h"
#include "test/toolchain/Microscaling.h"
#include "test/toolchain/Pooling.h"
#include "test/toolchain/Softmax.h"
#include "test/toolchain/VectorOps.h"

void MapOperation(const Operation& operation,
                  std::deque<BaseParams*>& mapped_params,
                  std::deque<AcceleratorMemoryMap>& memory_maps) {
  const auto param = operation.param;
  const auto op_list = get_op_list(param);
  const auto first_op = op_list[0];

  if (GEMM_OPS.find(first_op.target()) != GEMM_OPS.end()) {
    auto input = first_op.kwargs().at("input").tensor();
    if (is_fc_layer(first_op) && input.dtype() == "bfloat16") {
      MapMatrixVectorMultiply(param, mapped_params, memory_maps);
    } else {
      MapMatrixOperation(operation, mapped_params, memory_maps);
    }
  } else if (first_op.target() == "layer_norm") {
    MapLayerNorm(param, mapped_params, memory_maps);
  } else if (first_op.target() == "softmax") {
    MapSoftmax(param, mapped_params, memory_maps);
  } else if (first_op.target() == "max_pool2d" ||
             first_op.target() == "adaptive_avg_pool2d") {
    MapPoolingOperation(param, mapped_params, memory_maps);
  } else if (first_op.target() == "calculate_mx_qparam") {
    MapMicroscaling(param, mapped_params, memory_maps);
  } else {
    MapVectorOperations(param, mapped_params, memory_maps);
  }
}
