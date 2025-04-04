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

void MapOperation(const Operation &operation,
                  std::deque<BaseParams *> &mappedParams,
                  std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  const auto param = operation.param;
  const auto op_list = get_op_list(param);
  const auto first_op = op_list[0];

  if (GEMM_OPS.find(first_op.target()) != GEMM_OPS.end()) {
    const auto &input = first_op.kwargs().at("input").tensor();

    int dim = 1;
    for (int i = 0; i < input.shape_size() - 1; i++) {
      dim *= input.shape(i);
    }

    if (dim == 1) {
      MapMatrixVectorMultiply(param, mappedParams, opMemoryMaps);
    } else {
      MapMatrixOperation(operation, mappedParams, opMemoryMaps);
    }
  } else if (first_op.target() == "layer_norm") {
    MapLayerNorm(param, mappedParams, opMemoryMaps);
  } else if (first_op.target() == "softmax") {
    MapSoftmax(param, mappedParams, opMemoryMaps);
  } else if (first_op.target() == "max_pool2d" ||
             first_op.target() == "adaptive_avg_pool2d") {
    MapPoolingOperation(param, mappedParams, opMemoryMaps);
  } else if (first_op.target() == "calculate_mx_qparam") {
    MapMicroscaling(param, mappedParams, opMemoryMaps);
  } else {
    MapVectoreduce_operations(param, mappedParams, opMemoryMaps);
  }
}
