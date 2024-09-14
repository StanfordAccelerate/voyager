#include "src/AccelTypes.h"
#include "src/Params.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"
#include "test/toolchain/MatrixOps.h"
#include "test/toolchain/MatrixVectorMultiply.h"
#include "test/toolchain/Pooling.h"
#include "test/toolchain/ReduceOps.h"
#include "test/toolchain/ReshapeOps.h"
#include "test/toolchain/VectorOps.h"

void MapOperation(const codegen::AcceleratorParam &param,
                  std::deque<BaseParams *> &mappedParams,
                  std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  if (param.has_matrix_param()) {
    const auto &inputs = param.matrix_param().input();
    int dim = 1;
    for (int i = 0; i < inputs.shape_size() - 1; i++) {
      dim *= inputs.shape(i);
    }

    if (dim == 1) {
      MapMatrixVectorMultiply(param, mappedParams, opMemoryMaps);
    } else {
      MapMatrixOperation(param, mappedParams, opMemoryMaps);
    }
  } else if (param.has_reduce_param()) {
    MapReduceOperation(param, mappedParams, opMemoryMaps);
  } else if (param.has_pooling_param()) {
    MapPoolingOperation(param, mappedParams, opMemoryMaps);
  } else if (param.has_reshape_param()) {
    MapReshapeOperation(param, mappedParams, opMemoryMaps);
  } else if (param.vector_params_size() > 0) {
    MapVectorOperations(param, mappedParams, opMemoryMaps);
  }
}
