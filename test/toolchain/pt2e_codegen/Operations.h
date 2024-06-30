#pragma once

#include "src/AccelTypes.h"
#include "src/Params.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"
#include "test/toolchain/pt2e_codegen/MatrixOperation.h"
#include "test/toolchain/pt2e_codegen/ReduceOperation.h"
// #include "test/toolchain/pt2e_codegen/VectorOperation.h"

void MapPytorchOperation(const codegen::AcceleratorParam &param,
                         std::deque<BaseParams *> &mappedParams,
                         std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  if (param.has_matrix_param()) {
    MapMatrixOperation(param, mappedParams, opMemoryMaps);
  } else if (param.has_reduce_param()) {
    MapReduceOperation(param, mappedParams, opMemoryMaps);
  } else if (param.has_pooling_param()) {
    // TODO:
  } else if (param.has_reshape_param()) {
    // TODO:
  } else if (param.vector_params_size() > 0) {
    // TODO:
    // MapVectorOperations(param, mappedParams, opMemoryMaps);
  }
}
