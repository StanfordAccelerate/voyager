#pragma once
#define NO_SYSC
// clang-format off
#include "src/DataTypes.h"
// clang-format on
#include "src/ArchitectureParams.h"
#include "test/common/VerificationTypes.h"
#include "test/common/operations/MatrixOps.h"
#include "test/common/operations/Pooling.h"
#include "test/common/operations/ReduceOps.h"
#include "test/common/operations/ReshapeOps.h"
#include "test/common/operations/VectorOps.h"
#include "test/compiler/proto/param.pb.h"


void run_gold_model(const codegen::AcceleratorParam &param,
                    std::vector<float *> args);

void run_gold_model(const codegen::AcceleratorParam &param,
                    std::vector<INPUT_DATATYPE *> args);
