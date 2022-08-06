#pragma once

#include "src/AccelTypes.h"
#include "src/Params.h"
#include "test/common/VerificationTypes.h"
#include "src/ArchitectureParams.h"

void map_operation(const SimplifiedParams &params, MatrixParams &matrixParams,
                   bool &matrixParamsValid, VectorParams &vectorParams,
                   VectorInstructionConfig &vectorInstructionConfig,
                   bool &vectorParamsValid);
