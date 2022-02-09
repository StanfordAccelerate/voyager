#pragma once

#include "src/AccelTypes.h"
#include "src/ArchitectureParams.h"
#include "test/common/VerificationTypes.h"

void run_gold_op(const SimplifiedParams params, INPUT_DATATYPE *matrixA,
                 INPUT_DATATYPE *matrixB, OUTPUT_DATATYPE *matrixC,
                 INPUT_DATATYPE *biasMatrix, OUTPUT_DATATYPE *residualMatrix);
