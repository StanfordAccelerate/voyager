#pragma once
#define NO_SYSC
// clang-format off
#include "src/DataTypes.h"
// clang-format on
#include "src/ArchitectureParams.h"
#include "test/common/UniversalPosit.h"
#include "test/common/VerificationTypes.h"

void run_gold_model(const SimplifiedParams params, INPUT_DATATYPE *matrixA,
                    INPUT_DATATYPE *matrixB, INPUT_DATATYPE *matrixC,
                    INPUT_DATATYPE *biasMatrix, INPUT_DATATYPE *residualMatrix,
                    INPUT_DATATYPE *weightResidualMatrix);

void run_gold_model(const SimplifiedParams params, UniversalPosit *matrixA,
                    UniversalPosit *matrixB, UniversalPosit *matrixC,
                    UniversalPosit *biasMatrix, UniversalPosit *residualMatrix,
                    UniversalPosit *weightResidualMatrix);

void run_gold_model(const SimplifiedParams params, float *matrixA,
                    float *matrixB, float *matrixC, float *biasMatrix,
                    float *residualMatrix, float *weightResidualMatrix);