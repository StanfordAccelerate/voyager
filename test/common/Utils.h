#pragma once

#include "VerificationTypes.h"
#include "src/ArchitectureParams.h"
#include "src/PositTypes.h"
#include "test/common/UniversalPosit.h"

int compare_arrays(INPUT_DATATYPE *matrixA, INPUT_DATATYPE *matrixB,
                   size_t size, std::string &filename);
int compare_arrays(INPUT_DATATYPE *matrixA, UniversalPosit *matrixB,
                   size_t size, std::string &filename);
int compare_arrays(UniversalPosit *matrixA, UniversalPosit *matrixB,
                   size_t size, std::string &filename);