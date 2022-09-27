#pragma once

#define NO_SYSC

// clang-format off
#include "src/PositTypes.h"
// clang-format on
#include "VerificationTypes.h"
#include "src/ArchitectureParams.h"
#include "test/common/UniversalPosit.h"

int compare_arrays(INPUT_DATATYPE *matrixA, INPUT_DATATYPE *matrixB,
                   size_t size, std::string filename, bool accType);
int compare_arrays(INPUT_DATATYPE *matrixA, float *matrixB, size_t size,
                   std::string filename, bool accType);
#ifndef NO_UNIVERSAL
int compare_arrays(INPUT_DATATYPE *matrixA, UniversalPosit *matrixB,
                   size_t size, std::string filename, bool accType);
int compare_arrays(UniversalPosit *matrixA, UniversalPosit *matrixB,
                   size_t size, std::string filename, bool accType);
int compare_arrays(UniversalPosit *matrixA, float *matrixB, size_t size,
                   std::string filename, bool accType);
#endif
int compare_arrays(float *matrixA, float *matrixB, size_t size,
                   std::string filename, bool accType);

int validateMapping(SimplifiedParams params);