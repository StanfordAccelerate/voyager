#pragma once

#define NO_SYSC

#include <vector>

// clang-format off
#include "src/PositTypes.h"
// clang-format on
#include "VerificationTypes.h"
#include "src/ArchitectureParams.h"
#include "test/common/UniversalPosit.h"

float compare_arrays(INPUT_DATATYPE *matrixA, std::string matrixA_name,
                     INPUT_DATATYPE *matrixB, std::string matrixB_name,
                     size_t size, std::string filename, bool accType);
float compare_arrays(INPUT_DATATYPE *matrixA, std::string matrixA_name,
                     float *matrixB, std::string matrixB_name, size_t size,
                     std::string filename, bool accType);
#ifndef NO_UNIVERSAL
float compare_arrays(INPUT_DATATYPE *matrixA, std::string matrixA_name,
                     UniversalPosit *matrixB, std::string matrixB_name,
                     size_t size, std::string filename, bool accType);
float compare_arrays(UniversalPosit *matrixA, std::string matrixA_name,
                     UniversalPosit *matrixB, std::string matrixB_name,
                     size_t size, std::string filename, bool accType);
float compare_arrays(UniversalPosit *matrixA, std::string matrixA_name,
                     float *matrixB, std::string matrixB_name, size_t size,
                     std::string filename, bool accType);
#endif
float compare_arrays(float *matrixA, std::string matrixA_name, float *matrixB,
                     std::string matrixB_name, size_t size,
                     std::string filename, bool accType);

int validateMapping(SimplifiedParams params);

void validateMemoryMapping(std::vector<Workload>);