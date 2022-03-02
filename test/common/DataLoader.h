#pragma once

#include "src/ArchitectureParams.h"
#include "src/PositTypes.h"
#include "test/common/UniversalPosit.h"
#include "test/common/VerificationTypes.h"

void load_memory(
    const SimplifiedParams& params, const std::string& dataDir,
    const Files& files, const MemoryMap& memoryMap, bool useDataFile,
    INPUT_DATATYPE* sramMemory, INPUT_DATATYPE* rramMemory,
    INPUT_DATATYPE* matrixA, INPUT_DATATYPE* matrixB,
    INPUT_DATATYPE* biasMatrix, INPUT_DATATYPE* residualMatrix,
    INPUT_DATATYPE* matrixC, INPUT_DATATYPE* dataFileOutput,
    UniversalPosit* universalMatrixA, UniversalPosit* universalMatrixB,
    UniversalPosit* universalBiasMatrix,
    UniversalPosit* universalResidualMatrix, UniversalPosit* universalMatrixC,
    UniversalPosit* universalDataFileOutput, float* floatMatrixA,
    float* floatMatrixB, float* floatBiasMatrix, float* floatResidualMatrix,
    float* floatMatrixC, float* floatDataFileOutput);

void load_wb(
    const SimplifiedParams& params, const std::string& dataDir,
    const Files& files, const MemoryMap& memoryMap, bool useDataFile,
    INPUT_DATATYPE* sramMemory, INPUT_DATATYPE* rramMemory,
    INPUT_DATATYPE* matrixA, INPUT_DATATYPE* matrixB,
    INPUT_DATATYPE* biasMatrix, INPUT_DATATYPE* residualMatrix,
    INPUT_DATATYPE* matrixC, INPUT_DATATYPE* dataFileOutput,
    UniversalPosit* universalMatrixA, UniversalPosit* universalMatrixB,
    UniversalPosit* universalBiasMatrix,
    UniversalPosit* universalResidualMatrix, UniversalPosit* universalMatrixC,
    UniversalPosit* universalDataFileOutput, float* floatMatrixA,
    float* floatMatrixB, float* floatBiasMatrix, float* floatResidualMatrix,
    float* floatMatrixC, float* floatDataFileOutput);

void load_datafile_outputs(const SimplifiedParams params,
                           const std::string& filename,
                           INPUT_DATATYPE* outputMatrix,
                           UniversalPosit* universalOutputMatrix,
                           float* floatOutputMatrix);