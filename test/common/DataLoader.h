#pragma once

#include "src/ArchitectureParams.h"
#include "src/PositTypes.h"
#include "test/common/UniversalPosit.h"
#include "test/common/VerificationTypes.h"

void load_inputs(const SimplifiedParams& params, const std::string& filename,
                 bool useDataFile, INPUT_DATATYPE* acceleratorMemory,
                 INPUT_DATATYPE* goldMemory,
                 UniversalPosit* universalGoldMemory, float* floatGoldMemory);

void load_datafile_outputs(const SimplifiedParams params,
                           const std::string& filename,
                           INPUT_DATATYPE* outputMatrix,
                           UniversalPosit* universalOutputMatrix,
                           float* floatOutputMatrix);

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

void load_wb(const SimplifiedParams& params, const std::string& dataDir,
             const Files& files, const MemoryMap& memoryMap, bool useDataFile,
             INPUT_DATATYPE* sramMemory, INPUT_DATATYPE* rramMemory,
             INPUT_DATATYPE* matrixA, INPUT_DATATYPE* matrixB,
             INPUT_DATATYPE* biasMatrix, INPUT_DATATYPE* residualMatrix,
             INPUT_DATATYPE* matrixC, INPUT_DATATYPE* dataFileOutput,
             UniversalPosit* universalMatrixA, UniversalPosit* universalMatrixB,
             UniversalPosit* universalBiasMatrix,
             UniversalPosit* universalResidualMatrix,
             UniversalPosit* universalMatrixC,
             UniversalPosit* universalDataFileOutput, float* floatMatrixA,
             float* floatMatrixB, float* floatBiasMatrix,
             float* floatResidualMatrix, float* floatMatrixC,
             float* floatDataFileOutput);

void load_weight_and_bias(
    const SimplifiedParams& params, const std::string& dataDir,
    const Files& files, const MemoryMap& memoryMap, bool useDataFile,
    INPUT_DATATYPE* sramMemory, INPUT_DATATYPE* rramMemory,
    INPUT_DATATYPE* hlsWeightMatrix, INPUT_DATATYPE* hlsBiasMatrix,
    INPUT_DATATYPE* dataFileOutput, UniversalPosit* universalWeightMatrix,
    UniversalPosit* universalBiasMatrix,
    UniversalPosit* universalDataFileOutput, float* floatWeightMatrix,
    float* floatBiasMatrix, float* floatDataFileOutput);
