#pragma once

#include "src/ArchitectureParams.h"
#include "src/PositTypes.h"
#include "test/common/VerificationTypes.h"
#include "test/common/UniversalPosit.h"

void load_memory(const SimplifiedParams& params, const std::string& dataDir,
                 const Files& files, const MemoryMap& memoryMap,
                 bool useDataFile, INPUT_DATATYPE* sramMemory,
                 INPUT_DATATYPE* rramMemory, INPUT_DATATYPE* matrixA,
                 INPUT_DATATYPE* matrixB, INPUT_DATATYPE* biasMatrix,
                 INPUT_DATATYPE* residualMatrix, INPUT_DATATYPE* matrixC,
                 INPUT_DATATYPE* dataFileOutput,
                 UniversalPosit* universalMatrixA,
                 UniversalPosit* universalMatrixB,
                 UniversalPosit* universalBiasMatrix,
                 UniversalPosit* universalResidualMatrix,
                 UniversalPosit* universalMatrixC,
                 UniversalPosit* universalDataFileOutput);

// FIXME: add universal here
// void load_wb(const SimplifiedParams& params, const std::string& dataDir,
//              const Files& files, const MemoryMap& memoryMap, bool
//              useDataFile, INPUT_DATATYPE* sramMemory, INPUT_DATATYPE*
//              rramMemory, INPUT_DATATYPE* matrixA, INPUT_DATATYPE* matrixB,
//              INPUT_DATATYPE* biasMatrix, INPUT_DATATYPE* residualMatrix,
//              INPUT_DATATYPE* matrixC, INPUT_DATATYPE* dataFileOutput);
