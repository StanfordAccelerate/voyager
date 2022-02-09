#pragma once

#include "src/AccelTypes.h"
#include "src/ArchitectureParams.h"
#include "test/common/VerificationTypes.h"

void load_memory(const SimplifiedParams& params, const std::string& dataDir,
                 const Files& files, const MemoryMap& memoryMap,
                 bool useDataFile, INPUT_DATATYPE* sramMemory,
                 INPUT_DATATYPE* rramMemory, INPUT_DATATYPE* matrixA,
                 INPUT_DATATYPE* matrixB, INPUT_DATATYPE* biasMatrix,
                 INPUT_DATATYPE* residualMatrix, INPUT_DATATYPE* matrixC,
                 INPUT_DATATYPE* dataFileOutput);

void load_wb(const Params& params, const std::string& dataDir,
                 const Files& files, const MemoryMap& memoryMap,
                 bool useDataFile, INPUT_DATATYPE* sramMemory,
                 INPUT_DATATYPE* rramMemory, INPUT_DATATYPE* matrixA,
                 INPUT_DATATYPE* matrixB, INPUT_DATATYPE* biasMatrix,
                 INPUT_DATATYPE* residualMatrix, INPUT_DATATYPE* matrixC,
                 INPUT_DATATYPE* dataFileOutput);
