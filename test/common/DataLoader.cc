#include "test/common/DataLoader.h"

void load_inputs(const SimplifiedParams& params, const std::string& filename,
                 bool useDataFile, INPUT_DATATYPE* acceleratorMemory,
                 INPUT_DATATYPE* goldMemory) {
  int X = params.loops[0][params.inputXLoopIndex[0]] *
          params.loops[1][params.inputXLoopIndex[1]];
  int Y = params.loops[0][params.inputYLoopIndex[0]] *
          params.loops[1][params.inputYLoopIndex[1]];
  int C = params.loops[1][params.reductionLoopIndex[1]] * DIMENSION;
  int K = params.loops[0][params.weightLoopIndex[0]] *
          params.loops[1][params.weightLoopIndex[1]] * DIMENSION;
  int FX = params.loops[1][params.fxIndex];
  int FY = params.loops[1][params.fyIndex];
  int STRIDE = params.STRIDE;
  if (params.REPLICATION) {
    FX = 7;
    C = 3;
  }

  int size = STRIDE * Y * STRIDE * X * C;
  char* tmpValues = new char[size];
  char* tmpValuePtr = tmpValues;

  if (useDataFile) {
    std::ifstream is(filename, std::ios::binary);
    if (!is.good())
      throw std::runtime_error("File \"" + filename + "\" does not exist");
    is.read(tmpValues, size);

  } else {
    for (int i = 0; i < size; i++) {
      tmpValues[i] = rand() % 128;
    }
  }

  // if (params.REPLICATION) {
    // TODO:WARNING:CHANGE: quick fix
  if (false) {
    for (int y = 0; y < STRIDE * Y; y++) {
      for (int x_o = 0; x_o < (STRIDE * X) / 4; x_o++) {
        for (int x_i = 0; x_i < 4; x_i++) {  // 4 packed together
          for (int c = 0; c < C; c++) {
            int x = x_o * 4 + x_i;
            int val = (int)*(tmpValuePtr++);

            int address = y * ((STRIDE * X) / 4) * 16 + x_o * 16 + x_i * 3 + c;
            acceleratorMemory[params.INPUT_OFFSET + address] = val;

            address = y * (STRIDE * X) * C + x * C + c;
            goldMemory[address] = val;
          }
        }
      }
    }
  } else {
    for (int y = 0; y < STRIDE * Y; y++) {
      for (int x = 0; x < STRIDE * X; x++) {
        for (int c = 0; c < C; c++) {
          int val = (int)*(tmpValuePtr++);

          int address = y * (STRIDE * X) * C + x * C + c;

          acceleratorMemory[params.INPUT_OFFSET + address] = val;
          goldMemory[address] = val;
        }
      }
    }
  }

  delete[] tmpValues;
}

void load_weights(const SimplifiedParams& params, const std::string& filename,
                  bool useDataFile, INPUT_DATATYPE* acceleratorMemory,
                  INPUT_DATATYPE* goldMemory) {
  int X = params.loops[0][params.inputXLoopIndex[0]] *
          params.loops[1][params.inputXLoopIndex[1]];
  int Y = params.loops[0][params.inputYLoopIndex[0]] *
          params.loops[1][params.inputYLoopIndex[1]];
  int C = params.loops[1][params.reductionLoopIndex[1]] * DIMENSION;
  int K = params.loops[0][params.weightLoopIndex[0]] *
          params.loops[1][params.weightLoopIndex[1]] * DIMENSION;
  int FX = params.loops[1][params.fxIndex];
  int FY = params.loops[1][params.fyIndex];
  int STRIDE = params.STRIDE;
  if (params.REPLICATION) {
    FX = 7;
    C = 3;
  }

  int size = FY * FX * C * K;
  char* tmpValues = new char[size];
  char* tmpValuePtr = tmpValues;

  if (useDataFile) {
    std::ifstream is(filename, std::ios::binary);
    if (!is.good())
      throw std::runtime_error("File \"" + filename + "\" does not exist");
    is.read(tmpValues, size);
  } else {
    for (int i = 0; i < size; i++) {
      tmpValues[i] = rand() % 128;
    }
  }

  for (int fy = 0; fy < FY; fy++) {
    for (int fx = 0; fx < FX; fx++) {
      for (int c = 0; c < C; c++) {
        for (int k = 0; k < K; k++) {
          int address = fy * FX * C * K + fx * C * K + c * K + k;
          int val = (int)*(tmpValuePtr++);

          acceleratorMemory[params.WEIGHT_OFFSET + address] = val;
          goldMemory[address] = val;
        }
      }
    }
  }

  delete[] tmpValues;
}

void load_bias(const SimplifiedParams& params, const std::string& filename,
               bool useDataFile, INPUT_DATATYPE* acceleratorMemory,
               INPUT_DATATYPE* goldMemory) {
  int X = params.loops[0][params.inputXLoopIndex[0]] *
          params.loops[1][params.inputXLoopIndex[1]];
  int Y = params.loops[0][params.inputYLoopIndex[0]] *
          params.loops[1][params.inputYLoopIndex[1]];
  int C = params.loops[1][params.reductionLoopIndex[1]] * DIMENSION;
  int K = params.loops[0][params.weightLoopIndex[0]] *
          params.loops[1][params.weightLoopIndex[1]] * DIMENSION;
  int FX = params.loops[1][params.fxIndex];
  int FY = params.loops[1][params.fyIndex];
  int STRIDE = params.STRIDE;
  if (params.REPLICATION) {
    FX = 7;
    C = 3;
  }

  int size = K;
  char* tmpValues = new char[size];
  char* tmpValuePtr = tmpValues;

  if (useDataFile) {
    std::ifstream is(filename, std::ios::binary);
    if (!is.good())
      throw std::runtime_error("File \"" + filename + "\" does not exist");
    is.read(tmpValues, size);

  } else {
    for (int i = 0; i < size; i++) {
      tmpValues[i] = rand() % 128;
    }
  }

  if (params.BIAS) {
    for (int k = 0; k < K; k++) {
      int val = (int)*(tmpValuePtr++);
      acceleratorMemory[params.BIAS_OFFSET + k] = val;
      goldMemory[k] = val;
    }
  }

  delete[] tmpValues;
}

void load_residual(const SimplifiedParams& params, const std::string& filename,
                   bool useDataFile, INPUT_DATATYPE* acceleratorMemory,
                   INPUT_DATATYPE* goldMemory) {
  int X = params.loops[0][params.inputXLoopIndex[0]] *
          params.loops[1][params.inputXLoopIndex[1]];
  int Y = params.loops[0][params.inputYLoopIndex[0]] *
          params.loops[1][params.inputYLoopIndex[1]];
  int C = params.loops[1][params.reductionLoopIndex[1]] * DIMENSION;
  int K = params.loops[0][params.weightLoopIndex[0]] *
          params.loops[1][params.weightLoopIndex[1]] * DIMENSION;
  int FX = params.loops[1][params.fxIndex];
  int FY = params.loops[1][params.fyIndex];
  int STRIDE = params.STRIDE;
  if (params.REPLICATION) {
    FX = 7;
    C = 3;
  }

  int size = X * Y * K;
  char* tmpValues = new char[size];
  char* tmpValuePtr = tmpValues;

  if (useDataFile) {
    std::ifstream is(filename, std::ios::binary);
    if (!is.good())
      throw std::runtime_error("File \"" + filename + "\" does not exist");
    is.read(tmpValues, size);

  } else {
    for (int i = 0; i < size; i++) {
      tmpValues[i] = rand() % 128;
    }
  }

  for (int y = 0; y < Y; y++) {
    for (int x = 0; x < X; x++) {
      for (int k = 0; k < K; k++) {
        int val = (int)*(tmpValuePtr++);

        int address = y * X * K + x * K + k;
        acceleratorMemory[params.RESIDUAL_OFFSET + address] = val;
        goldMemory[address] = val;
      }
    }
  }

  delete[] tmpValues;
}

void load_datafile_outputs(const SimplifiedParams params, const std::string& filename,
                           INPUT_DATATYPE* outputMatrix) {
  int X = params.loops[0][params.inputXLoopIndex[0]] *
          params.loops[1][params.inputXLoopIndex[1]];
  int Y = params.loops[0][params.inputYLoopIndex[0]] *
          params.loops[1][params.inputYLoopIndex[1]];
  int C = params.loops[1][params.reductionLoopIndex[1]] * DIMENSION;
  int K = params.loops[0][params.weightLoopIndex[0]] *
          params.loops[1][params.weightLoopIndex[1]] * DIMENSION;
  int FX = params.loops[1][params.fxIndex];
  int FY = params.loops[1][params.fyIndex];
  int STRIDE = params.STRIDE;
  if (params.REPLICATION) {
    FX = 7;
    C = 3;
  }
  if (params.MAXPOOL) {
    X = X / 2;
    Y = Y / 2;
  }
  if (params.AVGPOOL) {
    X = 1;
    Y = 1;
  }

  int size = X * Y * K;
  char* tmpValues = new char[size];
  char* tmpValuePtr = tmpValues;

  std::ifstream is(filename, std::ios::binary);
  if (!is.good())
    throw std::runtime_error("File \"" + filename + "\" does not exist");
  is.read(tmpValues, size);

  for (int y = 0; y < Y; y++) {
    for (int x = 0; x < X; x++) {
      for (int k = 0; k < K; k++) {
        int val = (int)*(tmpValuePtr++);

        int address = y * X * K + x * K + k;
        outputMatrix[address] = val;
      }
    }
  }

  delete[] tmpValues;
}

void load_wb(const SimplifiedParams& params, const std::string& dataDir,
                 const Files& files, const MemoryMap& memoryMap,
                 bool useDataFile, INPUT_DATATYPE* sramMemory,
                 INPUT_DATATYPE* rramMemory, INPUT_DATATYPE* matrixA,
                 INPUT_DATATYPE* matrixB, INPUT_DATATYPE* biasMatrix,
                 INPUT_DATATYPE* residualMatrix, INPUT_DATATYPE* matrixC,
                 INPUT_DATATYPE* dataFileOutput) {
  load_weights(params, dataDir + files.weights_file, useDataFile,
               memoryMap.weights == SRAM ? sramMemory : rramMemory, matrixB);
  if (params.BIAS) {
    load_bias(params, dataDir + files.bias_file, useDataFile,
              memoryMap.bias == SRAM ? sramMemory : rramMemory, biasMatrix);
  }
                 }

void load_memory(const SimplifiedParams& params, const std::string& dataDir,
                 const Files& files, const MemoryMap& memoryMap,
                 bool useDataFile, INPUT_DATATYPE* sramMemory,
                 INPUT_DATATYPE* rramMemory, INPUT_DATATYPE* matrixA,
                 INPUT_DATATYPE* matrixB, INPUT_DATATYPE* biasMatrix,
                 INPUT_DATATYPE* residualMatrix, INPUT_DATATYPE* matrixC,
                 INPUT_DATATYPE* dataFileOutput) {
  load_inputs(params, dataDir + files.inputs_file, useDataFile,
              memoryMap.inputs == SRAM ? sramMemory : rramMemory, matrixA);
  load_weights(params, dataDir + files.weights_file, useDataFile,
               memoryMap.weights == SRAM ? sramMemory : rramMemory, matrixB);
  if (params.BIAS) {
    load_bias(params, dataDir + files.bias_file, useDataFile,
              memoryMap.bias == SRAM ? sramMemory : rramMemory, biasMatrix);
  }
  if (params.RESIDUAL) {
    load_residual(params, dataDir + files.residual_file, useDataFile,
                  memoryMap.residual == SRAM ? sramMemory : rramMemory,
                  residualMatrix);
  }
  if (useDataFile) {
    load_datafile_outputs(params, dataDir + files.outputs_file, dataFileOutput);
  }
}
