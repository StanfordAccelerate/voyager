#include "test/common/DataLoader.h"

#include <fstream>
#include <iostream>
#include <random>

#define PIPE_INPUT 0

void save_double(INPUT_DATATYPE* array, double val) {
  float fval = (float)val;
  *array = INPUT_DATATYPE(fval);
}

void save_double(UniversalPosit* array, double val) {
  float fval = (float)val;
  *array = fval;
}

void save_double(float* array, double val) {
  float fval = (float)val;
  *array = fval;
}

double* read_file_as_double(const std::string& filename, int size,
                            bool useDataFile) {
  // Files are written in binary format as dtype=float64 (double in c)
  char* tmpValuesArray = new char[size * sizeof(double)];
  double* tmpValuePtr = (double*)tmpValuesArray;

  if (useDataFile) {
    std::ifstream is(filename, std::ios::binary);
    if (!is.good())
      throw std::runtime_error("File \"" + filename + "\" does not exist");
    is.read(tmpValuesArray, size * sizeof(double));
  } else {
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(-1, 1);

    for (int i = 0; i < size; i++) {
      tmpValuePtr[i] = (double)dis(e);
    }
  }

  return tmpValuePtr;
}

double* read_input_as_double(int size
                            ) {
  // Files are written in binary format as dtype=float64 (double in c)
  char* tmpValuesArray = new char[size * sizeof(double)];
  double* tmpValuePtr = (double*)tmpValuesArray;

    // if (!std::cin.good())
    //   throw std::runtime_error("STDIN is bad");
    std::cin.read(tmpValuesArray, size * sizeof(double));

  return tmpValuePtr;
}


void load_inputs(const SimplifiedParams& params, const std::string& filename,
                 bool useDataFile, INPUT_DATATYPE* acceleratorMemory,
                 INPUT_DATATYPE* goldMemory,
                 UniversalPosit* universalGoldMemory, float* floatGoldMemory) {
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

#if PIPE_INPUT == 1
  double* tmpValues = read_input_as_double(size);
#else
  double* tmpValues = read_file_as_double(filename, size, useDataFile);
#endif
  double* tmpValuePtr = tmpValues;

  if (params.REPLICATION) {
    for (int y = 0; y < STRIDE * Y; y++) {
      for (int x_o = 0; x_o < (STRIDE * X) / 4; x_o++) {
        for (int x_i = 0; x_i < 4; x_i++) {  // 4 packed together
          for (int c = 0; c < C; c++) {
            int x = x_o * 4 + x_i;
            double val = *(tmpValuePtr++);

            int address = y * ((STRIDE * X) / 4) * 16 + x_o * 16 + x_i * 3 + c;
            save_double(&acceleratorMemory[params.INPUT_OFFSET + address], val);

            address = y * (STRIDE * X) * C + x * C + c;
            save_double(&goldMemory[address], val);
            save_double(&universalGoldMemory[address], val);
            save_double(&floatGoldMemory[address], val);
          }
        }
      }
    }
  } else {
    for (int y = 0; y < STRIDE * Y; y++) {
      for (int x = 0; x < STRIDE * X; x++) {
        for (int c = 0; c < C; c++) {
          double val = *(tmpValuePtr++);

          int address = y * (STRIDE * X) * C + x * C + c;

          save_double(&acceleratorMemory[params.INPUT_OFFSET + address], val);
          save_double(&goldMemory[address], val);
          save_double(&universalGoldMemory[address], val);
          save_double(&floatGoldMemory[address], val);
        }
      }
    }
  }

  delete[] tmpValues;
}

void load_weights(const SimplifiedParams& params, const std::string& filename,
                  bool useDataFile, INPUT_DATATYPE* acceleratorMemory,
                  INPUT_DATATYPE* goldMemory,
                  UniversalPosit* universalGoldMemory, float* floatGoldMemory) {
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
  if (params.NO_NORM) {
    FX = 1;
    FY = 1;
    K = 1;
  }

  int size = FY * FX * C * K;
  double* tmpValues = read_file_as_double(filename, size, useDataFile);
  double* tmpValuePtr = tmpValues;

  for (int fy = 0; fy < FY; fy++) {
    for (int fx = 0; fx < FX; fx++) {
      for (int c = 0; c < C; c++) {
        for (int k = 0; k < K; k++) {
          int address = fy * FX * C * K + fx * C * K + c * K + k;
          double val = *(tmpValuePtr++);

          save_double(&acceleratorMemory[params.WEIGHT_OFFSET + address], val);
          save_double(&goldMemory[address], val);
          save_double(&universalGoldMemory[address], val);
          save_double(&floatGoldMemory[address], val);
        }
      }
    }
  }

  delete[] tmpValues;
}

void load_bias(const SimplifiedParams& params, const std::string& filename,
               bool useDataFile, INPUT_DATATYPE* acceleratorMemory,
               INPUT_DATATYPE* goldMemory, UniversalPosit* universalGoldMemory,
               float* floatGoldMemory) {
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
  if (params.NO_NORM) {
    K = C;
  }

  int size = K;
  double* tmpValues = read_file_as_double(filename, size, useDataFile);
  double* tmpValuePtr = tmpValues;

  if (params.BIAS) {
    for (int k = 0; k < K; k++) {
      double val = *(tmpValuePtr++);
      save_double(&acceleratorMemory[params.BIAS_OFFSET + k], val);
      save_double(&goldMemory[k], val);
      save_double(&universalGoldMemory[k], val);
      save_double(&floatGoldMemory[k], val);
    }
  }

  delete[] tmpValues;
}

void load_residual(const SimplifiedParams& params, const std::string& filename,
                   bool useDataFile, INPUT_DATATYPE* acceleratorMemory,
                   INPUT_DATATYPE* goldMemory,
                   UniversalPosit* universalGoldMemory,
                   float* floatGoldMemory) {
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
  double* tmpValues = read_file_as_double(filename, size, useDataFile);
  double* tmpValuePtr = tmpValues;

  for (int y = 0; y < Y; y++) {
    for (int x = 0; x < X; x++) {
      for (int k = 0; k < K; k++) {
        double val = *(tmpValuePtr++);

        int address = y * X * K + x * K + k;
        save_double(&acceleratorMemory[params.RESIDUAL_OFFSET + address], val);
        save_double(&goldMemory[address], val);
        save_double(&universalGoldMemory[address], val);
        save_double(&floatGoldMemory[address], val);
      }
    }
  }

  delete[] tmpValues;
}

void load_datafile_outputs(const SimplifiedParams params,
                           const std::string& filename,
                           INPUT_DATATYPE* outputMatrix,
                           UniversalPosit* universalOutputMatrix,
                           float* floatOutputMatrix) {
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
  if (params.NO_NORM) {
    K = C;
  }

  int size = X * Y * K;
  double* tmpValues = read_file_as_double(filename, size, true);
  double* tmpValuePtr = tmpValues;

  for (int y = 0; y < Y; y++) {
    for (int x = 0; x < X; x++) {
      for (int k = 0; k < K; k++) {
        double val = *(tmpValuePtr++);

        int address = y * X * K + x * K + k;
        save_double(&outputMatrix[address], val);
        save_double(&universalOutputMatrix[address], val);
        save_double(&floatOutputMatrix[address], val);
      }
    }
  }

  delete[] tmpValues;
}

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
    float* floatMatrixC, float* floatDataFileOutput) {
  load_inputs(params, dataDir + files.inputs_file, useDataFile,
              memoryMap.inputs == SRAM ? sramMemory : rramMemory, matrixA,
              universalMatrixA, floatMatrixA);
  load_weights(params, dataDir + files.weights_file, useDataFile,
               memoryMap.weights == SRAM ? sramMemory : rramMemory, matrixB,
               universalMatrixB, floatMatrixB);
  if (params.BIAS) {
    load_bias(params, dataDir + files.bias_file, useDataFile,
              memoryMap.bias == SRAM ? sramMemory : rramMemory, biasMatrix,
              universalBiasMatrix, floatBiasMatrix);
  }
  if (params.RESIDUAL) {
    load_residual(params, dataDir + files.residual_file, useDataFile,
                  memoryMap.residual == SRAM ? sramMemory : rramMemory,
                  residualMatrix, universalResidualMatrix, floatResidualMatrix);
  }
  if (useDataFile) {
    load_datafile_outputs(params, dataDir + files.outputs_file, dataFileOutput,
                          universalDataFileOutput, floatDataFileOutput);
  }
}

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
    float* floatMatrixC, float* floatDataFileOutput) {
  load_weights(params, dataDir + files.weights_file, useDataFile,
               memoryMap.weights == SRAM ? sramMemory : rramMemory, matrixB,
               universalMatrixB, floatMatrixB);
  if (params.BIAS) {
    load_bias(params, dataDir + files.bias_file, useDataFile,
              memoryMap.bias == SRAM ? sramMemory : rramMemory, biasMatrix,
              universalBiasMatrix, floatBiasMatrix);
  }
}
