#include "test/common/MemoryModel.h"

#if __has_include(<filesystem>)
#include <filesystem>
#else
#include <experimental/filesystem>
namespace std {
namespace filesystem = experimental::filesystem;
}
#endif
#include <fstream>
#include <iostream>
#include <random>

MemoryModel::MemoryModel(bool isDut) : isDut(isDut) {}

double* readFileAsDouble(const std::string& filename, int size,
                         bool useDataFile) {
  // Files are written in binary format as dtype=float64 (double in c)
  char* tmpValuesArray = new char[size * sizeof(double)];
  double* tmpValuePtr = (double*)tmpValuesArray;

  if (useDataFile) {
    std::ifstream is(filename, std::ios::binary);
    if (!is.good() || !std::filesystem::exists(filename) ||
        !std::filesystem::is_regular_file(filename))
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

void MemoryModel::loadInputs(const SimplifiedParams& params,
                             const MemorySource& mem,
                             const std::string& filename, bool useDataFile) {
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
  if (params.SOFTMAX || params.SOFTMAX_GRAD || params.CROSS_ENTROPY_GRAD ||
      params.FC_GRAD) {
    C = 1;
  }
  if (params.WEIGHT_UPDATE) {
    X = K;
  }

  int size = STRIDE * Y * STRIDE * X * C;
  // std::cout << "size of inputs: " << size << std::endl;

  double* tmpValues = readFileAsDouble(filename, size, useDataFile);
  double* tmpValuePtr = tmpValues;

  if (params.REPLICATION) {
    for (int y = 0; y < STRIDE * Y; y++) {
      for (int x_o = 0; x_o < (STRIDE * X) / 4; x_o++) {
        for (int x_i = 0; x_i < 4; x_i++) {  // 4 packed together
          for (int c = 0; c < C; c++) {
            int x = x_o * 4 + x_i;
            double val = *(tmpValuePtr++);

            int address;
            if (isDut) {
              address = y * ((STRIDE * X) / 4) * 16 + x_o * 16 + x_i * 3 + c;
            } else {
              address = y * (STRIDE * X) * C + x * C + c;
            }

            writeToMemory(params.INPUT_OFFSET + address, val, mem, false);
          }
        }
      }
    }
  } else {
    for (int address = 0; address < size; address++) {
      double val = *(tmpValuePtr++);

      writeToMemory(params.INPUT_OFFSET + address, val, mem, false);
    }
  }

  delete[] tmpValues;
}

void MemoryModel::loadWeights(const SimplifiedParams& params,
                              const MemorySource& mem,
                              const std::string& filename, bool useDataFile) {
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
    C = 1;
  }
  if (params.NO_NORM_GRAD) {
    C = X;
  }
  if (params.CROSS_ENTROPY_GRAD) {
    C = X;
    K = 1;
  }

  int size = FY * FX * C * K;
  double* tmpValues = readFileAsDouble(filename, size, useDataFile);
  double* tmpValuePtr = tmpValues;

  for (int address = 0; address < size; address++) {
    double val = *(tmpValuePtr++);
    writeToMemory(params.WEIGHT_OFFSET + address, val, mem, false);
  }

  delete[] tmpValues;
}

void MemoryModel::loadBias(const SimplifiedParams& params,
                           const MemorySource& mem, const std::string& filename,
                           bool useDataFile) {
  int X = params.loops[0][params.inputXLoopIndex[0]] *
          params.loops[1][params.inputXLoopIndex[1]];
  int Y = params.loops[0][params.inputYLoopIndex[0]] *
          params.loops[1][params.inputYLoopIndex[1]];
  int C = params.loops[1][params.reductionLoopIndex[1]] * DIMENSION;
  int K = params.loops[0][params.weightLoopIndex[0]] *
          params.loops[1][params.weightLoopIndex[1]] * DIMENSION;
  int FX = params.loops[1][params.fxIndex];
  int FY = params.loops[1][params.fyIndex];
  // int STRIDE = params.STRIDE;
  // if (params.REPLICATION) {
  //   FX = 7;
  //   C = 3;
  // }

  int size = K;
  double* tmpValues = readFileAsDouble(filename, size, useDataFile);
  double* tmpValuePtr = tmpValues;

  for (int address = 0; address < size; address++) {
    double val = *(tmpValuePtr++);
    writeToMemory(params.BIAS_OFFSET + 2 * address, val, mem, true);
  }

  delete[] tmpValues;
}

void MemoryModel::loadResiduals(const SimplifiedParams& params,
                                const MemorySource& mem,
                                const std::string& filename, bool useDataFile) {
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
  if (params.SOFTMAX_GRAD) {
    K = 1;
  }

  int size = X * Y * K;
  double* tmpValues = readFileAsDouble(filename, size, useDataFile);
  double* tmpValuePtr = tmpValues;

  for (int address = 0; address < size; address++) {
    double val = *(tmpValuePtr++);

    // TODO: work with double precision
    writeToMemory(params.RESIDUAL_OFFSET + address, val, mem, false);
  }

  delete[] tmpValues;
}

void MemoryModel::loadReferenceOutput(const SimplifiedParams& params,
                                      const Files& files) {
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
  if (params.SOFTMAX || params.SOFTMAX_GRAD) {
    K = 1;
  }
  if (params.NO_NORM_GRAD || params.CROSS_ENTROPY_GRAD) {
    X = 1;
    Y = 1;
  }

  int size = params.NO_NORM_GRAD ? C : X * Y * K;
  double* tmpValues = readFileAsDouble(files.outputs_file, size, true);
  double* tmpValuePtr = tmpValues;

  for (int i = 0; i < size; i++) {
    double val = *(tmpValuePtr++);
    // TODO: work with double precision
    writeToReference(i, val);
  }

  delete[] tmpValues;
}

void MemoryModel::loadModelActivations(const SimplifiedParams& params,
                                       const Files& files,
                                       const MemoryMap& memoryMap,
                                       bool useDataFile) {
  if (!files.inputs_file.empty()) {
    loadInputs(params, memoryMap.inputs, files.inputs_file, useDataFile);
  }

  // TODO(fpedd): Revisit the params.WEIGHT_OFFSET != -1 logic here
  // We only want to load the 2nd input as weights if weights is false, but the
  // offset is not -1 (i.e. it is set to a valid value)
  if (!params.WEIGHT && params.WEIGHT_OFFSET != -1 &&
      !files.weights_file.empty()) {
    loadWeights(params, memoryMap.weights, files.weights_file, useDataFile);
  }
  if (params.RESIDUAL || params.RELU_GRAD || params.SOFTMAX_GRAD) {
    loadResiduals(params, memoryMap.residual, files.residual_file, useDataFile);
  }
  if (params.WEIGHT_SPLITTING) {
    // TODO: Hacky way of filling weight gradient memory
    SimplifiedParams copy = params;
    copy.WEIGHT_OFFSET = params.WEIGHT_GRADIENT_OFFSET;
    copy.BIAS_OFFSET = params.BIAS_GRADIENT_OFFSET;

    std::cerr << "Weight splitting" << std::endl;
    loadWeights(params, SRAM, files.weight_grad_file, useDataFile);
    loadBias(params, SRAM, files.bias_grad_file, useDataFile);
  }
}

void MemoryModel::loadModelParams(const SimplifiedParams& params,
                                  const Files& files,
                                  const MemoryMap& memoryMap,
                                  bool useDataFile) {
  // If weights is false, we don't load weights, but we load the second input as
  // weights from SRAM, see above
  if (params.WEIGHT) {
    loadWeights(params, memoryMap.weights, files.weights_file, useDataFile);
  }
  if (params.BIAS) {
    loadBias(params, memoryMap.bias, files.bias_file, useDataFile);
  }
}
