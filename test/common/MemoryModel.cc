#include "test/common/MemoryModel.h"

#include <fstream>
#include <iostream>
#include <random>

#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"

MemoryModel::MemoryModel(bool isDut) : isDut(isDut) {}

float* readFileAsFloat(const std::string& filename, int size,
                       bool useDataFile) {
  // Files are written in binary format as dtype=float64 (float in c)
  char* tmpValuesArray = new char[size * sizeof(float)];
  float* tmpValuePtr = (float*)tmpValuesArray;

  if (useDataFile) {
    std::ifstream is(filename, std::ios::binary);
    if (!is.good())
      throw std::runtime_error("File \"" + filename + "\" does not exist");
    is.read(tmpValuesArray, size * sizeof(float));
  } else {
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(-1, 1);

    for (int i = 0; i < size; i++) {
      tmpValuePtr[i] = (float)dis(e);
    }
  }

  return tmpValuePtr;
}

void MemoryModel::loadInputs(const SimplifiedParams& params,
                             const MemorySource& mem,
                             const std::string& filename, bool useDataFile) {
  size_t X = params.loops[0][params.inputXLoopIndex[0]] *
             params.loops[1][params.inputXLoopIndex[1]];
  size_t Y = params.loops[0][params.inputYLoopIndex[0]] *
             params.loops[1][params.inputYLoopIndex[1]];
  size_t C = params.loops[1][params.reductionLoopIndex[1]] * (16);
  size_t K = params.loops[0][params.weightLoopIndex[0]] *
             params.loops[1][params.weightLoopIndex[1]] * (16);
  size_t FX = params.loops[1][params.fxIndex];
  size_t FY = params.loops[1][params.fyIndex];
  size_t STRIDE = params.STRIDE;
  if (params.REPLICATION) {
    FX = 7;
    C = 3;
  }

  size_t size = STRIDE * Y * STRIDE * X * C;
  if (params.SOFTMAX || params.SOFTMAX_GRAD) {
    size = X * Y;
  } else if (params.CROSS_ENTROPY_GRAD) {
    size = X;
  }

  float* tmpValues = readFileAsFloat(filename, size, useDataFile);
  float* tmpValuePtr = tmpValues;

  // Pytorch input tensors are written as (B, C, H, W)
  // Our memory is written as (H, W, C)
  // We need to do some reshaping to get the correct values in the correct place
  xt::xarray<float> tensor =
      xt::adapt(tmpValues, size, xt::no_ownership(),
                std::vector<size_t>({1, C, STRIDE * Y, STRIDE * X}));
  if (params.name == "conv2d") {
    tensor = xt::transpose(tensor, {2, 3, 1, 0});
  }

  if (params.REPLICATION) {
    int packingFactor;  // number of 3-channel values packed into a single word

    if (IC_DIMENSION == 16) {
      packingFactor = 4;
    } else if (IC_DIMENSION == 32) {
      packingFactor = 8;
    }

    auto it = tensor.begin();
    for (int y = 0; y < STRIDE * Y; y++) {
      for (int x_o = 0; x_o < (STRIDE * X) / packingFactor; x_o++) {
        for (int x_i = 0; x_i < packingFactor; x_i++) {
          for (int c = 0; c < C; c++) {
            int x = x_o * packingFactor + x_i;

            float val = *(it++);

            int address;
            if (isDut) {
              address = y * ((STRIDE * X) / packingFactor) * IC_DIMENSION +
                        x_o * IC_DIMENSION + x_i * 3 + c;
            } else {
              address = y * (STRIDE * X) * C + x * C + c;
            }

            if (params.ACC_T_INPUT) {
              writeToMemory(params.INPUT_OFFSET + 2 * address, val, mem, true);
            } else {
              writeToMemory(params.INPUT_OFFSET + address, val, mem, false);
            }
          }
        }
      }
    }
  } else {
    int address = 0;
    for (auto it = tensor.begin(); it != tensor.end(); ++it) {
      float val = *it;

      if (params.ACC_T_INPUT) {
        writeToMemory(params.INPUT_OFFSET + 2 * address, val, mem, true);
      } else {
        writeToMemory(params.INPUT_OFFSET + address, val, mem, false);
      }
      address++;
    }
  }

  delete[] tmpValues;
}

void MemoryModel::loadWeights(const SimplifiedParams& params,
                              const MemorySource& mem,
                              const std::string& filename, bool useDataFile) {
  size_t X = params.loops[0][params.inputXLoopIndex[0]] *
             params.loops[1][params.inputXLoopIndex[1]];
  size_t Y = params.loops[0][params.inputYLoopIndex[0]] *
             params.loops[1][params.inputYLoopIndex[1]];
  size_t C = params.loops[1][params.reductionLoopIndex[1]] * (16);
  size_t K = params.loops[0][params.weightLoopIndex[0]] *
             params.loops[1][params.weightLoopIndex[1]] * (16);
  size_t FX = params.loops[1][params.fxIndex];
  size_t FY = params.loops[1][params.fyIndex];
  size_t STRIDE = params.STRIDE;
  if (params.REPLICATION) {
    FX = 7;
    C = 3;
  }

  size_t size = FY * FX * C * K;
  if (params.NO_NORM) {
    size = C;
  } else if (params.CROSS_ENTROPY_GRAD) {
    size = X;
  } else if (params.NO_NORM_GRAD) {
    size = X * K;
  } else if (params.WEIGHT_UPDATE) {
    size = X * C;
  }

  float* tmpValues = readFileAsFloat(filename, size, useDataFile);
  float* tmpValuePtr = tmpValues;

  // Pytorch weight tensors are written as (OC, IC, KH, KW)
  // Our memory is written as (KH, KW, IC, OC)
  // We need to do some reshaping to get the correct values in the correct place
  xt::xarray<float> tensor = xt::adapt(tmpValues, size, xt::no_ownership(),
                                       std::vector<size_t>({K, C, FY, FX}));
  if (filename.find("param_constant") != std::string::npos) {
    tensor = xt::transpose(tensor, {2, 3, 1, 0});
  }

  int address = 0;
  for (auto it = tensor.begin(); it != tensor.end(); ++it) {
    float val = *it;
    if (params.ACC_T_WEIGHT || params.NO_NORM) {
      writeToMemory(params.WEIGHT_OFFSET + 2 * address, val, mem, true);
    } else {
      writeToMemory(params.WEIGHT_OFFSET + address, val, mem, false);
    }
    address++;
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
  int C = params.loops[1][params.reductionLoopIndex[1]] * (16);
  int K = params.loops[0][params.weightLoopIndex[0]] *
          params.loops[1][params.weightLoopIndex[1]] * (16);
  int FX = params.loops[1][params.fxIndex];
  int FY = params.loops[1][params.fyIndex];

  int size = K;
  float* tmpValues = readFileAsFloat(filename, size, useDataFile);
  float* tmpValuePtr = tmpValues;

  for (int address = 0; address < size; address++) {
    float val = *(tmpValuePtr++);
    writeToMemory(params.BIAS_OFFSET + 2 * address, val, mem, true);
  }

  delete[] tmpValues;
}

void MemoryModel::loadResiduals(const SimplifiedParams& params,
                                const MemorySource& mem,
                                const std::string& filename, bool useDataFile) {
  size_t X = params.loops[0][params.inputXLoopIndex[0]] *
             params.loops[1][params.inputXLoopIndex[1]];
  size_t Y = params.loops[0][params.inputYLoopIndex[0]] *
             params.loops[1][params.inputYLoopIndex[1]];
  size_t C = params.loops[1][params.reductionLoopIndex[1]] * (16);
  size_t K = params.loops[0][params.weightLoopIndex[0]] *
             params.loops[1][params.weightLoopIndex[1]] * (16);
  size_t FX = params.loops[1][params.fxIndex];
  size_t FY = params.loops[1][params.fyIndex];
  size_t STRIDE = params.STRIDE;
  if (params.REPLICATION) {
    FX = 7;
    C = 3;
  }

  size_t size = X * Y * K;
  if (params.SOFTMAX_GRAD) {
    size = X * Y;
  }

  float* tmpValues = readFileAsFloat(filename, size, useDataFile);
  float* tmpValuePtr = tmpValues;

  // Pytorch input tensors are written as (C, H, W)
  // Our memory is written as (H, W, C)
  // We need to do some reshaping to get the correct values in the correct place
  xt::xarray<float> tensor = xt::adapt(tmpValues, size, xt::no_ownership(),
                                       std::vector<size_t>({K, Y, X}));
  if (params.name == "conv2d") {
    tensor = xt::transpose(tensor, {1, 2, 0});
  }

  int address = 0;
  for (auto it = tensor.begin(); it != tensor.end(); ++it) {
    float val = *it;

    if (params.ACC_T_RESIDUAL) {
      writeToMemory(params.RESIDUAL_OFFSET + 2 * address, val, mem, true);
    } else {
      writeToMemory(params.RESIDUAL_OFFSET + address, val, mem, false);
    }
    address++;
  }

  delete[] tmpValues;
}

void MemoryModel::loadReferenceOutput(const SimplifiedParams& params,
                                      const Files& files, bool floatPrecision) {
  size_t X = params.loops[0][params.inputXLoopIndex[0]] *
             params.loops[1][params.inputXLoopIndex[1]];
  size_t Y = params.loops[0][params.inputYLoopIndex[0]] *
             params.loops[1][params.inputYLoopIndex[1]];
  size_t C = params.loops[1][params.reductionLoopIndex[1]] * (16);
  size_t K = params.loops[0][params.weightLoopIndex[0]] *
             params.loops[1][params.weightLoopIndex[1]] * (16);
  size_t FX = params.loops[1][params.fxIndex];
  size_t FY = params.loops[1][params.fyIndex];
  size_t STRIDE = params.STRIDE;
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

  size_t size = X * Y * K;
  if (params.SOFTMAX || params.SOFTMAX_GRAD) {
    size = X * Y;
  } else if (params.CROSS_ENTROPY_GRAD) {
    size = X;
  } else if (params.NO_NORM_GRAD) {
    size = C;
  } else if (params.WEIGHT_UPDATE) {
    size = X * C;
  }

  float* tmpValues = readFileAsFloat(files.outputs_file, size, true);
  float* tmpValuePtr = tmpValues;

  // Pytorch input tensors are written as (C, H, W)
  // Our memory is written as (H, W, C)
  // We need to do some reshaping to get the correct values in the correct place
  xt::xarray<float> tensor = xt::adapt(tmpValues, size, xt::no_ownership(),
                                       std::vector<size_t>({K, Y, X}));
  if (params.name == "conv2d") {
    tensor = xt::transpose(tensor, {1, 2, 0});
  }

  int address = 0;
  for (auto it = tensor.begin(); it != tensor.end(); ++it) {
    float val = *it;

    writeToReference(address, val, params.ACC_T_OUTPUT);
    address++;
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
  if (!params.WEIGHT && params.WEIGHT_OFFSET != -1 &&
      !files.weights_file.empty()) {
    loadWeights(params, memoryMap.weights, files.weights_file, useDataFile);
  }
  if (params.ATTENTION_MASK) {
    loadBias(params, memoryMap.bias, files.bias_file, useDataFile);
  }
  if ((params.RESIDUAL || params.RELU_GRAD || params.SOFTMAX_GRAD) &&
      !files.residual_file.empty()) {
    loadResiduals(params, memoryMap.residual, files.residual_file, useDataFile);
  }
  if (params.WEIGHT_SPLITTING) {
    SimplifiedParams gradParams = params;
    gradParams.WEIGHT_OFFSET = params.WEIGHT_RESIDUAL_OFFSET;
    loadWeights(gradParams, SRAM, files.weight_grad_file, useDataFile);
  }
}

void MemoryModel::loadModelParams(const SimplifiedParams& params,
                                  const Files& files,
                                  const MemoryMap& memoryMap,
                                  bool useDataFile) {
  // If weights is false, we don't load weights, but we load the second input
  // as weights from SRAM, see above
  if (params.WEIGHT) {
    loadWeights(params, memoryMap.weights, files.weights_file, useDataFile);
  }
  // if ATTENTION_MASK is true, we don't load bias as a model parameter, but
  // as an input, see above
  if (params.BIAS && !params.ATTENTION_MASK) {
    loadBias(params, memoryMap.bias, files.bias_file, useDataFile);
  } else if (params.ATTENTION_MASK) {
    if (files.bias_file != "") {
      loadBias(params, memoryMap.bias, files.bias_file, useDataFile);
    }
  }
  if (params.CROSS_ENTROPY_GRAD) {
    loadWeights(params, SRAM, files.weights_file, useDataFile);
  }
}
