#include <locale>
#include <string>

#include "test/common/GoldModel.h"
#include "test/common/Harness.h"
#include "test/common/Utils.h"
#include "test/mobilebert/params.h"
#include "test/resnet/params.h"
#include "test/simple/params.h"

#include <stdexcept>
#define SRAM_MEMORY_SIZE (2*1024*1024)
#define RRAM_MEMORY_SIZE (12*1024*1024)

size_t load_layer_memory(const std::string& filename, INPUT_DATATYPE* memory)
{
	std::ifstream is(filename, std::ios::binary);

  if (!is.good())
    throw std::runtime_error("File \"" + filename + "\" does not exist");

  // Read data into buffer and copy data into memory
	std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(is), {});
  std::memcpy(memory, buffer.data(), buffer.size());
  return buffer.size();
}

int run_complete(const std::string& group, std::map<std::string, Params>* param_map)
{
  // Allocate accelerator memory
  INPUT_DATATYPE *sram_memory = new INPUT_DATATYPE[SRAM_MEMORY_SIZE];
  INPUT_DATATYPE *rram_memory = new INPUT_DATATYPE[RRAM_MEMORY_SIZE];
  if (sram_memory == nullptr || rram_memory == nullptr)
    throw std::runtime_error("Failed to allocate accelerator memory");
  std::memset(sram_memory, 0, SRAM_MEMORY_SIZE );
  std::memset(rram_memory, 0, RRAM_MEMORY_SIZE);

  // Load input
  std::string data_path = "data/" + group + '/';
  load_layer_memory(data_path + "input", sram_memory + (*(param_map->begin())).second.INPUT_OFFSET);

  // Load weights and biases
  for (const auto& param_pair: *param_map)
  {
    const std::string& name = param_pair.first;
    Params param = param_pair.second;
    // Load weights
    load_layer_memory(data_path + name + "_weight", rram_memory + param.WEIGHT_OFFSET);

    // Load biases
    // load_layer_memory(data_path + name + ".bias", main_memory + weight_offset + param.BIAS_OFFSET);
  }

  // Perform run with main memory
  for (const auto& param_pair: *param_map)
  {
    Params param = param_pair.second;
    run_op(param, (INPUT_DATATYPE*)sram_memory, (INPUT_DATATYPE*)rram_memory, true);
  }

  // Compare with pytorch data
	std::ifstream is(data_path + "input", std::ios::binary);
  if (!is.good())
    throw std::runtime_error("File does not exist");
	std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(is), {});
  INPUT_DATATYPE* pytorch_output = new INPUT_DATATYPE[buffer.size()];
  std::memcpy(pytorch_output, buffer.data(), buffer.size());

  int errors =
      compare_arrays(&sram_memory[(*(--param_map->end())).second.OUTPUT_OFFSET], pytorch_output, buffer.size());

  delete[] sram_memory;
  delete[] rram_memory;

  if (errors == 0) {
    std::cout << "Test passed!" << std::endl;
  } else {
    std::cout << "Test failed!" << std::endl;
  }

  return errors;
}

void validateMapping(Params params) {
  int x0 = params.loops[1][params.inputXLoopIndex[1]];
  int y0 = params.loops[1][params.inputYLoopIndex[1]];
  int c0 = params.loops[1][params.reductionLoopIndex[1]];
  int k0 = params.loops[1][params.weightLoopIndex[1]];
  int fx = params.loops[1][params.fxIndex];
  int fy = params.loops[1][params.fyIndex];
  int stride = params.STRIDE;

  // Input buffer
  int input_buffer_tile_size = (x0 * stride + fx - 1) * (y0 * stride + fy - 1);
  if (params.REPLICATION) {
    // don't check temporarily
    input_buffer_tile_size = 1;
  }
  if (input_buffer_tile_size > INPUT_BUFFER_SIZE) {
    std::cout << "[ERROR] Input buffer tile size violation." << std::endl;
    std::terminate();
  }

  // Weight buffer
  if (fx * fy * k0 > WEIGHT_BUFFER_SIZE) {
    std::cout << "[ERROR] Weight buffer tile size violation." << std::endl;
    std::terminate();
  }

  if (x0 * y0 * k0 > ACCUMULATION_BUFFER_SIZE) {
    std::cout << "[ERROR] Accumulation buffer tile size violation."
              << std::endl;
    std::terminate();
  }
}

int run_test(Params params) {
  validateMapping(params);

  INPUT_DATATYPE *sramMemory = new INPUT_DATATYPE[SRAM_MEMORY_SIZE];
  INPUT_DATATYPE *rramMemory = new INPUT_DATATYPE[RRAM_MEMORY_SIZE];

  if (sramMemory == nullptr || rramMemory == nullptr)
    throw std::runtime_error("Failed to allocate accelerator memory");

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

  std::cout << "Performing the following operation:" << std::endl;
  std::cout << "(" << X << "x" << Y << "x" << C << ")"
            << " * "
            << "(" << FX << "x" << FY << "x" << C << "x" << K << ")"
            << std::endl;

  // Create matrix A
  INPUT_DATATYPE *matrixA = new INPUT_DATATYPE[(STRIDE * X) * (STRIDE * Y) * C];

  if (params.REPLICATION) {
    for (int y = 0; y < STRIDE * Y; y++) {
      for (int x_o = 0; x_o < (STRIDE * X) / 4; x_o++) {
        for (int x_i = 0; x_i < 4; x_i++) {  // 4 packed together
          for (int c = 0; c < C; c++) {
            int x = x_o * 4 + x_i;
            int val = rand() % 128;

            int address = y * ((STRIDE * X) / 4) * 16 + x_o * 16 + x_i * 3 + c;
            sramMemory[params.INPUT_OFFSET + address] = val;

            address = y * (STRIDE * X) * C + x * C + c;
            matrixA[address] = val;
          }
        }
      }
    }
  } else {
    for (int y = 0; y < STRIDE * Y; y++) {
      for (int x = 0; x < STRIDE * X; x++) {
        for (int c = 0; c < C; c++) {
          int val = rand() % 128;

          int address = y * (STRIDE * X) * C + x * C + c;

          sramMemory[params.INPUT_OFFSET + address] = val;
          matrixA[address] = val;
        }
      }
    }
  }

  for (int i = 0; i < 512; i++) {
    if (i % 16 == 0) {
      std::cout << std::endl;
    }
    std::cout << sramMemory[params.INPUT_OFFSET + i] << " ";
  }

  INPUT_DATATYPE *matrixB = new INPUT_DATATYPE[FX * FY * C * K];
  for (int fy = 0; fy < FY; fy++) {
    for (int fx = 0; fx < FX; fx++) {
      for (int c = 0; c < C; c++) {
        for (int k = 0; k < K; k++) {
          int val = rand() % 128;

          int address = fy * FX * C * K + fx * C * K + c * K + k;
          rramMemory[params.WEIGHT_OFFSET + address] = val;
          matrixB[address] = val;
        }
      }
    }
  }

  INPUT_DATATYPE *biasMatrix = new INPUT_DATATYPE[K];

  if (params.BIAS) {
    for (int k = 0; k < K; k++) {
      int val = rand() % 128;
      rramMemory[params.BIAS_OFFSET + k] = val;
      biasMatrix[k] = val;
    }
  }

  INPUT_DATATYPE *residualMatrix = new INPUT_DATATYPE[X * Y * K];
  if (params.RESIDUAL) {
    for (int y = 0; y < Y; y++) {
      for (int x = 0; x < X; x++) {
        for (int k = 0; k < K; k++) {
          int val = rand() % 128;

          int address = y * X * K + x * K + k;
          sramMemory[params.RESIDUAL_OFFSET + address] = val;
          residualMatrix[address] = val;
        }
      }
    }
  }

  OUTPUT_DATATYPE *matrixC = new OUTPUT_DATATYPE[X * Y * K];

  if (params.MAXPOOL) {
    X = X / 2;
    Y = Y / 2;
  }

  if (params.AVGPOOL) {
    X = 1;
    Y = 1;
  }

  run_op(params, sramMemory, rramMemory, true);
  run_gold_op(params, matrixA, matrixB, matrixC, biasMatrix, residualMatrix);
  int errors =
      compare_arrays(&sramMemory[params.OUTPUT_OFFSET], matrixC, X * Y * K);

  delete[] matrixA;
  delete[] matrixB;
  delete[] matrixC;
  delete[] sramMemory;
  delete[] rramMemory;

  if (errors == 0) {
    std::cout << "Test passed!" << std::endl;
  } else {
    std::cout << "Test failed!" << std::endl;
  }

  return errors;
}

int sc_main(int argc, char *argv[]) {
  Params params;

  const char *groupName = std::getenv("GROUP");
  const char *testName = std::getenv("TEST");
  if (testName && groupName) {
    std::string group(groupName);
    std::string test(testName);

    std::cout << "Running: " << group << ", " << test << std::endl;

    std::map<std::string, Params> *mapPtr;

    if (group == "simple") {
      mapPtr = &simple;
    } else if (group == "mobilebert") {
      mapPtr = &mobilebert;
    } else if (group == "resnet") {
      mapPtr = &resnet;
    } else {
      std::cout << "Warning! Group " << group << " not found!" << std::endl;
    }

    // Run end to end if complete is specified
    if (test == "complete")
    {
      run_complete(group, mapPtr);
      return 0;
    }

    auto search = mapPtr->find(test);
    if (search != mapPtr->end()) {
      params = search->second;
    } else {
      std::cout << "Warning! Test " << test << " not found!" << std::endl;
    }

  } else {
    std::cout << "Warning! No group/test specified! Please set the environment "
                 "variables GROUP and TEST"
              << std::endl;
  }

  return run_test(params);
}
