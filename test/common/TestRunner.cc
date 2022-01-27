#include <sys/wait.h>
#include <unistd.h>

#include <locale>
#include <stdexcept>
#include <string>

#include "test/common/GoldModel.h"
#include "test/common/Harness.h"
#include "test/common/Utils.h"
#include "test/mobilebert/params.h"
#include "test/resnet/params.h"
#include "test/simple/params.h"
#define SRAM_MEMORY_SIZE (2 * 1024 * 1024)
#define RRAM_MEMORY_SIZE (12 * 1024 * 1024)
#define USE_RRAM_WEIGHTS (true)

// NOTE: Binary data files are always supplied in [Y][X][C][K] ordering

size_t load_layer_memory(const std::string& filename, INPUT_DATATYPE* memory);
void run_op_contain(const Params& param, INPUT_DATATYPE* sram,
                    INPUT_DATATYPE* rram, bool weightFromRRAM);
int run_test_torch(const std::string& group, const std::string& test,
                   std::map<std::string, Params>* param_map);

size_t load_layer_bias(const std::string& filename, INPUT_DATATYPE* memory,
                       const Params& param) {
  // Load file into buffer
  std::ifstream is(filename, std::ios::binary);
  if (!is.good())
    throw std::runtime_error("File \"" + filename + "\" does not exist");
  std::vector<char> buffer(std::istreambuf_iterator<char>(is), {});
  char* bp = buffer.data();

  int X = param.loops[0][param.inputXLoopIndex[0]] *
          param.loops[1][param.inputXLoopIndex[1]];
  int Y = param.loops[0][param.inputYLoopIndex[0]] *
          param.loops[1][param.inputYLoopIndex[1]];
  int C = param.loops[1][param.reductionLoopIndex[1]] * DIMENSION;
  int K = param.loops[0][param.weightLoopIndex[0]] *
          param.loops[1][param.weightLoopIndex[1]] * DIMENSION;
  int FX = param.loops[1][param.fxIndex];
  int FY = param.loops[1][param.fyIndex];
  int STRIDE = param.STRIDE;

  if (param.REPLICATION) {
    FX = 7;
    C = 3;
  }

  if (param.BIAS) {
    for (int k = 0; k < K; k++) {
      int val = (int)*(bp++);
      // rramMemory[params.BIAS_OFFSET + k] = val;
      memory[k] = val;
    }
  }
  return 0;
}

/// @brief: Returns name of residual layer for layer_name
std::string get_resnet_residual(const std::string layer_name) {
  int i = 0;
  for (; i < resnet_order.size(); i++) {
    if (layer_name == resnet_order[i]) break;
  }

  assert(i >= 2);

  return resnet_order[i - 2];
}

size_t load_layer_weight(const std::string& filename, INPUT_DATATYPE* memory,
                         const Params& param) {
  // Load file into buffer
  std::ifstream is(filename, std::ios::binary);
  if (!is.good())
    throw std::runtime_error("File \"" + filename + "\" does not exist");
  std::vector<char> buffer(std::istreambuf_iterator<char>(is), {});
  char* bp = buffer.data();

  int X = param.loops[0][param.inputXLoopIndex[0]] *
          param.loops[1][param.inputXLoopIndex[1]];
  int Y = param.loops[0][param.inputYLoopIndex[0]] *
          param.loops[1][param.inputYLoopIndex[1]];
  int C = param.loops[1][param.reductionLoopIndex[1]] * DIMENSION;
  int K = param.loops[0][param.weightLoopIndex[0]] *
          param.loops[1][param.weightLoopIndex[1]] * DIMENSION;
  int FX = param.loops[1][param.fxIndex];
  int FY = param.loops[1][param.fyIndex];
  int STRIDE = param.STRIDE;

  if (param.REPLICATION) {
    FX = 7;
    C = 3;
  }

  std::cout << "possible segfault 1" << std::endl;
  printf("fx: %d, fy: %d, c: %d, k: %d\n", FX, FY, C, K);
  for (int fy = 0; fy < FY; fy++) {
    for (int fx = 0; fx < FX; fx++) {
      for (int c = 0; c < C; c++) {
        for (int k = 0; k < K; k++) {
          int address = fy * FX * C * K + fx * C * K + c * K + k;
          int val = (int)*(bp++);

          // rramMemory[params.WEIGHT_OFFSET + address] = val;
          memory[address] = val;
        }
      }
    }
  }

  std::cout << "possible segfault 2" << std::endl;
  return 0;
}

size_t load_layer_input(const std::string& filename, INPUT_DATATYPE* memory,
                        const Params& param) {
  // Load file into buffer
  std::ifstream is(filename, std::ios::binary);
  if (!is.good())
    throw std::runtime_error("File \"" + filename + "\" does not exist");
  std::vector<char> buffer(std::istreambuf_iterator<char>(is), {});
  char* bp = buffer.data();

  int X = param.loops[0][param.inputXLoopIndex[0]] *
          param.loops[1][param.inputXLoopIndex[1]];
  int Y = param.loops[0][param.inputYLoopIndex[0]] *
          param.loops[1][param.inputYLoopIndex[1]];
  int C = param.loops[1][param.reductionLoopIndex[1]] * DIMENSION;
  int K = param.loops[0][param.weightLoopIndex[0]] *
          param.loops[1][param.weightLoopIndex[1]] * DIMENSION;
  int FX = param.loops[1][param.fxIndex];
  int FY = param.loops[1][param.fyIndex];
  int STRIDE = param.STRIDE;

  if (param.REPLICATION) {
    FX = 7;
    C = 3;
  }
  std::cout << "possible segfault 1" << std::endl;
  printf("x: %d, y: %d, c: %d\n", STRIDE * X, STRIDE * Y, C);
  if (param.REPLICATION) {
    // size_t data_address = 0;
    for (int y = 0; y < STRIDE * Y; y++) {
      for (int x_o = 0; x_o < (STRIDE * X) / 4; x_o++) {
        for (int x_i = 0; x_i < 4; x_i++) {  // 4 packed together
          for (int c = 0; c < C; c++) {
            int x = x_o * 4 + x_i;

            int val = (int)*(bp++);

            int address = y * ((STRIDE * X) / 4) * 16 + x_o * 16 + x_i * 3 + c;
            // memory[params.INPUT_OFFSET + address] = val;
            address = y * (STRIDE * X) * C + x * C + c;
            memory[address] = val;
          }
        }
      }
    }
  } else {
    for (int y = 0; y < STRIDE * Y; y++) {
      for (int x = 0; x < STRIDE * X; x++) {
        for (int c = 0; c < C; c++) {
          int val = (int)*(bp++);

          int address = y * (STRIDE * X) * C + x * C + c;

          // memory[param.INPUT_OFFSET + address] = val;
          memory[address] = val;
        }
      }
    }
  }
  std::cout << "possible segfault 2" << std::endl;
  return 0;
}

/// @brief: Simulates param in child process and writes memory to parent
/// process.
void run_op_contain(const Params& param, INPUT_DATATYPE* sram,
                    INPUT_DATATYPE* rram, bool weightFromRRAM) {
  // Create pipe
  int pipefd[2];
  if (pipe(pipefd) < 0) throw std::runtime_error("Failed to create pipe");

  pid_t pid = fork();
  if (pid == 0) {
    // Close read end
    close(pipefd[0]);

    // Execute simulation
    run_op(param, sram, rram, weightFromRRAM);

    // Write sram then rram
    size_t sram_written = 0;
    while (sram_written < SRAM_MEMORY_SIZE) {
      sram_written += write(pipefd[1], sram + sram_written,
                            SRAM_MEMORY_SIZE - sram_written);
    }

    size_t rram_written = 0;
    while (rram_written < RRAM_MEMORY_SIZE) {
      rram_written += write(pipefd[1], rram + rram_written,
                            RRAM_MEMORY_SIZE - rram_written);
    }

    close(pipefd[1]);
    exit(0);
  } else {
    // Close write end
    close(pipefd[1]);

    // Read sram then rram
    size_t sram_read = 0;
    while (sram_read < SRAM_MEMORY_SIZE) {
      sram_read +=
          read(pipefd[0], sram + sram_read, SRAM_MEMORY_SIZE - sram_read);
    }

    size_t rram_read = 0;
    while (rram_read < RRAM_MEMORY_SIZE) {
      rram_read +=
          write(pipefd[0], rram + rram_read, RRAM_MEMORY_SIZE - rram_read);
    }

    close(pipefd[0]);
    int status;
    waitpid(-1, &status, 0);
  }
}

/// @brief: Loads input and comparison data from filename to memory.
size_t load_layer_memory(const std::string& filename, INPUT_DATATYPE* memory) {
  std::ifstream is(filename, std::ios::binary);
  if (!is.good())
    throw std::runtime_error("File \"" + filename + "\" does not exist");

  // Read data into buffer and copy data into memory
  std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(is), {});
  for (unsigned char c : buffer) {
    // Converts memory from char to INPUT_DATATYPE/OUTPUT_DATATYPE
    *memory = (int)c;
    ++memory;
  }

  return buffer.size();
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

/// @brief: Simulates param and compares against pytorch output
int run_test_torch(const std::string& group, const std::string& test,
                   std::map<std::string, Params>* param_map) {
  // Validate params
  Params param = (*param_map)[test];
  validateMapping(param);

  // Allocate memory
  INPUT_DATATYPE* sram_memory = new INPUT_DATATYPE[SRAM_MEMORY_SIZE];
  INPUT_DATATYPE* rram_memory = new INPUT_DATATYPE[RRAM_MEMORY_SIZE];
  if (sram_memory == nullptr || rram_memory == nullptr)
    throw std::runtime_error("Failed to allocate accelerator memory");
  std::memset(sram_memory, 0, SRAM_MEMORY_SIZE);
  std::memset(rram_memory, 0, RRAM_MEMORY_SIZE);

  // Load input, weights, and biases
  std::string data_path = "data/" + group + '/';

  // Run operation
  INPUT_DATATYPE* matrixA = new INPUT_DATATYPE[SRAM_MEMORY_SIZE];
  load_layer_input(data_path + test + "_input", matrixA, param);
  INPUT_DATATYPE* matrixB = new INPUT_DATATYPE[RRAM_MEMORY_SIZE];
  load_layer_weight(data_path + test + "_weight", matrixB, param);
  INPUT_DATATYPE* biasMatrix = new INPUT_DATATYPE[SRAM_MEMORY_SIZE];
  load_layer_bias(data_path + test + "_bias", biasMatrix, param);
  INPUT_DATATYPE* resMatrix = nullptr;

  if (param.RESIDUAL) {
    resMatrix = new INPUT_DATATYPE[SRAM_MEMORY_SIZE];
    load_layer_memory(data_path + get_resnet_residual(test) + "_comp",
                      resMatrix);
  }
  OUTPUT_DATATYPE* matrixC = new OUTPUT_DATATYPE[SRAM_MEMORY_SIZE];
  run_gold_op(param, matrixA, matrixB, matrixC, biasMatrix, resMatrix);

  // Compare with pytorch data
  INPUT_DATATYPE* pytorch_output = new INPUT_DATATYPE[RRAM_MEMORY_SIZE];
  if (pytorch_output == nullptr)
    throw std::runtime_error("Failed to allocate comparison memory");
  size_t output_size =
      load_layer_memory(data_path + test + "_comp", pytorch_output);

  int errors = compare_arrays(matrixC, pytorch_output, output_size);

  delete[] sram_memory;
  delete[] rram_memory;
  delete[] matrixA;
  delete[] matrixB;
  delete[] matrixC;
  delete[] biasMatrix;

  if (errors == 0) {
    std::cout << "Test passed!" << std::endl;
  } else {
    std::cout << "Test failed!" << std::endl;
  }

  return errors;
}

int run_test(Params params) {
  validateMapping(params);

  INPUT_DATATYPE* sramMemory = new INPUT_DATATYPE[SRAM_MEMORY_SIZE];
  INPUT_DATATYPE* rramMemory = new INPUT_DATATYPE[RRAM_MEMORY_SIZE];

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
  INPUT_DATATYPE* matrixA = new INPUT_DATATYPE[(STRIDE * X) * (STRIDE * Y) * C];

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

  INPUT_DATATYPE* matrixB = new INPUT_DATATYPE[FX * FY * C * K];
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

  INPUT_DATATYPE* biasMatrix = new INPUT_DATATYPE[K];

  if (params.BIAS) {
    for (int k = 0; k < K; k++) {
      int val = rand() % 128;
      rramMemory[params.BIAS_OFFSET + k] = val;
      biasMatrix[k] = val;
    }
  }

  INPUT_DATATYPE* residualMatrix = new INPUT_DATATYPE[X * Y * K];
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

  OUTPUT_DATATYPE* matrixC = new OUTPUT_DATATYPE[X * Y * K];

  if (params.MAXPOOL) {
    X = X / 2;
    Y = Y / 2;
  }

  if (params.AVGPOOL) {
    X = 1;
    Y = 1;
  }

  run_op(params, sramMemory, rramMemory, USE_RRAM_WEIGHTS);
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

int sc_main(int argc, char* argv[]) {
  Params params;

  const char* groupName = std::getenv("GROUP");
  const char* testName = std::getenv("TEST");

  if (!(testName && groupName)) {
    std::cout << "Warning! No group/test specified! Please set the environment "
                 "variables GROUP and TEST"
              << std::endl;
    return -1;
  }

  std::string group(groupName);
  std::string test(testName);

  std::cout << "Running: " << group << ": " << test << std::endl;

  std::map<std::string, Params>* mapPtr;

  if (group == "simple") {
    mapPtr = &simple;
  } else if (group == "mobilebert") {
    mapPtr = &mobilebert;
  } else if (group == "resnet") {
    mapPtr = &resnet;
  } else {
    throw std::runtime_error("Group: " + group + " not found");
  }

  // Run end to end if complete is specified
  // if (test == "complete")
  // {
  //   run_complete(group, mapPtr);
  //   return 0;
  // }

  auto search = mapPtr->find(test);
  if (search != mapPtr->end()) {
    params = search->second;
  } else {
    throw std::runtime_error("Test: " + test + " not found");
  }

  // return comp_gold ? run_test(params) : run_test_torch(group, test, mapPtr);
  return run_test_torch(group, test, mapPtr);
}
