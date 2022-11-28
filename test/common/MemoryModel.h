#pragma once

#include "test/common/VerificationTypes.h"

#define DIMENSION 16

void readFileAsDouble();

class MemoryModel {
 public:
  MemoryModel(bool);

  void loadModelActivations(const SimplifiedParams& params, const Files& files,
                            const MemoryMap& memoryMap, bool useDataFile);
  void loadModelParams(const SimplifiedParams& params, const Files& files,
                       const MemoryMap& memoryMap, bool useDataFile);
  void loadReferenceOutput(const SimplifiedParams& params, const Files& files);

 private:
  void loadInputs(const SimplifiedParams& params, const MemorySource& mem,
                  const std::string& filename, bool useDataFile);
  void loadResiduals(const SimplifiedParams& params, const MemorySource& mem,
                     const std::string& filename, bool useDataFile);
  void loadWeights(const SimplifiedParams& params, const MemorySource& mem,
                   const std::string& filename, bool useDataFile);
  void loadBias(const SimplifiedParams& params, const MemorySource& mem,
                const std::string& filename, bool useDataFile);

  virtual void writeToMemory(int address, double val, const MemorySource& mem,
                             bool doublePrecision) = 0;
  virtual void writeToReference(int address, double val) = 0;

  // special addressing is sometimes needed for DUT memory (ex. replication)
  bool isDut;
};