#pragma once

#include "test/common/MemoryModel.h"

template <class T>
class SimpleMemoryModel : public MemoryModel {
 public:
  SimpleMemoryModel(bool);
  ~SimpleMemoryModel();

  T* sram;
  T* rram;
  T* reference;

 private:
  void writeToMemory(int address, double val, const MemorySource& mem,
                     bool doublePrecision) override;
  void writeToReference(int address, double val, bool doublePrecision) override;
};
