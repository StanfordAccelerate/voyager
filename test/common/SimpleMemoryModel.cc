#include "test/common/SimpleMemoryModel.h"

#define NO_SYSC

// clang-format off
#include "src/PositTypes.h"
// clang-format on
#include "src/ArchitectureParams.h"
#include "test/common/UniversalPosit.h"
#include "test/common/VerificationTypes.h"

template <class T>
SimpleMemoryModel<T>::SimpleMemoryModel(bool isDut) : MemoryModel(isDut) {
  try {
    sram = new T[SRAM_MEMORY_SIZE];
    rram = new T[RRAM_MEMORY_SIZE];
    reference = new T[SRAM_MEMORY_SIZE];
  } catch (const std::bad_alloc&) {
    throw std::runtime_error("ERROR: Failed to allocate simulation memory");
  }
}

template <class T>
SimpleMemoryModel<T>::~SimpleMemoryModel() {
  delete[] sram;
  delete[] rram;
  delete[] reference;
}

template <class T>
void SimpleMemoryModel<T>::writeToReference(int address, double val) {
  reference[address] = val;
}

template <>
void SimpleMemoryModel<INPUT_DATATYPE>::writeToMemory(int address, double val,
                                                      const MemorySource& mem,
                                                      bool doublePrecision) {
  INPUT_DATATYPE* memArray = mem == SRAM ? sram : rram;

  if (!doublePrecision) {
    memArray[address] = val;
  } else {
    ACCUM_DATATYPE p16 = val;
    int bits = p16.bits;
    memArray[address].setbits(bits & 0xFF);
    memArray[address + 1].setbits((bits >> 8) & 0xFF);
  }
}

template <>
void SimpleMemoryModel<float>::writeToMemory(int address, double val,
                                             const MemorySource& mem,
                                             bool doublePrecision) {
  float* memArray = mem == SRAM ? sram : rram;

  memArray[address] = val;
}

#ifndef NO_UNIVERSAL
template <>
void SimpleMemoryModel<UniversalPosit>::writeToMemory(int address, double val,
                                                      const MemorySource& mem,
                                                      bool doublePrecision) {
  UniversalPosit* memArray = mem == SRAM ? sram : rram;

  if (!doublePrecision) {
    memArray[address] = val;
  } else {
    UniversalPositAccum p16 = val;
    int bits = p16.encoding();
    memArray[address].setbits(bits & 0xFF);
    memArray[address + 1].setbits((bits >> 8) & 0xFF);
  }
}
#endif

// explicit instantiations
template class SimpleMemoryModel<INPUT_DATATYPE>;
template class SimpleMemoryModel<float>;
#ifndef NO_UNIVERSAL
template class SimpleMemoryModel<UniversalPosit>;
#endif
