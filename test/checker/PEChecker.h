#pragma once
#define NO_SYSC

#include <deque>
#include <tuple>

#include "src/ArchitectureParams.h"

template <typename T>
class Checker {
 public:
  Checker() { reference = new std::deque<T>(); }

  ~Checker() { delete reference; }

  void add_reference(T value) { reference->push_back(value); }

  void check_reference(int pe_num, T value) {
    if (reference->size() == 0) {
      // if weight is 0, then we don't need to check (happens during
      // replication)
      if (static_cast<float>(std::get<1>(value)) == 0) {
        return;
      }
      throw std::runtime_error("No reference values left to check!");
    }

    T ref = reference->front();

    if (ref != value) {
      if (static_cast<float>(std::get<1>(value)) == 0) {
        return;
      }
      std::cerr << "ERROR: PE " << pe_num << " input mismatch detected!"
                << std::endl;

      std::cerr << "  Expected inputs: "
                << "input=" << std::get<0>(ref) << ", "
                << "weight=" << std::get<1>(ref) << ", "
                << "psum=" << std::get<2>(ref) << std::endl;

      std::cerr << "  Actual inputs:   "
                << "input=" << std::get<0>(value) << ", "
                << "weight=" << std::get<1>(value) << ", "
                << "psum=" << std::get<2>(value) << std::endl;

      throw std::runtime_error("PE input mismatch detected");
    }
    reference->pop_front();
  }

 private:
  std::deque<T>* reference;
};

template <typename Input, typename Weight, typename Psum, int rows, int cols>
class PEChecker {
 public:
  PEChecker() {
    for (int i = 0; i < rows * cols; i++) {
      checkers[i] = new Checker<std::tuple<Input, Weight, Psum>>();
    }
  }

  ~PEChecker() {
    for (int i = 0; i < rows * cols; i++) {
      delete checkers[i];
    }
  }

  void add_reference(int pe_num, Input input, Weight weight, Psum accum) {
    checkers[pe_num]->add_reference(std::make_tuple(input, weight, accum));
  }

  void check_reference(int pe_num, Input input, Weight weight, Psum accum) {
    checkers[pe_num]->check_reference(pe_num,
                                      std::make_tuple(input, weight, accum));
  }

 private:
  Checker<std::tuple<Input, Weight, Psum>>* checkers[rows * cols];
};

extern PEChecker<SA_INPUT_TYPE, SA_WEIGHT_TYPE, ACCUM_DATATYPE, IC_DIMENSION,
                 OC_DIMENSION>
    pe_checker;
