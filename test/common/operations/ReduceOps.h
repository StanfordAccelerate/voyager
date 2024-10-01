#pragma once

#include "test/common/operations/Common.h"

// inline float exponent(const float &x) { return exp(x); }

template <typename T>
inline T exponent(const T &x) {
  typename T::AccumulationDatatype tmp =
      static_cast<typename T::AccumulationDatatype>(x);
  tmp.exponential();
  return static_cast<T>(tmp);
}

template <typename T>
inline T * sum(std::any input_tensor, const std::vector<int> shape, const std::vector<int> dims) {
    T * output = std::any_cast<T *>(input_tensor);
    std::vector<int> current(shape);
    
    // Iterate over the summation dims
    for (int dim : dims) {
	if (dim < 0) {
	    dim = current.size() + dim;
	}

	std::vector<int> updated(current);
	updated[dim] = 1;
	T * reduced = new T[get_size(updated)];

	for (int i = 0; i < updated[0]; ++i) {
	    for (int j = 0; j < updated[1]; ++j) {
		for (int k = 0; k < updated[2]; ++k) {
		    reduced[i * updated[1] * updated[2] + j * updated[2] + k] = 0.0;
		}
	    }
	}

	for (int i = 0; i < current[0]; ++i) {
	    for (int j = 0; j < current[1]; ++j) {
		for (int k = 0; k < current[2]; ++k) {
		    reduced[i * updated[1] * updated[2] * (dim != 0) + j * updated[2] * (dim != 1) + k * (dim != 2)] += output[i * current[1] * current[2] + j * current[2] + k];
		}
	    }
	}

	current = updated;
	delete[] output;
	output = reduced;
    }
    return output;
}

template <typename T>
inline T *softmax(std::any input_tensor, const std::vector<int> shape) {
  T *inputs = std::any_cast<T *>(input_tensor);

  int num_rows = 1;
  for (int i = 0; i < shape.size() - 1; i++) {
    num_rows *= shape[i];
  }
  int num_cols = shape[shape.size() - 1];

  T *outputs = new T[num_rows * num_cols];

  for (int i = 0; i < num_rows; i++) {
    int offset = i * num_cols;
    T max = -32768;
    for (int j = 0; j < num_cols; j++) {
      max = inputs[offset + j] > max ? inputs[offset + j] : max;
    }

    for (int j = 0; j < num_cols; j++) {
      T normalized = static_cast<T>(inputs[offset + j] - max);
      normalized.exponential();
      outputs[offset + j] = normalized;
    }

    // perform a tree addition
    T sum = 0.0;
    for (int j = 0; j < num_cols; j += OC_DIMENSION) {
      T buffer[OC_DIMENSION];
      for (int k = 0; k < OC_DIMENSION; k++) {
        buffer[k] = outputs[offset + j + k];
      }

      int depth = OC_DIMENSION;
      while (depth > 1) {
        for (int k = 0; k < depth; k += 2) {
          buffer[k / 2] = static_cast<T>(buffer[k] + buffer[k + 1]);
        }
        depth = depth / 2;
      }
      sum = static_cast<T>(sum + buffer[0]);
    }

    T divisor = sum;
    divisor.reciprocal();
    for (int j = 0; j < num_cols; j++) {
      outputs[offset + j] *= divisor;
    }
  }

  delete[] inputs;

  return outputs;
}
