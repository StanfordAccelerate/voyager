#pragma once

#include "test/common/operations/Common.h"

template <typename T>
inline T* pad_tensor(std::any input_ptr, const std::vector<int>& input_shape,
                     const std::vector<int>& pad_list) {
  for (int i = 0; i < pad_list.size(); i++) {
    spdlog::debug("Padding tensor with amount: {}\n", pad_list[i]);
  }

  T* inputs = std::any_cast<T*>(input_ptr);

  std::vector<int> output_shape(input_shape);
  int pad_dim = pad_list.size() / 2;
  std::vector<int> output_start(pad_dim, 0);
  std::vector<int> output_end(pad_dim, 0);
  assert(pad_dim <= output_shape.size());

  for (int i = 0; i < pad_dim; i++) {
    output_start[i] = pad_list[i * 2];
    output_end[i] = pad_list[i * 2 + 1];
    output_shape[output_shape.size() - i - 1] +=
        output_start[i] + output_end[i];
  }

  T* outputs = new T[get_size(output_shape)];

  for (int i = 0; i < get_size(output_shape); i++) {
    std::vector<int> indices(output_shape.size(), 0);
    int index = i;
    for (int j = output_shape.size() - 1; j >= 0; --j) {
      indices[j] = index % output_shape[j];
      index /= output_shape[j];
    }

    bool out_bound = false;
    for (int j = 0; j < pad_dim; j++) {
      if (indices[output_shape.size() - j - 1] < output_start[j] ||
          indices[output_shape.size() - j - 1] >=
              output_shape[output_shape.size() - j - 1] - output_end[j]) {
        out_bound = true;
        break;
      }
    }

    if (out_bound) {
      outputs[i] = T::zero();  // Fill padded area with zero
    } else {
      std::vector<int> input_indices(indices);
      for (int j = 0; j < pad_dim; j++) {
        input_indices[output_shape.size() - j - 1] -= output_start[j];
      }
      int input_index = get_flat_index(input_indices, input_shape);
      outputs[i] = inputs[input_index];
    }
  }

  delete[] inputs;

  return outputs;
}

template <typename T>
inline T* pad_tensor(std::any input_ptr, const voyager::PrimOp& op,
                     const ScalarEnv& env) {
  const auto input = resolve(op, "input", env);
  const auto input_shape = get_shape(input);
  const auto pad_list = arg_ints(op, "pad", env);
  std::vector<int> pad_list_vector(pad_list.begin(), pad_list.end());
  // const auto pad_amount = pad_list[pad_list.size() - 1];

  return pad_tensor<T>(input_ptr, input_shape, pad_list_vector);
}

template <typename T>
inline T* permute(std::any input_ptr, const std::vector<int>& shape,
                  const std::vector<int> dims) {
  T* inputs = std::any_cast<T*>(input_ptr);

  std::vector<int> permuted_shape(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    permuted_shape[i] = shape[dims[i]];
  }

  const int size = get_size(shape);
  T* outputs = new T[size];

  for (int i = 0; i < size; ++i) {
    std::vector<int> indices = get_indices(i, shape);

    std::vector<int> permuted_indices(dims.size());
    for (size_t j = 0; j < dims.size(); ++j) {
      permuted_indices[j] = indices[dims[j]];
    }

    int permuted_index = get_flat_index(permuted_indices, permuted_shape);
    outputs[permuted_index] = inputs[i];
  }

  delete[] inputs;

  return outputs;
}

template <typename T>
inline T* permute(std::any input_ptr, const voyager::PrimOp& op,
                  const ScalarEnv& env) {
  if (strip_namespace(op.target()) == "permute") {
    const auto input = resolve(op, "input", env);
    const auto input_shape = get_shape(input);

    const auto dims = arg_ints(op, "dims", env);
    std::vector<int> dims_vector(dims.begin(), dims.end());

    return permute<T>(input_ptr, input_shape, dims_vector);
  } else if (strip_namespace(op.target()) == "transpose") {
    const auto input = resolve(op, "input", env);
    const auto input_shape = get_shape(input);

    const int dim0 = arg_int(op, "dim0", env);
    const int dim1 = arg_int(op, "dim1", env);

    std::vector<int> dims_vector;
    for (int i = 0; i < input_shape.size(); ++i) {
      dims_vector.push_back(i);
    }

    std::swap(dims_vector[dim0], dims_vector[dim1]);

    return permute<T>(input_ptr, input_shape, dims_vector);
  }

  return std::any_cast<T*>(input_ptr);
}

template <typename T>
inline T* expand(std::any input_ptr, const std::vector<int>& input_shape,
                 const std::vector<int>& sizes) {
  T* inputs = std::any_cast<T*>(input_ptr);

  // Align the input shape to the expanded rank from the right, the way
  // broadcasting does.
  std::vector<int> aligned(sizes.size(), 1);
  const int offset = sizes.size() - input_shape.size();
  for (size_t i = 0; i < input_shape.size(); ++i) {
    aligned[offset + i] = input_shape[i];
  }

  // A -1 keeps the input's dimension.
  std::vector<int> output_shape(sizes);
  for (size_t i = 0; i < output_shape.size(); ++i) {
    if (output_shape[i] == -1) output_shape[i] = aligned[i];
  }

  const int size = get_size(output_shape);
  T* outputs = new T[size];

  for (int i = 0; i < size; ++i) {
    std::vector<int> indices = get_indices(i, output_shape);
    for (size_t d = 0; d < indices.size(); ++d) {
      if (aligned[d] == 1) indices[d] = 0;
    }
    outputs[i] = inputs[get_flat_index(indices, aligned)];
  }

  delete[] inputs;

  return outputs;
}

template <typename T>
inline T* expand(std::any input_ptr, const voyager::PrimOp& op,
                 const ScalarEnv& env) {
  const auto input = resolve(op, "input", env);
  return expand<T>(input_ptr, get_shape(input), arg_ints(op, "size", env));
}

template <typename T>
inline T* transpose(std::any input_ptr, const std::vector<int>& shape, int dim0,
                    int dim1) {
  const int ndim = shape.size();
  dim0 = dim0 < 0 ? dim0 + ndim : dim0;
  dim1 = dim1 < 0 ? dim1 + ndim : dim1;

  std::vector<int> dims;
  for (int i = 0; i < ndim; ++i) {
    dims.push_back(i);
  }

  std::swap(dims[dim0], dims[dim1]);

  return permute<T>(input_ptr, shape, dims);
}

template <typename T>
inline T* transpose(std::any input_ptr, const voyager::PrimOp& op,
                    const ScalarEnv& env) {
  if (strip_namespace(op.target()) != "transpose") {
    return std::any_cast<T*>(input_ptr);
  }

  const auto input = resolve(op, "input", env);
  const auto input_shape = get_shape(input);
  const int dim0 = arg_int(op, "dim0", env);
  const int dim1 = arg_int(op, "dim1", env);

  return transpose<T>(input_ptr, input_shape, dim0, dim1);
}

template <typename T>
inline T* reshape_if_needed(std::any input_ptr, const voyager::PrimOp& op,
                            const ScalarEnv& env) {
  if (strip_namespace(op.target()) == "transpose") {
    return transpose<T>(input_ptr, op);
  } else if (strip_namespace(op.target()) == "permute") {
    return permute<T>(input_ptr, op);
  } else {
    return std::any_cast<T*>(input_ptr);
  }
}
