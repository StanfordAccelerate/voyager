#pragma once

#include "test/common/operations/Common.h"

template <typename T>
struct ReductionRegistry {
  // Define the signature: takes two Ts, returns one T
  using Reducer = std::function<T(const T&, const T&)>;

  static const std::unordered_map<std::string, Reducer>& get_ops() {
    static const std::unordered_map<std::string, Reducer> ops = {
        {"amax", [](const T& a, const T& b) { return std::max(a, b); }},
        {"sum", [](const T& a, const T& b) { return a + b; }},
        {"prod", [](const T& a, const T& b) { return a * b; }},
        {"min", [](const T& a, const T& b) { return std::min(a, b); }}};
    return ops;
  }

  // Helper to check if an op exists
  static bool is_supported(const std::string& op) {
    return get_ops().find(op) != get_ops().end();
  }
};

template <typename T>
T reduce_op(const T& a, const T& b, const std::string& op) {
  auto& ops = ReductionRegistry<T>::get_ops();
  auto it = ops.find(op);

  if (it != ops.end()) {
    return it->second(a, b);
  }

  throw std::runtime_error("Reduction operation " + op + " not implemented");
}

template <typename T>
T* reduce(std::any input_tensor, std::vector<int> shape, int dim,
          const std::string& op) {
  T* inputs = std::any_cast<T*>(input_tensor);

  if (dim < 0) {
    dim += shape.size();
  }

  const int input_size = get_size(shape);
  const int output_size = input_size / shape[dim];

  std::vector<int> output_shape(shape);
  output_shape[dim] = 1;

  T* outputs = new T[output_size];

  for (int i = 0; i < output_size; i++) {
    auto indices = get_indices(i, output_shape);

    indices[dim] = 0;
    int first_index = get_flat_index(indices, shape);
    float acc = inputs[first_index];

    for (int j = 1; j < shape[dim]; j++) {
      indices[dim] = j;
      int index = get_flat_index(indices, shape);
      acc = reduce_op(acc, static_cast<float>(inputs[index]), op);
    }

    outputs[i] = static_cast<T>(acc);
  }

  return outputs;
}

template <typename T>
T* reduce(std::map<std::string, std::any>& kwargs,
          const codegen::OpOverload op) {
  const auto input = op.kwargs().at("input").tensor();
  std::any input_ptr = kwargs[input.node()];
  const auto input_shape = get_shape(input);
  const auto dim = op.kwargs().at("dim").int_list().values();
  assert(dim.size() == 1);  // Only support reduction over one dimension for now
  return reduce<T>(input_ptr, input_shape, dim[0], op.target());
}
