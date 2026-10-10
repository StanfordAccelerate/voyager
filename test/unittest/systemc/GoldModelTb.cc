#include <iostream>

#include "test/common/GoldModel.h"

namespace {

void set_tensor(voyager::TensorBox* box, const std::string& name,
                const std::vector<int>& shape) {
  box->set_node(name);
  box->set_dtype("bfloat16");
  box->mutable_memory()->set_level(voyager::MEMORY_LEVEL_SCRATCHPAD);
  for (int extent : shape) box->add_shape(extent);
}

void check_gemv(bool with_bias) {
  using Vector = VECTOR_DATATYPE;
  constexpr int channels = 2 * REDUCER_WIDTH;
  constexpr int outputs = 2;
  voyager::Operation operation;
  operation.set_name("gemv");
  auto* prim = operation.mutable_prim();
  prim->set_name("gemv");
  prim->set_op("call_function");
  prim->set_target("aten::linear");
  set_tensor(
      (*prim->mutable_kwargs())["input"].mutable_tensor_box()->mutable_box(),
      "input", {1, channels});
  set_tensor(
      (*prim->mutable_kwargs())["weight"].mutable_tensor_box()->mutable_box(),
      "weight", {outputs, channels});
  if (with_bias) {
    set_tensor(
        (*prim->mutable_kwargs())["bias"].mutable_tensor_box()->mutable_box(),
        "bias", {outputs});
  }
  auto* output = operation.add_outputs();
  output->set_name("gemv");
  set_tensor(output->mutable_tensor_box(), "gemv", {1, outputs});

  std::shared_ptr<Vector[]> input(new Vector[channels]);
  std::shared_ptr<Vector[]> weight(new Vector[outputs * channels]);
  std::shared_ptr<Vector[]> bias(new Vector[outputs]);
  for (int c = 0; c < channels; ++c) {
    input[c] = 1.0f;
    weight[c] = 1.0f;
    weight[channels + c] = -2.0f;
  }
  bias[0] = 3.0f;
  bias[1] = -5.0f;
  std::map<std::string, std::any> operands{{"input", input},
                                           {"weight", weight}};
  if (with_bias) operands["bias"] = bias;
  const auto result = run_gold_model(operation, ScalarEnv{}, operands);
  const auto values = std::any_cast<std::shared_ptr<Vector[]>>(result.at(0));
  const float expected[] = {float(channels + (with_bias ? 3 : 0)),
                            float(-2 * channels - (with_bias ? 5 : 0))};
  for (int k = 0; k < outputs; ++k) {
    if (static_cast<float>(values[k]) != expected[k])
      throw std::runtime_error("BF16 GEMV gold result differs");
  }
}

}  // namespace

extern "C" int sc_main(int, char**) {
  // INT8 builds have integer matrix accumulators and BF16 vector arithmetic.
  // Reduction tiles can omit bias even when the complete layer has one.
  check_gemv(false);
  check_gemv(true);
  std::cout << "PASS BF16 GEMV gold with and without bias\n";
  return 0;
}
