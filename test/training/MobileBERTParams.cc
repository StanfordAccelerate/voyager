#include <string>

#include "test/common/Utils.h"
#include "test/common/VerificationTypes.h"

#define LORA

void getMobileBERTParams(std::string layerName, std::string taskName,
                         SimplifiedParams &params, MemoryMap &memoryMap) {
#include "test/mobilebert/mobilebert_tiny2/inference.h"
#include "test/mobilebert/mobilebert_tiny2/backprop.h"
#include "test/mobilebert/mobilebert_tiny2/gradient.h"
#include "test/mobilebert/mobilebert_tiny2/weight.h"

  if (taskName == "inference") {
    params = inferenceParams.at(layerName);
    memoryMap = {SRAM, params.WEIGHT ? RRAM : SRAM, RRAM, SRAM, SRAM};

#ifdef LORA
    if (layerName == "attention_self_query_layer" ||
        layerName == "attention_self_value_layer") {
      memoryMap.weights = SRAM;
    }
#endif

    // if operation involves attention mask, use SRAM as bias
    if (inferenceTestFiles.at(layerName).bias_file.find(
            "mobilebert_attention_mask") != std::string::npos) {
      memoryMap.bias = SRAM;
      params.ATTENTION_MASK = true;
    }
  } else if (taskName == "backward") {
    params = backpropParams.at(layerName);
    memoryMap = {SRAM, params.WEIGHT ? RRAM : SRAM, RRAM, SRAM, SRAM};
  } else if (taskName == "gradient") {
    params = gradientParams.at(layerName);
    memoryMap = {SRAM, SRAM, RRAM, SRAM, SRAM};
  } else if (taskName == "weight") {
    params = weightParams.at(layerName);
    memoryMap = {SRAM, SRAM, RRAM, SRAM, SRAM};
  } else {
    assert(false);
  }
}