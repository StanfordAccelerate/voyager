#include <string>

#include "test/common/Utils.h"
#include "test/common/VerificationTypes.h"

void getMobileBERTParams(std::string layerName, std::string taskName,
                         SimplifiedParams &params, MemoryMap &memoryMap) {
#include "test/mobilebert/mobilebert_tiny2/backprop.h"
#include "test/mobilebert/mobilebert_tiny2/inference.h"
#include "test/mobilebert/mobilebert_tiny2/gradient.h"

  if (taskName == "inference") {
    params = inferenceParams.at(layerName);
    memoryMap = {SRAM, params.WEIGHT ? RRAM : SRAM, RRAM, SRAM, SRAM};
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
  } else {
    assert(false);
  }
}