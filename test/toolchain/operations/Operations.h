#pragma once

#include "src/AccelTypes.h"
#include "src/ArchitectureParams.h"
#include "src/Params.h"
#include "test/common/VerificationTypes.h"

void MapGradNormUnitTest(const SimplifiedParams &params,
                         const MemoryMap &memoryMap,
                         std::deque<BaseParams *> &mappedParams,
                         std::deque<AcceleratorMemoryMap> &opMemoryMaps);

void MapGradNormClipping(const SimplifiedParams &params,
                         const MemoryMap &memoryMap,
                         std::deque<BaseParams *> &mappedParams,
                         std::deque<AcceleratorMemoryMap> &opMemoryMaps,
                         int size);

void MapBiasGrad(const SimplifiedParams &params, const MemoryMap &memoryMap,
                 std::deque<BaseParams *> &mappedParams,
                 std::deque<AcceleratorMemoryMap> &opMemoryMaps);

void MapCrossEntropyGrad(const SimplifiedParams &params,
                         const MemoryMap &memoryMap,
                         std::deque<BaseParams *> &mappedParams,
                         std::deque<AcceleratorMemoryMap> &opMemoryMaps);

void MapSoftmax(const SimplifiedParams &params, const MemoryMap &memoryMap,
                std::deque<BaseParams *> &mappedParams,
                std::deque<AcceleratorMemoryMap> &opMemoryMaps);

void MapSoftmaxGrad(const SimplifiedParams &params, const MemoryMap &memoryMap,
                    std::deque<BaseParams *> &mappedParams,
                    std::deque<AcceleratorMemoryMap> &opMemoryMaps);

void MapFCGrad(const SimplifiedParams &params, const MemoryMap &memoryMap,
               std::deque<BaseParams *> &mappedParams,
               std::deque<AcceleratorMemoryMap> &opMemoryMaps);

void MapFCGradFusedWithNormClipping(
    const SimplifiedParams &params, const MemoryMap &memoryMap,
    std::deque<BaseParams *> &mappedParams,
    std::deque<AcceleratorMemoryMap> &opMemoryMaps);

void MapFC(const SimplifiedParams &params, const MemoryMap &memoryMap,
           std::deque<BaseParams *> &mappedParams,
           std::deque<AcceleratorMemoryMap> &opMemoryMaps);

void MapNoNorm(const SimplifiedParams &params, const MemoryMap &memoryMap,
               std::deque<BaseParams *> &mappedParams,
               std::deque<AcceleratorMemoryMap> &opMemoryMaps);

void MapNoNormGrad(const SimplifiedParams &params, const MemoryMap &memoryMap,
                   std::deque<BaseParams *> &mappedParams,
                   std::deque<AcceleratorMemoryMap> &opMemoryMaps);

void MapGenericErrorGrad(const SimplifiedParams &params,
                         const MemoryMap &memoryMap,
                         std::deque<BaseParams *> &mappedParams,
                         std::deque<AcceleratorMemoryMap> &opMemoryMaps);

void MapMatrixOp(const SimplifiedParams &params, const MemoryMap &memoryMap,
                 std::deque<BaseParams *> &mappedParams,
                 std::deque<AcceleratorMemoryMap> &opMemoryMaps);

void MapWeightUpdate(const SimplifiedParams &params, const MemoryMap &memoryMap,
                     std::deque<BaseParams *> &mappedParams,
                     std::deque<AcceleratorMemoryMap> &opMemoryMaps);

void MapNop(const SimplifiedParams &params, const MemoryMap &memoryMap,
            std::deque<BaseParams *> &mappedParams,
            std::deque<AcceleratorMemoryMap> &opMemoryMaps);
void MapLoRACombination(const SimplifiedParams &params,
                        const MemoryMap &memoryMap,
                        std::deque<BaseParams *> &mappedParams,
                        std::deque<AcceleratorMemoryMap> &opMemoryMaps);
void MapLoRAQuantize(const SimplifiedParams &params, const MemoryMap &memoryMap,
                     std::deque<BaseParams *> &mappedParams,
                     std::deque<AcceleratorMemoryMap> &opMemoryMaps);
