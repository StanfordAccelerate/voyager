#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
// clang-format off
#include "VectorOps.h"
// clang-format on
#include "Broadcaster.h"
#include "MaxpoolUnit.h"
#include "OutputAddressGenerator.h"
#include "ParamsDeserializer.h"
#include "VectorFetch.h"

template <typename IO_DTYPE, typename VEC_DTYPE, typename MU_OUTPUT_DTYPE,
          int WIDTH>
SC_MODULE(VectorOpUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<VectorInstructions> CCS_INIT_S1(vectorOpUnitInstructions);
  Connections::In<VectorInstructions> CCS_INIT_S1(
      accumulationOpUnitInstructions);
  Connections::In<VectorInstructions> CCS_INIT_S1(reductionOpUnitInstructions);

  Connections::In<Pack1D<MU_OUTPUT_DTYPE, WIDTH> > CCS_INIT_S1(
      systolicArrayOutput);
  Connections::In<Pack1D<VEC_DTYPE, WIDTH> > CCS_INIT_S1(vectorFetch0Output);
  Connections::In<Pack1D<VEC_DTYPE, WIDTH> > CCS_INIT_S1(vectorFetch1Output);
  Connections::In<Pack1D<VEC_DTYPE, WIDTH> > CCS_INIT_S1(vectorFetch2Output);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch3AddressRequest);
  Connections::In<IO_DTYPE> CCS_INIT_S1(vectorFetch3DataResponse);

  Connections::Out<Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> >
      CCS_INIT_S1(vectorOpUnitOutput);

  Connections::Combinational<
      Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> >
      CCS_INIT_S1(accumulationOpInput);
  Connections::Combinational<
      Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> >
      CCS_INIT_S1(accumulationOpOutput);

  Connections::Combinational<
      Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> >
      CCS_INIT_S1(reductionOpInput);
  Connections::Combinational<
      Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> >
      CCS_INIT_S1(reductionOpOutputOp0Src0);
  Connections::Combinational<
      Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> >
      CCS_INIT_S1(reductionOpOutputOp0Src1);
  Connections::Combinational<
      Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> >
      CCS_INIT_S1(reductionOpOutputOp3Src1);

  Broadcaster<Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> >
      CCS_INIT_S1(broadcastReduction0);
  Connections::Combinational<ac_int<16, false> > broadcastReduction0Count;
  Connections::Combinational<
      Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> >
      CCS_INIT_S1(broadcastReductionOpOutputOp0Src0);

  Broadcaster<Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> >
      CCS_INIT_S1(broadcastReduction1);
  Connections::Combinational<ac_int<16, false> > broadcastReduction1Count;
  Connections::Combinational<
      Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> >
      CCS_INIT_S1(broadcastReductionOpOutputOp0Src1);

  Broadcaster<Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> >
      CCS_INIT_S1(broadcastReduction2);
  Connections::Combinational<ac_int<16, false> > broadcastReduction2Count;
  Connections::Combinational<
      Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> >
      CCS_INIT_S1(broadcastReductionOpOutputOp3Src1);

  SC_CTOR(VectorOpUnit) {
    // systolicArrayOutput.enable_local_rand_stall();

    broadcastReduction0.clk(clk);
    broadcastReduction0.rstn(rstn);
    broadcastReduction0.dataIn(reductionOpOutputOp0Src0);
    broadcastReduction0.count(broadcastReduction0Count);
    broadcastReduction0.dataOut(broadcastReductionOpOutputOp0Src0);

    broadcastReduction1.clk(clk);
    broadcastReduction1.rstn(rstn);
    broadcastReduction1.dataIn(reductionOpOutputOp0Src1);
    broadcastReduction1.count(broadcastReduction1Count);
    broadcastReduction1.dataOut(broadcastReductionOpOutputOp0Src1);

    broadcastReduction2.clk(clk);
    broadcastReduction2.rstn(rstn);
    broadcastReduction2.dataIn(reductionOpOutputOp3Src1);
    broadcastReduction2.count(broadcastReduction2Count);
    broadcastReduction2.dataOut(broadcastReductionOpOutputOp3Src1);

    SC_THREAD(vectorOpRun);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(accumulationOpRun);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(reductionOpRun);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void vectorOpRun() {
    vectorOpUnitInstructions.Reset();
    systolicArrayOutput.Reset();
    vectorFetch0Output.Reset();
    vectorFetch1Output.Reset();
    vectorFetch2Output.Reset();
    vectorOpUnitOutput.Reset();
    accumulationOpInput.ResetWrite();
    accumulationOpOutput.ResetRead();
    reductionOpInput.ResetWrite();
    broadcastReductionOpOutputOp0Src0.ResetRead();
    broadcastReductionOpOutputOp0Src1.ResetRead();
    broadcastReductionOpOutputOp3Src1.ResetRead();
    vectorFetch3AddressRequest.Reset();
    vectorFetch3DataResponse.Reset();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode bubble
    while (true) {
      VectorInstructions inst = vectorOpUnitInstructions.Pop();

      Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> op0;
      Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> op1;
      Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> op2;

      Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> res0;
      Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> res1;
      Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> res2;
      Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> res3;
      Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> res4;
      Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> res5;

      /*
       * Stage 0: add, sub, mult
       */
      // Read from interfaces
      Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> op0Src0;
      if (inst.vInput == VectorInstructions::readFromSystolicArray) {
        Pack1D<MU_OUTPUT_DTYPE, WIDTH> tmp = systolicArrayOutput.Pop();

        if constexpr (!MU_OUTPUT_DTYPE::is_floating_point &&
                      VEC_DTYPE::is_floating_point) {
          vdequantize<VEC_DTYPE, MU_OUTPUT_DTYPE, WIDTH>(tmp, op0Src0,
                                                         inst.vDequantizeScale);
        } else if constexpr (std::is_same_v<MU_OUTPUT_DTYPE, VEC_DTYPE>) {
          // with MX types, we might need to perform a dequantize operation
          if (inst.vDequantize) {
            vdequantize<VEC_DTYPE, MU_OUTPUT_DTYPE, WIDTH>(
                tmp, op0Src0, inst.vDequantizeScale);
          } else {
          UNROLL_0:
            for (int i = 0; i < WIDTH; i++) {
              op0Src0[i] = tmp[i];
            }
          }
        } else {
        UNROLL_1:
          for (int i = 0; i < WIDTH; i++) {
            op0Src0[i] = tmp[i];
          }
        }
      } else if (inst.vInput == VectorInstructions::readFromVectorFetch) {
        Pack1D<VEC_DTYPE, WIDTH> tmp = vectorFetch0Output.Pop();

#pragma hls_unroll yes
        for (int i = 0; i < WIDTH; i++) {
          op0Src0[i] = tmp[i];
        }
      } else if (inst.vInput == VectorInstructions::readFromAccumulation) {
        op0Src0 = accumulationOpOutput.Pop();
      } else if (inst.vInput == VectorInstructions::readFromReduce) {
        op0Src0 = broadcastReductionOpOutputOp0Src0.Pop();
      }

      Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> op0Src1;
      if (inst.vOp0Src1 == VectorInstructions::readInterface) {
        Pack1D<VEC_DTYPE, WIDTH> tmp = vectorFetch1Output.Pop();
#pragma hls_unroll yes
        for (int i = 0; i < WIDTH; i++) {
          op0Src1[i] = tmp[i];
        }
      } else if (inst.vOp0Src1 == VectorInstructions::op0immediate) {
        typename VEC_DTYPE::AccumulationDatatype immediate;
        immediate.setbits(inst.immediate0);

#pragma hls_unroll yes
        for (int i = 0; i < WIDTH; i++) {
          op0Src1[i] = immediate;
        }
      } else if (inst.vOp0Src1 == VectorInstructions::readFromReduce) {
        op0Src1 = broadcastReductionOpOutputOp0Src1.Pop();
      }

      // DLOG("vector unit input: " << op0Src0);

      if (inst.vOp0 == VectorInstructions::vadd ||
          inst.vOp0 == VectorInstructions::vsub) {
        if (inst.vOp0 == VectorInstructions::vsub) {
#pragma hls_unroll yes
          for (int i = 0; i < WIDTH; i++) {
            op0Src1[i].negate();
          }
        }
        vadd<typename VEC_DTYPE::AccumulationDatatype, WIDTH>(op0Src0, op0Src1,
                                                              res0);
        DLOG(op0Src0 << std::endl
                     << " + " << std::endl
                     << op0Src1 << std::endl
                     << " = " << std::endl
                     << res0);
      } else if (inst.vOp0 == VectorInstructions::vmult) {
        vmult<typename VEC_DTYPE::AccumulationDatatype, WIDTH>(op0Src0, op0Src1,
                                                               res0);
        DLOG(op0Src0 << std::endl
                     << " * " << std::endl
                     << op0Src1 << std::endl
                     << " = " << std::endl
                     << res0);
      } else {
        res0 = op0Src0;
      }

      // DLOG("res0: " << res0);
      /*
       * Stage 1: exp
       */
      if (inst.vOp1 == VectorInstructions::vexp) {
        vexp<typename VEC_DTYPE::AccumulationDatatype, WIDTH>(res0, res1);
      } else if (inst.vOp1 == VectorInstructions::vscaleexp) {
        ac_int<8, true> scaleVal;
        if (inst.vOp1Src1 == VectorInstructions::op1immediate) {
          scaleVal = inst.immediate0;
        }
        vscaleexp<typename VEC_DTYPE::AccumulationDatatype, WIDTH>(
            res0, scaleVal, res1);
      } else {
        res1 = res0;
      }

      // DLOG("res1: " << res1);

      /*
       * Stage 2: reduction
       */
      // FIXME: delete this stage (moved reduction moved to end of pipeline)
      res2 = res1;

      // DLOG("res2: " << res2);

      /*
       * Stage 3: add, div
       */
      Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> op3Src0;
      op3Src0 = res2;

      Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> op3Src1;
      if (inst.vOp3Src1 == VectorInstructions::readReduceInterface) {
        op3Src1 = broadcastReductionOpOutputOp3Src1.Pop();
      } else if (inst.vOp3Src1 == VectorInstructions::readNormalInterface) {
        Pack1D<VEC_DTYPE, WIDTH> tmp = vectorFetch2Output.Pop();

#pragma hls_unroll yes
        for (int i = 0; i < WIDTH; i++) {
          op3Src1[i] = tmp[i];
        }
      } else if (inst.vOp3Src1 == VectorInstructions::op3immediate) {
        typename VEC_DTYPE::AccumulationDatatype immediate;
        immediate.setbits(inst.immediate1);

#pragma hls_unroll yes
        for (int i = 0; i < WIDTH; i++) {
          op3Src1[i] = immediate;
        }
      }

      if (inst.vOp3 == VectorInstructions::vadd) {
        vadd<typename VEC_DTYPE::AccumulationDatatype, WIDTH>(op3Src0, op3Src1,
                                                              res3);
        DLOG(op3Src0 << std::endl
                     << " + " << std::endl
                     << op3Src1 << std::endl
                     << " = " << std::endl
                     << res3);
      } else if (inst.vOp3 == VectorInstructions::vmult ||
                 inst.vOp3 == VectorInstructions::vsquare) {
        if (inst.vOp3 == VectorInstructions::vsquare) {
          op3Src1 = op3Src0;
        }
        vmult<typename VEC_DTYPE::AccumulationDatatype, WIDTH>(op3Src0, op3Src1,
                                                               res3);
        DLOG(op3Src0 << std::endl
                     << " * " << std::endl
                     << op3Src1 << std::endl
                     << " = " << std::endl
                     << res3);
      } else if (inst.vOp3 == VectorInstructions::vscaleexp) {
        ac_int<8, true> scaleVal;
        scaleVal = inst.immediate1;
        vscaleexp<typename VEC_DTYPE::AccumulationDatatype, WIDTH>(
            op3Src0, scaleVal, res3);
      } else {
        res3 = op3Src0;
      }

      // DLOG("res3: " << res3);

      /*
       * Stage 4: relu and value mapping
       */
      if (inst.vOp4 == VectorInstructions::vrelu ||
          inst.vOp4 == VectorInstructions::vrelumask) {
        bool useMask = inst.vOp4 == VectorInstructions::vrelumask;
        vrelu<typename VEC_DTYPE::AccumulationDatatype, WIDTH>(res3, op0Src1,
                                                               useMask, res4);
      } else if (inst.vOp4 == VectorInstructions::vmap) {
        for (int i = 0; i < WIDTH; i++) {
          DataTypes::bfloat16 value = res3[i];
          uint offset = value.bits_rep().to_uint();

          MemoryRequest memRequest = {inst.vmapOffset + offset * 2, 2};
          vectorFetch3AddressRequest.Push(memRequest);

          ac_int<16, false> bits = 0;
          for (int j = 0; j < 2; j++) {
            IO_DTYPE response = vectorFetch3DataResponse.Pop();
            ac_int<16, false> bits_rep = response.bits_rep();
            bits = bits | (bits_rep << (8 * j));
          }

          DataTypes::bfloat16 mappedValue;
          mappedValue.setbits(bits);

          res4[i] = mappedValue;
        }
      } else {
        res4 = res3;
      }

      if (inst.vAccumulatePush) {
        accumulationOpInput.Push(res4);
      }
      if (inst.vOp2 == VectorInstructions::toReduce) {
        reductionOpInput.Push(res4);
      }

      // DLOG("res4: " << res4);

      if (inst.vDest == VectorInstructions::vWriteOut) {
        vectorOpUnitOutput.Push(res4);
      }
    }
  }

  void accumulationOpRun() {
    accumulationOpInput.ResetRead();
    accumulationOpOutput.ResetWrite();
    accumulationOpUnitInstructions.Reset();

    wait();

    while (true) {
      VectorInstructions inst = accumulationOpUnitInstructions.Pop();

      Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> accum;

#pragma hls_unroll yes
      for (int i = 0; i < WIDTH; i++) {
        accum[i].setZero();
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int count = 0; count < inst.rCount; count++) {
        DLOG("accumulation " << count << " / " << inst.rCount);
        Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> op =
            accumulationOpInput.Pop();

        if (inst.rOp == VectorInstructions::radd) {
#pragma hls_unroll yes
          for (int i = 0; i < WIDTH; i++) {
            // accum[i] = accum[i] + op[i];
            accum[i] += op[i];
          }
        } else if (inst.rOp == VectorInstructions::rmax) {
#pragma hls_unroll yes
          for (int i = 0; i < WIDTH; i++) {
            accum[i] = accum[i] < op[i] ? op[i] : accum[i];
          }
        }
      }

      DLOG("accumulation finished: " << accum);
      accumulationOpOutput.Push(accum);
    }
  }

  void reductionOpRun() {
    reductionOpUnitInstructions.Reset();
    reductionOpInput.ResetRead();
    reductionOpOutputOp0Src0.ResetWrite();
    reductionOpOutputOp0Src1.ResetWrite();
    reductionOpOutputOp3Src1.ResetWrite();
    broadcastReduction0Count.ResetWrite();
    broadcastReduction1Count.ResetWrite();
    broadcastReduction2Count.ResetWrite();

    wait();

    while (true) {
      VectorInstructions inst = reductionOpUnitInstructions.Pop();

      Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> res;

      int iterationCount = inst.rDuplicate ? 1 : WIDTH;

      typename VEC_DTYPE::AccumulationDatatype scalarResult;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int index = 0; index < iterationCount; index++) {
        typename VEC_DTYPE::AccumulationDatatype prevResult;

        if (inst.rOp == VectorInstructions::radd) {
          for (int i = 0; i < inst.rCount; i++) {
            Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> op =
                reductionOpInput.Pop();
            typename VEC_DTYPE::AccumulationDatatype result = treeadd(op);
            // TreeOps<typename VEC_DTYPE::AccumulationDatatype,
            // WIDTH>().treeadd(
            //     op);
            DLOG("reduction: " << op << " = " << result);
            if (i != 0) {
              // result = (typename VEC_DTYPE::AccumulationDatatype)(result +
              // prevResult);
              result += prevResult;
            }

            prevResult = result;
          }
        } else if (inst.rOp == VectorInstructions::rmax) {
          for (int i = 0; i < inst.rCount; i++) {
            Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> op =
                reductionOpInput.Pop();
            typename VEC_DTYPE::AccumulationDatatype result = treemax(op);
            // TreeOps<typename VEC_DTYPE::AccumulationDatatype,
            // WIDTH>().treemax(
            //     op);
            if (i != 0) {
              result = result < prevResult ? prevResult : result;
            }

            prevResult = result;
          }
        } else if (inst.rOp == VectorInstructions::rmxscale) {
          typedef ac_int<VEC_DTYPE::AccumulationDatatype::exponent_width, false>
              exp_type_t;

          exp_type_t prevExpResult;
          for (int i = 0; i < inst.rCount; i++) {
            Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> op =
                reductionOpInput.Pop();

            Pack1D<exp_type_t, WIDTH> exponents;

#pragma hls_unroll yes
            for (int j = 0; j < WIDTH; j++) {
              exponents[j] = op[j].unbiased_exponent();
            }

            exp_type_t result = treemax(exponents);

            if (i != 0) {
              result = result < prevExpResult ? prevExpResult : result;
            }

            prevExpResult = result;
          }

          ac_int<VEC_DTYPE::AccumulationDatatype::exponent_width, true>
              scaledExp =
                  prevExpResult -
                  VEC_DTYPE::AccumulationDatatype::ac_float_rep::exp_bias -
                  (IO_DTYPE::width - 2);

          prevResult.setbits(scaledExp);
        }

        if (!inst.rDuplicate) {
          res[index] = prevResult;
        } else {
          scalarResult = prevResult;
        }

        DLOG("Reduction " << index << "/" << iterationCount << " : "
                          << prevResult << std::endl
                          << res);
      }

      if (inst.rSqrt) {
        scalarResult = scalarResult.sqrt();
      }

      if (inst.rReciprocal) {
        scalarResult.reciprocal();
      }

      if (inst.rMax1) {
        scalarResult = scalarResult.max1();
      }

      if (inst.rDuplicate) {
        // Duplicate the scalar result into a vector
#pragma hls_unroll yes
        for (int i = 0; i < WIDTH; i++) {
          res[i] = scalarResult;
        }
      }

      if (inst.rDest != 0) {
        if (inst.rDest == VectorInstructions::toVectorOp0Src0) {
          ac_int<16, false> broadcastCount = 1;
          if (inst.rBroadcast) {
            broadcastCount = inst.immediate0;
          }
          broadcastReduction0Count.Push(broadcastCount);
          reductionOpOutputOp0Src0.Push(res);
        } else if (inst.rDest == VectorInstructions::toVectorOp0Src1) {
          ac_int<16, false> broadcastCount = 1;
          if (inst.rBroadcast) {
            broadcastCount = inst.immediate0;
          }
          broadcastReduction1Count.Push(broadcastCount);
          reductionOpOutputOp0Src1.Push(res);
        } else if (inst.rDest == VectorInstructions::toVectorOp3Src1) {
          ac_int<16, false> broadcastCount = 1;
          if (inst.rBroadcast) {
            broadcastCount = inst.immediate0;
          }
          broadcastReduction2Count.Push(broadcastCount);
          reductionOpOutputOp3Src1.Push(res);
        }
      }
    }
  }
};

template <typename IO_DTYPE, typename VEC_DTYPE, typename MU_OUTPUT_DTYPE,
          typename MX_DTYPE, int WIDTH>
SC_MODULE(VectorUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<int> CCS_INIT_S1(serialParamsIn);

  Connections::In<Pack1D<MU_OUTPUT_DTYPE, WIDTH> > CCS_INIT_S1(
      systolicArrayOutput);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch0AddressRequest);
  Connections::In<Pack1D<IO_DTYPE, WIDTH> > CCS_INIT_S1(
      vectorFetch0DataResponse);
  Connections::Combinational<Pack1D<VEC_DTYPE, WIDTH> > CCS_INIT_S1(
      vectorFetch0DataResponseBroadcasted);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch1AddressRequest);
  Connections::In<Pack1D<IO_DTYPE, WIDTH> > CCS_INIT_S1(
      vectorFetch1DataResponse);
  Connections::Combinational<Pack1D<VEC_DTYPE, WIDTH> > CCS_INIT_S1(
      vectorFetch1DataResponseConverted);
  Connections::Combinational<MX_DTYPE> CCS_INIT_S1(vectorFetch1Scale);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch2AddressRequest);
  Connections::In<Pack1D<IO_DTYPE, WIDTH> > CCS_INIT_S1(
      vectorFetch2DataResponse);
  Connections::Combinational<Pack1D<VEC_DTYPE, WIDTH> > CCS_INIT_S1(
      vectorFetch2DataResponseConverted);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch3AddressRequest);
  Connections::In<IO_DTYPE> CCS_INIT_S1(vectorFetch3DataResponse);

  Connections::Out<ac_int<64, false> > CCS_INIT_S1(vectorOutputAddress);
  Connections::Out<Pack1D<IO_DTYPE, WIDTH> > CCS_INIT_S1(finalVectorOutput);
  Connections::Combinational<
      Pack1D<typename VEC_DTYPE::AccumulationDatatype, WIDTH> >
      CCS_INIT_S1(vectorOpUnitOutput);

  Connections::SyncOut CCS_INIT_S1(start);
  Connections::SyncOut CCS_INIT_S1(done);

  VectorFetchUnit<IO_DTYPE, VEC_DTYPE, MX_DTYPE, WIDTH> CCS_INIT_S1(
      vectorFetch);
  Connections::Combinational<VectorParams> CCS_INIT_S1(vectorFetchParams);

  VectorOpUnit<IO_DTYPE, VEC_DTYPE, MU_OUTPUT_DTYPE, WIDTH> CCS_INIT_S1(
      vectorOpUnit);

  MaxpoolUnit<VEC_DTYPE, IO_DTYPE, MX_DTYPE, WIDTH> CCS_INIT_S1(maxpoolUnit);
  Connections::Combinational<VectorParams> CCS_INIT_S1(maxpoolUnitParams);

  OutputAddressGenerator<IO_DTYPE, WIDTH> CCS_INIT_S1(outputAddressGenerator);
  Connections::Combinational<VectorParams> CCS_INIT_S1(outputAddressGenParams);

  Connections::Combinational<VectorInstructions> CCS_INIT_S1(
      vectorOpInstructions);
  Connections::Combinational<VectorInstructions> CCS_INIT_S1(
      accumulationOpInstructions);
  Connections::Combinational<VectorInstructions> CCS_INIT_S1(
      reduceOpInstructions);

  VectorParamsDeserializer CCS_INIT_S1(paramsDeserializer);
  Connections::Combinational<VectorParams> CCS_INIT_S1(vectorParamsIn);
  Connections::Combinational<VectorInstructionConfig> CCS_INIT_S1(
      vectorInstructionsIn);

  SC_CTOR(VectorUnit) {
    paramsDeserializer.clk(clk);
    paramsDeserializer.rstn(rstn);
    paramsDeserializer.serialParamsIn(serialParamsIn);
    paramsDeserializer.vectorParamsOut(vectorParamsIn);
    paramsDeserializer.vectorInstructionsOut(vectorInstructionsIn);

    vectorFetch.clk(clk);
    vectorFetch.rstn(rstn);
    vectorFetch.paramsIn(vectorFetchParams);
    vectorFetch.vectorFetch0AddressRequest(vectorFetch0AddressRequest);
    vectorFetch.vectorFetch0DataResponse(vectorFetch0DataResponse);
    vectorFetch.vectorFetch0DataResponseBroadcasted(
        vectorFetch0DataResponseBroadcasted);
    vectorFetch.vectorFetch1AddressRequest(vectorFetch1AddressRequest);
    vectorFetch.vectorFetch1DataResponse(vectorFetch1DataResponse);
    vectorFetch.vectorFetch1DataResponseConverted(
        vectorFetch1DataResponseConverted);
    vectorFetch.vectorFetch2AddressRequest(vectorFetch2AddressRequest);
    vectorFetch.vectorFetch2DataResponse(vectorFetch2DataResponse);
    vectorFetch.vectorFetch2DataResponseConverted(
        vectorFetch2DataResponseConverted);
    vectorFetch.vectorFetch1Scale(vectorFetch1Scale);

    vectorOpUnit.clk(clk);
    vectorOpUnit.rstn(rstn);
    vectorOpUnit.vectorOpUnitInstructions(vectorOpInstructions);
    vectorOpUnit.accumulationOpUnitInstructions(accumulationOpInstructions);
    vectorOpUnit.reductionOpUnitInstructions(reduceOpInstructions);
    vectorOpUnit.systolicArrayOutput(systolicArrayOutput);
    vectorOpUnit.vectorFetch0Output(vectorFetch0DataResponseBroadcasted);
    vectorOpUnit.vectorFetch1Output(vectorFetch1DataResponseConverted);
    vectorOpUnit.vectorFetch2Output(vectorFetch2DataResponseConverted);
    vectorOpUnit.vectorOpUnitOutput(vectorOpUnitOutput);
    vectorOpUnit.vectorFetch3AddressRequest(vectorFetch3AddressRequest);
    vectorOpUnit.vectorFetch3DataResponse(vectorFetch3DataResponse);

    maxpoolUnit.clk(clk);
    maxpoolUnit.rstn(rstn);
    maxpoolUnit.paramsIn(maxpoolUnitParams);
    maxpoolUnit.tensorIn(vectorOpUnitOutput);
    maxpoolUnit.tensorOut(finalVectorOutput);
    maxpoolUnit.doneSignal(done);
    maxpoolUnit.mxScaleIn(vectorFetch1Scale);

    outputAddressGenerator.clk(clk);
    outputAddressGenerator.rstn(rstn);
    outputAddressGenerator.paramsIn(outputAddressGenParams);
    outputAddressGenerator.vectorOutputAddress(vectorOutputAddress);

    SC_THREAD(read_params);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(instructionSender);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void read_params() {
    vectorParamsIn.ResetRead();
    vectorFetchParams.ResetWrite();
    maxpoolUnitParams.ResetWrite();
    outputAddressGenParams.ResetWrite();

    wait();

    while (true) {
      VectorParams params = vectorParamsIn.Pop();

      vectorFetchParams.Push(params);
      maxpoolUnitParams.Push(params);
      outputAddressGenParams.Push(params);
    }
  }

  void instructionSender() {
    vectorOpInstructions.ResetWrite();
    reduceOpInstructions.ResetWrite();
    accumulationOpInstructions.ResetWrite();
    vectorInstructionsIn.ResetRead();
    start.Reset();

    wait();

    while (true) {
      VectorInstructionConfig instConfig = vectorInstructionsIn.Pop();

      start.SyncPush();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int loop = 0; loop < instConfig.instLoopCount; loop++) {
        for (int instAddress = 0; instAddress < 8; instAddress++) {
          VectorInstructions inst = instConfig.inst[instAddress];

          for (int instRepeatCount = 0;
               instRepeatCount < instConfig.instCount[instAddress];
               instRepeatCount++) {
            if (inst.instType == VectorInstructions::vector) {
              vectorOpInstructions.Push(inst);
            } else if (inst.instType == VectorInstructions::accumulation) {
              accumulationOpInstructions.Push(inst);
            } else {
              reduceOpInstructions.Push(inst);
            }
          }

          if (instAddress >= instConfig.instLen - 1) {
            break;
          }
        }
      }
    }
  }
};
