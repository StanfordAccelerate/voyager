#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "Broadcaster.h"
#include "MaxpoolUnit.h"
#include "OutputAddressGenerator.h"
#include "ParamsDeserializer.h"
#include "VectorFetch.h"
#include "VectorOps.h"

template <typename IDTYPE, typename ACC_DTYPE, int WIDTH>
SC_MODULE(VectorOpUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<VectorInstructions> CCS_INIT_S1(vectorOpUnitInstructions);
  Connections::In<VectorInstructions> CCS_INIT_S1(
      accumulationOpUnitInstructions);
  Connections::In<VectorInstructions> CCS_INIT_S1(reductionOpUnitInstructions);

  Connections::In<Pack1D<ACC_DTYPE, WIDTH> > CCS_INIT_S1(systolicArrayOutput);
  Connections::In<Pack1D<ACC_DTYPE, WIDTH> > CCS_INIT_S1(vectorFetch0Output);
  Connections::In<Pack1D<ACC_DTYPE, WIDTH> > CCS_INIT_S1(vectorFetch1Output);
  Connections::In<Pack1D<ACC_DTYPE, WIDTH> > CCS_INIT_S1(vectorFetch2Output);

  Connections::Out<Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> >
      CCS_INIT_S1(vectorOpUnitOutput);

  Connections::Combinational<
      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> >
      CCS_INIT_S1(accumulationOpInput);
  Connections::Combinational<
      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> >
      CCS_INIT_S1(accumulationOpOutput);

  Connections::Combinational<
      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> >
      CCS_INIT_S1(reductionOpInput);
  Connections::Combinational<
      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> >
      CCS_INIT_S1(reductionOpOutputOp0Src0);
  Connections::Combinational<
      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> >
      CCS_INIT_S1(reductionOpOutputOp0Src1);
  Connections::Combinational<
      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> >
      CCS_INIT_S1(reductionOpOutputOp3Src1);

  Broadcaster<Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> > CCS_INIT_S1(
      broadcastReduction0);
  Connections::Combinational<ac_int<16, false> > broadcastReduction0Count;
  Connections::Combinational<
      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> >
      CCS_INIT_S1(broadcastReductionOpOutputOp0Src0);

  Broadcaster<Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> > CCS_INIT_S1(
      broadcastReduction1);
  Connections::Combinational<ac_int<16, false> > broadcastReduction1Count;
  Connections::Combinational<
      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> >
      CCS_INIT_S1(broadcastReductionOpOutputOp0Src1);

  Broadcaster<Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> > CCS_INIT_S1(
      broadcastReduction2);
  Connections::Combinational<ac_int<16, false> > broadcastReduction2Count;
  Connections::Combinational<
      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> >
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

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode bubble
    while (true) {
      VectorInstructions inst = vectorOpUnitInstructions.Pop();

      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> op0;
      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> op1;
      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> op2;

      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> res0;
      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> res1;
      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> res2;
      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> res3;
      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> res4;

      /*
       * Stage 0: add, sub, mult
       */
      // Read from interfaces
      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> op0Src0;
      if (inst.vInput == VectorInstructions::readFromSystolicArray) {
        Pack1D<ACC_DTYPE, WIDTH> tmp = systolicArrayOutput.Pop();
#pragma hls_unroll yes
        for (int i = 0; i < WIDTH; i++) {
          op0Src0[i] = tmp[i];
        }
      } else if (inst.vInput == VectorInstructions::readFromVectorFetch) {
        Pack1D<ACC_DTYPE, WIDTH> tmp = vectorFetch0Output.Pop();
#pragma hls_unroll yes
        for (int i = 0; i < WIDTH; i++) {
          op0Src0[i] = tmp[i];
        }
      } else if (inst.vInput == VectorInstructions::readFromAccumulation) {
        op0Src0 = accumulationOpOutput.Pop();
      } else if (inst.vInput == VectorInstructions::readFromReduce) {
        op0Src0 = broadcastReductionOpOutputOp0Src0.Pop();
      }

      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> op0Src1;
      if (inst.vOp0Src1 == VectorInstructions::readInterface) {
        Pack1D<ACC_DTYPE, WIDTH> tmp = vectorFetch1Output.Pop();
#pragma hls_unroll yes
        for (int i = 0; i < WIDTH; i++) {
          op0Src1[i] = tmp[i];
        }
      } else if (inst.vOp0Src1 == VectorInstructions::op0immediate0 ||
                 inst.vOp0Src1 == VectorInstructions::op0immediate1) {
        IDTYPE immediate;
        if (inst.vOp0Src1 == VectorInstructions::op0immediate0) {
          immediate.bits = inst.immediate0;
        } else {
          immediate.bits = inst.immediate1;
        }

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
        vadd<typename ACC_DTYPE::DecomposedPosit, WIDTH>(op0Src0, op0Src1,
                                                         res0);
      } else if (inst.vOp0 == VectorInstructions::vmult) {
        vmult<typename ACC_DTYPE::DecomposedPosit, WIDTH>(op0Src0, op0Src1,
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
        vexp<typename ACC_DTYPE::DecomposedPosit, WIDTH>(res0, res1);
      } else if (inst.vOp1 == VectorInstructions::vscaleexp) {
        ac_int<8, true> scaleVal;
        if (inst.vOp1Src1 == VectorInstructions::op1immediate0) {
          scaleVal = inst.immediate0;
        } else {
          scaleVal = inst.immediate1;
        }
        vscaleexp<typename ACC_DTYPE::DecomposedPosit, WIDTH>(res0, scaleVal,
                                                              res1);
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
      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> op3Src0;
      op3Src0 = res2;

      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> op3Src1;
      if (inst.vOp3Src1 == VectorInstructions::readReduceInterface) {
        op3Src1 = broadcastReductionOpOutputOp3Src1.Pop();
      } else if (inst.vOp3Src1 == VectorInstructions::readNormalInterface) {
        Pack1D<ACC_DTYPE, WIDTH> tmp = vectorFetch2Output.Pop();

#pragma hls_unroll yes
        for (int i = 0; i < WIDTH; i++) {
          op3Src1[i] = tmp[i];
        }
      } else if (inst.vOp3Src1 == VectorInstructions::op3immediate0 ||
                 inst.vOp3Src1 == VectorInstructions::op3immediate1) {
        IDTYPE immediate;
        if (inst.vOp3Src1 == VectorInstructions::op3immediate0) {
          immediate.bits = inst.immediate0;
        } else {
          immediate.bits = inst.immediate1;
        }

#pragma hls_unroll yes
        for (int i = 0; i < WIDTH; i++) {
          op3Src1[i] = immediate;
        }
      }

      if (inst.vOp3 == VectorInstructions::vadd) {
        vadd<typename ACC_DTYPE::DecomposedPosit, WIDTH>(op3Src0, op3Src1,
                                                         res3);

        // DLOG(op3Src0 << std::endl
        //              << " + " << std::endl
        //              << op3Src1 << std::endl
        //              << " = " << std::endl
        //              << res3);
      } else if (inst.vOp3 == VectorInstructions::vmult ||
                 inst.vOp3 == VectorInstructions::vdiv ||
                 inst.vOp3 == VectorInstructions::vsquare) {
        vmultdiv<typename ACC_DTYPE::DecomposedPosit, WIDTH>(
            op3Src0, op3Src1, res3, inst.vOp3 == VectorInstructions::vdiv,
            inst.vOp3 == VectorInstructions::vsquare);
      } else if (inst.vOp3 == VectorInstructions::vscaleexp) {
        ac_int<8, true> scaleVal;
        if (inst.vOp3Src1 == VectorInstructions::op3immediate0) {
          scaleVal = inst.immediate0;
        } else {
          scaleVal = inst.immediate1;
        }
        vscaleexp<typename ACC_DTYPE::DecomposedPosit, WIDTH>(op3Src0, scaleVal,
                                                              res3);
      } else {
        res3 = op3Src0;
      }

      // DLOG("res3: " << res3);

      /*
       * Stage 4: relu
       */
      if (inst.vOp4 == VectorInstructions::vrelu ||
          inst.vOp4 == VectorInstructions::vrelumask) {
        bool useMask = inst.vOp4 == VectorInstructions::vrelumask;
        vrelu<typename ACC_DTYPE::DecomposedPosit, WIDTH>(res3, op0Src1,
                                                          useMask, res4);
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

      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> accum;

#pragma hls_unroll yes
      for (int i = 0; i < WIDTH; i++) {
        accum[i].setZero();
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int count = 0; count < inst.rCount; count++) {
        DLOG("accumulation " << count << " / " << inst.rCount);
        Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> op =
            accumulationOpInput.Pop();

#pragma hls_unroll yes
        for (int i = 0; i < WIDTH; i++) {
          accum[i] += op[i];
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

      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> res;

      int iterationCount = inst.rDuplicate ? 1 : WIDTH;

      typename ACC_DTYPE::DecomposedPosit scalarResult;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int index = 0; index < iterationCount; index++) {
        typename ACC_DTYPE::DecomposedPosit prevResult;

        if (inst.rOp == VectorInstructions::radd) {
          for (int i = 0; i < inst.rCount; i++) {
            Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> op =
                reductionOpInput.Pop();
            typename ACC_DTYPE::DecomposedPosit result = treeadd16(op);
            // TreeOps<typename ACC_DTYPE::DecomposedPosit, WIDTH>().treeadd(
            //     op);
            DLOG("reduction: " << op << " = " << result);
            if (i != 0) {
              // result = (typename ACC_DTYPE::DecomposedPosit)(result +
              // prevResult);
              result += prevResult;
            }

            prevResult = result;
          }
        } else if (inst.rOp == VectorInstructions::rmax) {
          for (int i = 0; i < inst.rCount; i++) {
            Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> op =
                reductionOpInput.Pop();
            typename ACC_DTYPE::DecomposedPosit result = treemax16(op);
            // TreeOps<typename ACC_DTYPE::DecomposedPosit, WIDTH>().treemax(
            //     op);
            if (i != 0) {
              result = result < prevResult ? prevResult : result;
            }

            prevResult = result;
          }
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

      if (inst.rInvSqrt) {
        scalarResult = scalarResult.inv_sqrt();
      }

      if (inst.rMax1) {
        typename ACC_DTYPE::DecomposedPosit one;
        one._zero = false;
        one.fraction = 0;
        one.scale = 0;
        one.sign = false;

        if (scalarResult > one) {
          scalarResult = one;
        }
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
            broadcastCount.set_slc(0, inst.immediate0);
            broadcastCount.set_slc(8, inst.immediate1);
          }
          broadcastReduction0Count.Push(broadcastCount);
          reductionOpOutputOp0Src0.Push(res);
        } else if (inst.rDest == VectorInstructions::toVectorOp0Src1) {
          ac_int<16, false> broadcastCount = 1;
          if (inst.rBroadcast) {
            broadcastCount.set_slc(0, inst.immediate0);
            broadcastCount.set_slc(8, inst.immediate1);
          }

          broadcastReduction1Count.Push(broadcastCount);
          reductionOpOutputOp0Src1.Push(res);
        } else if (inst.rDest == VectorInstructions::toVectorOp3Src1) {
          ac_int<16, false> broadcastCount = 1;
          if (inst.rBroadcast) {
            broadcastCount.set_slc(0, inst.immediate0);
            broadcastCount.set_slc(8, inst.immediate1);
          }
          broadcastReduction2Count.Push(broadcastCount);
          reductionOpOutputOp3Src1.Push(res);
        }
      }
    }
  }
};

template <typename ODTYPE, typename ACC_DTYPE, int WIDTH>
SC_MODULE(VectorUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<int> CCS_INIT_S1(serialParamsIn);

  Connections::In<Pack1D<ACC_DTYPE, WIDTH> > CCS_INIT_S1(systolicArrayOutput);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch0AddressRequest);
  Connections::In<Pack1D<ODTYPE, WIDTH> > CCS_INIT_S1(vectorFetch0DataResponse);
  Connections::Combinational<Pack1D<ACC_DTYPE, WIDTH> > CCS_INIT_S1(
      vectorFetch0DataResponseBroadcasted);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch1AddressRequest);
  Connections::In<Pack1D<ODTYPE, WIDTH> > CCS_INIT_S1(vectorFetch1DataResponse);
  Connections::Combinational<Pack1D<ACC_DTYPE, WIDTH> > CCS_INIT_S1(
      vectorFetch1DataResponseConverted);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch2AddressRequest);
  Connections::In<Pack1D<ODTYPE, WIDTH> > CCS_INIT_S1(vectorFetch2DataResponse);
  Connections::Combinational<Pack1D<ACC_DTYPE, WIDTH> > CCS_INIT_S1(
      vectorFetch2DataResponseReplicated);

  Connections::Out<int> CCS_INIT_S1(vectorOutputAddress);
  Connections::Out<Pack1D<ODTYPE, WIDTH> > CCS_INIT_S1(finalVectorOutput);
  Connections::Combinational<
      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> >
      CCS_INIT_S1(vectorOpUnitOutput);

  Connections::SyncOut CCS_INIT_S1(start);
  Connections::SyncOut CCS_INIT_S1(done);

  VectorFetchUnit<ODTYPE, ACC_DTYPE, WIDTH> CCS_INIT_S1(vectorFetch);
  Connections::Combinational<VectorParams> CCS_INIT_S1(vectorFetchParams);

  VectorOpUnit<ODTYPE, ACC_DTYPE, WIDTH> CCS_INIT_S1(vectorOpUnit);

  MaxpoolUnit<ACC_DTYPE, ODTYPE, WIDTH> CCS_INIT_S1(maxpoolUnit);
  Connections::Combinational<VectorParams> CCS_INIT_S1(maxpoolUnitParams);

  OutputAddressGenerator<WIDTH> CCS_INIT_S1(outputAddressGenerator);
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
    vectorFetch.vectorFetch2DataResponseReplicated(
        vectorFetch2DataResponseReplicated);

    vectorOpUnit.clk(clk);
    vectorOpUnit.rstn(rstn);
    vectorOpUnit.vectorOpUnitInstructions(vectorOpInstructions);
    vectorOpUnit.accumulationOpUnitInstructions(accumulationOpInstructions);
    vectorOpUnit.reductionOpUnitInstructions(reduceOpInstructions);
    vectorOpUnit.systolicArrayOutput(systolicArrayOutput);
    vectorOpUnit.vectorFetch0Output(vectorFetch0DataResponseBroadcasted);
    vectorOpUnit.vectorFetch1Output(vectorFetch1DataResponseConverted);
    vectorOpUnit.vectorFetch2Output(vectorFetch2DataResponseReplicated);
    vectorOpUnit.vectorOpUnitOutput(vectorOpUnitOutput);

    maxpoolUnit.clk(clk);
    maxpoolUnit.rstn(rstn);
    maxpoolUnit.paramsIn(maxpoolUnitParams);
    maxpoolUnit.tensorIn(vectorOpUnitOutput);
    maxpoolUnit.tensorOut(finalVectorOutput);
    maxpoolUnit.doneSignal(done);

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
