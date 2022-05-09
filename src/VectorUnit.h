#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
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
  Connections::In<Pack1D<IDTYPE, WIDTH> > CCS_INIT_S1(vectorFetch0Output);
  Connections::In<Pack1D<IDTYPE, WIDTH> > CCS_INIT_S1(vectorFetch1Output);
  Connections::In<Pack1D<IDTYPE, WIDTH> > CCS_INIT_S1(vectorFetch2Output);

  Connections::Out<Pack1D<IDTYPE, WIDTH> > CCS_INIT_S1(vectorOpUnitOutput);
  Connections::Out<Pack1D<IDTYPE, WIDTH> > CCS_INIT_S1(scalarOpUnitOutput);

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
      CCS_INIT_S1(reductionOpOutputSrc0);
  Connections::Combinational<
      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> >
      CCS_INIT_S1(reductionOpOutputSrc1);

  SC_CTOR(VectorOpUnit) {
    // systolicArrayOutput.enable_local_rand_stall();

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
    reductionOpOutputSrc0.ResetRead();
    reductionOpOutputSrc1.ResetRead();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      VectorInstructions inst = vectorOpUnitInstructions.Pop();
      DLOG("vector inst: ");

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
        Pack1D<IDTYPE, WIDTH> tmp = vectorFetch0Output.Pop();
#pragma hls_unroll yes
        for (int i = 0; i < WIDTH; i++) {
          op0Src0[i] = tmp[i];
        }
      } else if (inst.vInput == VectorInstructions::readFromAccumulation) {
        op0Src0 = accumulationOpOutput.Pop();
      }

      if (inst.vAccumulatePush) {
        accumulationOpInput.Push(op0Src0);
      }

      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> op0Src1;
      if (inst.vOp0Src1 == VectorInstructions::readInterface) {
        Pack1D<IDTYPE, WIDTH> tmp = vectorFetch1Output.Pop();
#pragma hls_unroll yes
        for (int i = 0; i < WIDTH; i++) {
          op0Src1[i] = tmp[i];
        }
        DLOG("read from fetch: " << op0Src1);
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
        DLOG(op0Src0 << " * " << op0Src1);

      } else {
        res0 = op0Src0;
      }

      DLOG("res0: " << res0);
      /*
       * Stage 1: exp
       */
      if (inst.vOp1 == VectorInstructions::vexp) {
        vexp<typename ACC_DTYPE::DecomposedPosit, WIDTH>(res0, res1);
      } else {
        res1 = res0;
      }

      DLOG("res1: " << res1);

      /*
       * Stage 2: reduction
       */
      if (inst.vOp2 == VectorInstructions::toReduce) {
        reductionOpInput.Push(res1);
      } else {
        res2 = res1;
      }

      DLOG("res2: " << res2);

      /*
       * Stage 3: add, div
       */
      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> op3Src0;
      if (inst.vOp3Src0 == VectorInstructions::readReduceInterface) {
        op3Src0 = reductionOpOutputSrc0.Pop();
        DLOG("reading from reduce: " << op3Src0);
      } else {
        op3Src0 = res2;
      }

      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> op3Src1;
      if (inst.vOp3Src1 == VectorInstructions::readReduceInterface) {
        op3Src1 = reductionOpOutputSrc1.Pop();
      } else if (inst.vOp3Src1 == VectorInstructions::readNormalInterface) {
        Pack1D<IDTYPE, WIDTH> tmp = vectorFetch2Output.Pop();

#pragma hls_unroll yes
        for (int i = 0; i < WIDTH; i++) {
          op3Src1[i] = tmp[i];
        }
        DLOG("reading from vectorfetch2: " << op3Src1);
      }

      if (inst.vOp3 == VectorInstructions::vadd) {
        vadd<typename ACC_DTYPE::DecomposedPosit, WIDTH>(op3Src0, op3Src1,
                                                         res3);
      } else if (inst.vOp3 == VectorInstructions::vdiv) {
        // vdiv<typename ACC_DTYPE::DecomposedPosit, WIDTH>(op3Src0, op3Src1,
        //                                                  res3);
      } else {
        res3 = op3Src0;
      }

      DLOG("res3: " << res3);

      /*
       * Stage 4: relu
       */
      if (inst.vOp4 == VectorInstructions::vrelu) {
        vrelu<typename ACC_DTYPE::DecomposedPosit, WIDTH>(res3, res4);
      } else {
        res4 = res3;
      }

      DLOG("res4: " << res4);

      if (inst.vDest == VectorInstructions::vWriteOut) {
        // convert to Posit8 and write out
        Pack1D<IDTYPE, WIDTH> tmp;
#pragma hls_unroll yes
        for (int i = 0; i < WIDTH; i++) {
          tmp[i] = static_cast<IDTYPE>(res4[i]);
        }
        DLOG("vector unit output: " << tmp);

        vectorOpUnitOutput.Push(tmp);
      }
    }
  }

  void accumulationOpRun() {
    accumulationOpInput.ResetRead();
    accumulationOpOutput.ResetWrite();
    accumulationOpUnitInstructions.Reset();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      VectorInstructions inst = accumulationOpUnitInstructions.Pop();

      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> accum;

#pragma hls_unroll yes
      for (int i = 0; i < WIDTH; i++) {
        accum[i].setZero();
      }

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
    scalarOpUnitOutput.Reset();
    reductionOpUnitInstructions.Reset();
    reductionOpInput.ResetRead();
    reductionOpOutputSrc0.ResetWrite();
    reductionOpOutputSrc1.ResetWrite();
    scalarOpUnitOutput.Reset();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      VectorInstructions inst = reductionOpUnitInstructions.Pop();

      Pack1D<typename ACC_DTYPE::DecomposedPosit, WIDTH> res;

      int iterationCount = inst.rDuplicate ? 1 : WIDTH;

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

        if (inst.rDuplicate) {  // Duplicate the scalar result into a vector
#pragma hls_unroll yes
          for (int i = 0; i < WIDTH; i++) {
            res[i] = prevResult;
          }
        } else {
          res[index] = prevResult;
        }
        DLOG("Reduction " << index << "/" << iterationCount << " : "
                          << prevResult << std::endl
                          << res);
      }

      if (inst.rDest != 0) {
        if (inst.rDest == VectorInstructions::toVectorSrc0) {
          reductionOpOutputSrc0.Push(res);
        } else if (inst.rDest == VectorInstructions::toVectorSrc1) {
          reductionOpOutputSrc1.Push(res);
        } else if (inst.rDest == VectorInstructions::sWriteOut) {
          Pack1D<IDTYPE, WIDTH> outputRes;
#pragma hls_unroll yes
          for (int i = 0; i < WIDTH; i++) {
            outputRes[i] = res[i];
          }

          scalarOpUnitOutput.Push(outputRes);
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

  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch1AddressRequest);
  Connections::In<Pack1D<ODTYPE, WIDTH> > CCS_INIT_S1(vectorFetch1DataResponse);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch2AddressRequest);
  Connections::In<Pack1D<ODTYPE, WIDTH> > CCS_INIT_S1(vectorFetch2DataResponse);
  Connections::Combinational<Pack1D<ODTYPE, WIDTH> > CCS_INIT_S1(
      vectorFetch2DataResponseReplicated);

  Connections::Out<int> CCS_INIT_S1(scalarOutputAddress);
  Connections::Out<Pack1D<ODTYPE, WIDTH> > CCS_INIT_S1(scalarUnitOutput);

  Connections::Out<int> CCS_INIT_S1(vectorOutputAddress);
  Connections::Out<Pack1D<ODTYPE, WIDTH> > CCS_INIT_S1(finalVectorOutput);
  Connections::Combinational<Pack1D<ODTYPE, WIDTH> > CCS_INIT_S1(
      vectorOpUnitOutput);

  Connections::SyncOut CCS_INIT_S1(done);

  VectorFetchUnit<ODTYPE, WIDTH> CCS_INIT_S1(vectorFetch);
  Connections::Combinational<VectorParams> CCS_INIT_S1(vectorFetchParams);

  VectorOpUnit<ODTYPE, ACC_DTYPE, WIDTH> CCS_INIT_S1(vectorOpUnit);

  MaxpoolUnit<ODTYPE, WIDTH> CCS_INIT_S1(maxpoolUnit);
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
    vectorFetch.vectorFetch1AddressRequest(vectorFetch1AddressRequest);
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
    vectorOpUnit.vectorFetch0Output(vectorFetch0DataResponse);
    vectorOpUnit.vectorFetch1Output(vectorFetch1DataResponse);
    vectorOpUnit.vectorFetch2Output(vectorFetch2DataResponseReplicated);
    vectorOpUnit.vectorOpUnitOutput(vectorOpUnitOutput);
    vectorOpUnit.scalarOpUnitOutput(scalarUnitOutput);

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
    outputAddressGenerator.scalarOutputAddress(scalarOutputAddress);

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

    wait();

    while (true) {
      VectorInstructionConfig instConfig = vectorInstructionsIn.Pop();

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

          if (instAddress >= instConfig.instLen) {
            break;
          }
        }
      }
    }
  }
};
