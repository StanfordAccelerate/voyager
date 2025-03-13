#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
// clang-format off
#include "VectorOps.h"
// clang-format on
#include "Broadcaster.h"
#include "ParamsDeserializer.h"
#include "VectorFetch.h"
#include "VectorUnitOutput.h"

template <typename IOType, typename VectorType, typename BufferType,
          typename ScaleType, int Width>
SC_MODULE(VectorOpUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<VectorInstructions> CCS_INIT_S1(vector_op_inst);
  Connections::In<VectorInstructions> CCS_INIT_S1(accumulation_inst);
  Connections::In<VectorInstructions> CCS_INIT_S1(reduction_inst);

  Connections::In<Pack1D<BufferType, Width>> CCS_INIT_S1(systolicArrayOutput);
  Connections::In<Pack1D<VectorType, Width>> CCS_INIT_S1(vectorFetch0Output);
  Connections::In<Pack1D<VectorType, Width>> CCS_INIT_S1(vectorFetch1Output);
  Connections::In<Pack1D<VectorType, Width>> CCS_INIT_S1(vectorFetch2Output);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch3AddressRequest);
  Connections::In<Pack1D<IOType, 16 / IOType::width>> CCS_INIT_S1(
      vectorFetch3DataResponse);

  Connections::Out<Pack1D<VectorType, Width>> CCS_INIT_S1(
      vector_op_unit_output);

  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      accumulation_input);
  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      accumulation_output);

  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      reduction_input);

  Broadcaster<Pack1D<VectorType, Width>> CCS_INIT_S1(reduction_broadcaster);
  Connections::Combinational<ac_int<16, false>> broadcast_count;
  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      broadcast_input);
  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      reduction_output);

  Broadcaster<Pack1D<VectorType, Width>> CCS_INIT_S1(reduction_broadcaster_1);
  Connections::Combinational<ac_int<16, false>> broadcast1_count;
  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      broadcast1_input);
  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      reduction_output_1);

  SC_CTOR(VectorOpUnit) {
    reduction_broadcaster.clk(clk);
    reduction_broadcaster.rstn(rstn);
    reduction_broadcaster.dataIn(broadcast_input);
    reduction_broadcaster.count(broadcast_count);
    reduction_broadcaster.dataOut(reduction_output);

    reduction_broadcaster_1.clk(clk);
    reduction_broadcaster_1.rstn(rstn);
    reduction_broadcaster_1.dataIn(broadcast1_input);
    reduction_broadcaster_1.count(broadcast1_count);
    reduction_broadcaster_1.dataOut(reduction_output_1);

    SC_THREAD(run_vector_ops);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(run_accumulation);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(run_reduction);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run_vector_ops() {
    vector_op_inst.Reset();
    systolicArrayOutput.Reset();
    vectorFetch0Output.Reset();
    vectorFetch1Output.Reset();
    vectorFetch2Output.Reset();
    vector_op_unit_output.Reset();
    accumulation_input.ResetWrite();
    accumulation_output.ResetRead();
    reduction_input.ResetWrite();
    reduction_output.ResetRead();
    vectorFetch3AddressRequest.Reset();
    vectorFetch3DataResponse.Reset();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode bubble
    while (true) {
      VectorInstructions inst = vector_op_inst.Pop();

      Pack1D<VectorType, Width> res0;
      Pack1D<VectorType, Width> res1;
      Pack1D<VectorType, Width> res2;
      Pack1D<VectorType, Width> res3;

      // Vector unit inputs
      Pack1D<VectorType, Width> op0_src0;
      Pack1D<VectorType, Width> op0_src1;
      Pack1D<VectorType, Width> op2_src1;
      Pack1D<VectorType, Width> op3_src1;

      if (inst.vector_op0_src0 == VectorInstructions::from_matrix_unit ||
          inst.vector_op0_src1 == VectorInstructions::from_matrix_unit) {
        Pack1D<BufferType, Width> sa_output = systolicArrayOutput.Pop();

        Pack1D<VectorType, Width> temp;
        if (inst.vdequantize) {
          vdequantize<BufferType, VectorType, Width>(sa_output, temp,
                                                     inst.vector_dq_scale);
        } else {
#pragma hls_unroll yes
          for (int i = 0; i < Width; i++) {
            temp[i] = sa_output[i];
          }
        }

        if (inst.vector_op0_src0 == VectorInstructions::from_matrix_unit) {
          op0_src0 = temp;
        } else {
          op0_src1 = temp;
        }
      }

      if (inst.vector_op0_src0 == VectorInstructions::from_vector_fetch_0 ||
          inst.vector_op0_src1 == VectorInstructions::from_vector_fetch_0) {
        Pack1D<VectorType, Width> temp = vectorFetch0Output.Pop();
        if (inst.vector_op0_src0 == VectorInstructions::from_vector_fetch_0) {
          op0_src0 = temp;
        } else {
          op0_src1 = temp;
        }
      }

      if (inst.vector_op0_src0 == VectorInstructions::from_vector_fetch_1 ||
          inst.vector_op0_src1 == VectorInstructions::from_vector_fetch_1) {
        Pack1D<VectorType, Width> temp = vectorFetch1Output.Pop();
        if (inst.vector_op0_src0 == VectorInstructions::from_vector_fetch_1) {
          op0_src0 = temp;
        } else {
          op0_src1 = temp;
        }
      }

      if (inst.vector_op2_src1 == VectorInstructions::from_vector_fetch_2 ||
          inst.vector_op3_src1 == VectorInstructions::from_vector_fetch_2) {
        Pack1D<VectorType, Width> temp = vectorFetch2Output.Pop();
        if (inst.vector_op2_src1 == VectorInstructions::from_vector_fetch_2) {
          op2_src1 = temp;
        } else {
          op3_src1 = temp;
        }
      }

      if (inst.vector_op0_src0 == VectorInstructions::from_accumulation ||
          inst.vector_op0_src1 == VectorInstructions::from_accumulation ||
          inst.vector_op2_src1 == VectorInstructions::from_accumulation) {
        Pack1D<VectorType, Width> temp = accumulation_output.Pop();
        if (inst.vector_op0_src0 == VectorInstructions::from_accumulation) {
          op0_src0 = temp;
        } else if (inst.vector_op0_src1 ==
                   VectorInstructions::from_accumulation) {
          op0_src1 = temp;
        } else {
          op2_src1 = temp;
        }
      }

      if (inst.vector_op0_src0 == VectorInstructions::from_reduction_0 ||
          inst.vector_op0_src1 == VectorInstructions::from_reduction_0 ||
          inst.vector_op2_src1 == VectorInstructions::from_reduction_0) {
        Pack1D<VectorType, Width> temp = reduction_output.Pop();
        if (inst.vector_op0_src0 == VectorInstructions::from_reduction_0) {
          op0_src0 = temp;
        } else if (inst.vector_op0_src1 ==
                   VectorInstructions::from_reduction_0) {
          op0_src1 = temp;
        } else {
          op2_src1 = temp;
        }
      }

      if (inst.vector_op2_src1 == VectorInstructions::from_reduction_1) {
        op2_src1 = reduction_output_1.Pop();
      }

      if (inst.vector_op0_src0 == VectorInstructions::from_immediate_0 ||
          inst.vector_op0_src1 == VectorInstructions::from_immediate_0) {
        Pack1D<VectorType, Width> temp;
#pragma hls_unroll yes
        for (int i = 0; i < Width; i++) {
          temp[i].set_bits(inst.immediate0);
        }

        if (inst.vector_op0_src0 == VectorInstructions::from_immediate_0) {
          op0_src0 = temp;
        } else {
          op0_src1 = temp;
        }
      }

      if (inst.vector_op2_src1 == VectorInstructions::from_immediate_1 ||
          inst.vector_op3_src1 == VectorInstructions::from_immediate_1) {
        Pack1D<VectorType, Width> temp;
#pragma hls_unroll yes
        for (int i = 0; i < Width; i++) {
          temp[i].set_bits(inst.immediate1);
        }

        if (inst.vector_op2_src1 == VectorInstructions::from_immediate_1) {
          op2_src1 = temp;
        } else {
          op3_src1 = temp;
        }
      }

      // Stage 0: add, sub, mult
      if (inst.vector_op0 == VectorInstructions::vadd ||
          inst.vector_op0 == VectorInstructions::vsub) {
        if (inst.vector_op0 == VectorInstructions::vsub) {
#pragma hls_unroll yes
          for (int i = 0; i < Width; i++) {
            op0_src1[i] = op0_src1[i].negate();
          }
        }
        vadd<VectorType, Width>(op0_src0, op0_src1, res0);
      } else if (inst.vector_op0 == VectorInstructions::vmult) {
        vmult<VectorType, Width>(op0_src0, op0_src1, res0);
      } else {
        res0 = op0_src0;
      }

      // Stage 1: exp, abs, activations
      if (inst.vector_op1 == VectorInstructions::vexp) {
        vexp<VectorType, Width>(res0, res1);
      } else if (inst.vector_op1 == VectorInstructions::vabs) {
        vabs<VectorType, Width>(res0, res1);
      } else if (inst.vector_op1 == VectorInstructions::vrelu) {
        vrelu<VectorType, Width>(res0, res1);
      } else if (inst.vector_op1 == VectorInstructions::vgelu) {
        vgelu<VectorType, Width>(res0, res1);
      } else if (inst.vector_op1 == VectorInstructions::vsilu) {
        vsilu<VectorType, Width>(res0, res1);
      } else if (inst.vector_op1 == VectorInstructions::vmap) {
        for (int i = 0; i < Width; i++) {
          DataTypes::bfloat16 value = res0[i];

          ac_int<32, false> address = value.bits_rep() * 2;
          MemoryRequest request = {inst.VMAP_OFFSET + address, 2};
          vectorFetch3AddressRequest.Push(request);

          constexpr int num_words = 16 / IOType::width;
          Pack1D<IOType, num_words> response = vectorFetch3DataResponse.Pop();

          ac_int<16, false> bits = 0;
#pragma hls_unroll yes
          for (int j = 0; j < num_words; j++) {
            ac_int<16, false> bits_rep = response[j].bits_rep();
            bits = bits | (bits_rep << (IOType::width * j));
          }

          value.set_bits(bits);
          res1[i] = value;
        }
      } else {
        res1 = res0;
      }

      // Stage 2: add, mult, square
      if (inst.vector_op2 == VectorInstructions::vadd) {
        vadd<VectorType, Width>(res1, op2_src1, res2);
      } else if (inst.vector_op2 == VectorInstructions::vmult ||
                 inst.vector_op2 == VectorInstructions::vsquare) {
        if (inst.vector_op2 == VectorInstructions::vsquare) {
          op2_src1 = res1;
        }
        vmult<VectorType, Width>(res1, op2_src1, res2);
      } else {
        res2 = res1;
      }

      // Stage 3: div, quantize
      if (inst.vector_op3 == VectorInstructions::vdiv) {
        vdiv<VectorType, Width>(res2, op3_src1, res3);
      } else {
        res3 = res2;
      }

      // Write outputs
      if (inst.vdest == VectorInstructions::to_output) {
        vector_op_unit_output.Push(res3);
      } else if (inst.vdest == VectorInstructions::to_reduce) {
        reduction_input.Push(res3);
      } else if (inst.vdest == VectorInstructions::to_accumulate) {
        accumulation_input.Push(res3);
      }
    }
  }

  void run_accumulation() {
    accumulation_inst.Reset();
    accumulation_input.ResetRead();
    accumulation_output.ResetWrite();

    wait();

    while (true) {
      VectorInstructions inst = accumulation_inst.Pop();

      Pack1D<VectorType, Width> outputs;

#pragma hls_unroll yes
      for (int i = 0; i < Width; i++) {
        outputs[i].set_zero();
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int i = 0; i < inst.rCount; i++) {
        Pack1D<VectorType, Width> op = accumulation_input.Pop();

        if (inst.rOp == VectorInstructions::radd) {
#pragma hls_unroll yes
          for (int j = 0; j < Width; j++) {
            outputs[j] += op[j];
          }
        } else if (inst.rOp == VectorInstructions::rmax) {
#pragma hls_unroll yes
          for (int j = 0; j < Width; j++) {
            outputs[j] = (outputs[j] < op[j] || i == 0) ? op[j] : outputs[j];
          }
        }
      }

      DLOG("accumulation finished: " << outputs);
      accumulation_output.Push(outputs);
    }
  }

  void run_reduction() {
    reduction_inst.Reset();
    reduction_input.ResetRead();
    broadcast_input.ResetWrite();
    broadcast_count.ResetWrite();
    broadcast1_input.ResetWrite();
    broadcast1_count.ResetWrite();

    wait();

    while (true) {
      VectorInstructions inst = reduction_inst.Pop();

      Pack1D<VectorType, Width> res;
      VectorType output;

      int num_outputs = inst.rDuplicate ? 1 : Width;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int i = 0; i < num_outputs; i++) {
        for (int j = 0; j < inst.rCount; j++) {
          Pack1D<VectorType, Width> op = reduction_input.Pop();

          if (inst.rOp == VectorInstructions::radd) {
            VectorType sum = treeadd(op);
            output = (j == 0) ? sum : output + sum;
          } else if (inst.rOp == VectorInstructions::rmax) {
            VectorType max = treemax(op);
            output = (output < max || j == 0) ? max : output;
          }
        }

        if (!inst.rDuplicate) {
          res[i] = output;
        }

        DLOG("Reduction " << i << "/" << num_outputs << " : " << output);
      }

      if (inst.rSqrt) {
        output = output.sqrt();
      }

      if (inst.rReciprocal) {
        output = output.reciprocal();
      }

      if (inst.rMax1) {
        output = output.max1();
      }

      // Duplicate the scalar result into a vector
      if (inst.rDuplicate) {
#pragma hls_unroll yes
        for (int i = 0; i < Width; i++) {
          res[i] = output;
        }
      }

      if (inst.rdest == 0) {
        broadcast_count.Push(inst.immediate0);
        broadcast_input.Push(res);
      } else {
        broadcast1_count.Push(inst.immediate0);
        broadcast1_input.Push(res);
      }
    }
  }
};

template <typename IOType, typename VectorType, typename BufferType,
          typename ScaleType, int Width>
SC_MODULE(VectorUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<int> CCS_INIT_S1(serialParamsIn);

  Connections::In<Pack1D<BufferType, Width>> CCS_INIT_S1(systolicArrayOutput);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch0AddressRequest);
  Connections::In<Pack1D<IOType, Width>> CCS_INIT_S1(vectorFetch0DataResponse);
  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      vectorFetch0DataResponseConverted);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch1AddressRequest);
  Connections::In<Pack1D<IOType, Width>> CCS_INIT_S1(vectorFetch1DataResponse);
  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      vectorFetch1DataResponseConverted);
  Connections::Combinational<ScaleType> CCS_INIT_S1(vectorFetch1Scale);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch2AddressRequest);
  Connections::In<Pack1D<IOType, Width>> CCS_INIT_S1(vectorFetch2DataResponse);
  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      vectorFetch2DataResponseConverted);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vectorFetch3AddressRequest);
  Connections::In<Pack1D<IOType, 16 / IOType::width>> CCS_INIT_S1(
      vectorFetch3DataResponse);

  Connections::Out<Pack1D<IOType, Width>> CCS_INIT_S1(vector_output);
  Connections::Out<ac_int<64, false>> CCS_INIT_S1(vector_output_address);
  Connections::Out<Pack1D<DataTypes::int8, 1>> CCS_INIT_S1(scalar_output);
  Connections::Out<ac_int<64, false>> CCS_INIT_S1(scalar_output_address);

  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      vector_op_unit_output);

  Connections::SyncOut CCS_INIT_S1(start);
  Connections::SyncOut CCS_INIT_S1(done);

  VectorFetchUnit<IOType, VectorType, ScaleType, Width, VECTOR_INPUT_DATATYPES>
      CCS_INIT_S1(vector_fetch);
  Connections::Combinational<VectorParams> CCS_INIT_S1(vectorFetchParams);

  VectorOpUnit<IOType, VectorType, BufferType, ScaleType, Width> CCS_INIT_S1(
      vector_op_unit);

  VectorUnitOutput<VectorType, ScaleType, IOType, Width, OUTPUT_DATATYPES>
      CCS_INIT_S1(vector_unit_output);
  Connections::Combinational<VectorParams> CCS_INIT_S1(
      vector_unit_output_params);

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

    vector_fetch.clk(clk);
    vector_fetch.rstn(rstn);
    vector_fetch.paramsIn(vectorFetchParams);
    vector_fetch.vectorFetch0AddressRequest(vectorFetch0AddressRequest);
    vector_fetch.vectorFetch0DataResponse(vectorFetch0DataResponse);
    vector_fetch.vectorFetch0DataResponseConverted(
        vectorFetch0DataResponseConverted);
    vector_fetch.vectorFetch1AddressRequest(vectorFetch1AddressRequest);
    vector_fetch.vectorFetch1DataResponse(vectorFetch1DataResponse);
    vector_fetch.vectorFetch1DataResponseConverted(
        vectorFetch1DataResponseConverted);
    vector_fetch.vectorFetch2AddressRequest(vectorFetch2AddressRequest);
    vector_fetch.vectorFetch2DataResponse(vectorFetch2DataResponse);
    vector_fetch.vectorFetch2DataResponseConverted(
        vectorFetch2DataResponseConverted);

    vector_op_unit.clk(clk);
    vector_op_unit.rstn(rstn);
    vector_op_unit.vector_op_inst(vectorOpInstructions);
    vector_op_unit.accumulation_inst(accumulationOpInstructions);
    vector_op_unit.reduction_inst(reduceOpInstructions);
    vector_op_unit.systolicArrayOutput(systolicArrayOutput);
    vector_op_unit.vectorFetch0Output(vectorFetch0DataResponseConverted);
    vector_op_unit.vectorFetch1Output(vectorFetch1DataResponseConverted);
    vector_op_unit.vectorFetch2Output(vectorFetch2DataResponseConverted);
    vector_op_unit.vector_op_unit_output(vector_op_unit_output);
    vector_op_unit.vectorFetch3AddressRequest(vectorFetch3AddressRequest);
    vector_op_unit.vectorFetch3DataResponse(vectorFetch3DataResponse);

    vector_unit_output.clk(clk);
    vector_unit_output.rstn(rstn);
    vector_unit_output.params_in(vector_unit_output_params);
    vector_unit_output.tensor_in(vector_op_unit_output);
    vector_unit_output.vector_output(vector_output);
    vector_unit_output.vector_output_address(vector_output_address);
    vector_unit_output.scalar_output(scalar_output);
    vector_unit_output.scalar_output_address(scalar_output_address);
    vector_unit_output.done(done);

    SC_THREAD(read_params);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(send_instructions);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void read_params() {
    vectorParamsIn.ResetRead();
    vectorFetchParams.ResetWrite();
    vector_unit_output_params.ResetWrite();

    wait();

    while (true) {
      VectorParams params = vectorParamsIn.Pop();

      vectorFetchParams.Push(params);
      vector_unit_output_params.Push(params);
      // outputAddressGenParams.Push(params);
    }
  }

  void send_instructions() {
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
        for (int i = 0; i < 8; i++) {
          VectorInstructions inst = instConfig.inst[i];

          for (int count = 0; count < instConfig.instCount[i]; count++) {
            if (inst.instType == VectorInstructions::vector) {
              vectorOpInstructions.Push(inst);
            } else if (inst.instType == VectorInstructions::accumulation) {
              accumulationOpInstructions.Push(inst);
            } else {
              reduceOpInstructions.Push(inst);
            }
          }

          if (i >= instConfig.instLen - 1) {
            break;
          }
        }
      }
    }
  }
};
