#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "Broadcaster.h"
#include "OutputController.h"
#include "ParamsDeserializer.h"
#include "VectorFetch.h"
#include "VectorOps.h"

template <typename VectorType, typename BufferType, typename ScaleType,
          int Width>
SC_MODULE(VectorOpUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<VectorInstructions> CCS_INIT_S1(vector_op_inst_in);
  Connections::In<VectorInstructions> CCS_INIT_S1(accumulation_inst_in);
  Connections::In<VectorInstructions> CCS_INIT_S1(reduction_inst_in);

  Connections::In<Pack1D<BufferType, Width>> CCS_INIT_S1(
      accumulationBufferOutput);
  Connections::In<Pack1D<VectorType, Width>> CCS_INIT_S1(
      vector_fetch_0_data_in);
  Connections::In<Pack1D<VectorType, Width>> CCS_INIT_S1(
      vector_fetch_1_data_in);
  Connections::In<Pack1D<VectorType, Width>> CCS_INIT_S1(
      vector_fetch_2_data_in);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_3_request_out);
  Connections::In<ac_int<16, false>> CCS_INIT_S1(vector_fetch_3_response_in);

  Connections::Out<Pack1D<VectorType, Width>> CCS_INIT_S1(vector_op_unit_out);
  Connections::Out<ScaleType> CCS_INIT_S1(mx_scale_out);

  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      accumulation_input);
  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      accumulation_output);

  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      reduction_input);

  Broadcaster<Pack1D<VectorType, Width>> CCS_INIT_S1(reduction_broadcaster_0);
  Connections::Combinational<ac_int<16, false>> broadcast_count_0;
  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      broadcast_input_0);
  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      reduction_output_0);

  Broadcaster<Pack1D<VectorType, Width>> CCS_INIT_S1(reduction_broadcaster_1);
  Connections::Combinational<ac_int<16, false>> broadcast_count_1;
  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      broadcast_input_1);
  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      reduction_output_1);

  SC_CTOR(VectorOpUnit) {
    reduction_broadcaster_0.clk(clk);
    reduction_broadcaster_0.rstn(rstn);
    reduction_broadcaster_0.dataIn(broadcast_input_0);
    reduction_broadcaster_0.count(broadcast_count_0);
    reduction_broadcaster_0.dataOut(reduction_output_0);

    reduction_broadcaster_1.clk(clk);
    reduction_broadcaster_1.rstn(rstn);
    reduction_broadcaster_1.dataIn(broadcast_input_1);
    reduction_broadcaster_1.count(broadcast_count_1);
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
    vector_op_inst_in.Reset();
    accumulationBufferOutput.Reset();
    vector_fetch_0_data_in.Reset();
    vector_fetch_1_data_in.Reset();
    vector_fetch_2_data_in.Reset();
    vector_fetch_3_request_out.Reset();
    vector_fetch_3_response_in.Reset();
    mx_scale_out.Reset();
    vector_op_unit_out.Reset();
    accumulation_input.ResetWrite();
    accumulation_output.ResetRead();
    reduction_input.ResetWrite();
    reduction_output_0.ResetRead();
    reduction_output_1.ResetRead();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode bubble
    while (true) {
      VectorInstructions inst = vector_op_inst_in.Pop();

      Pack1D<VectorType, Width> res0;
      Pack1D<VectorType, Width> res1;
      Pack1D<VectorType, Width> res2;
      Pack1D<VectorType, Width> res3;
      ScaleType scale;

      // Vector unit inputs
      Pack1D<VectorType, Width> op0_src0;
      Pack1D<VectorType, Width> op0_src1;
      Pack1D<VectorType, Width> op2_src1;
      Pack1D<VectorType, Width> op3_src1;

      if (inst.vector_op0_src0 == VectorInstructions::from_matrix_unit ||
          inst.vector_op0_src1 == VectorInstructions::from_matrix_unit) {
        Pack1D<BufferType, Width> sa_output = accumulationBufferOutput.Pop();

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
        Pack1D<VectorType, Width> temp = vector_fetch_0_data_in.Pop();
        if (inst.vector_op0_src0 == VectorInstructions::from_vector_fetch_0) {
          op0_src0 = temp;
        } else {
          op0_src1 = temp;
        }
      }

      if (inst.vector_op0_src0 == VectorInstructions::from_vector_fetch_1 ||
          inst.vector_op0_src1 == VectorInstructions::from_vector_fetch_1) {
        Pack1D<VectorType, Width> temp = vector_fetch_1_data_in.Pop();
        if (inst.vector_op0_src0 == VectorInstructions::from_vector_fetch_1) {
          op0_src0 = temp;
        } else {
          op0_src1 = temp;
        }
      }

      if (inst.vector_op2_src1 == VectorInstructions::from_vector_fetch_2 ||
          inst.vector_op3_src1 == VectorInstructions::from_vector_fetch_2) {
        Pack1D<VectorType, Width> temp = vector_fetch_2_data_in.Pop();
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
        Pack1D<VectorType, Width> temp = reduction_output_0.Pop();
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

      if (inst.vector_op2_src1 == VectorInstructions::from_immediate_1) {
#pragma hls_unroll yes
        for (int i = 0; i < Width; i++) {
          op2_src1[i].set_bits(inst.immediate1);
        }
      }

      if (inst.vector_op3_src1 == VectorInstructions::from_immediate_2) {
#pragma hls_unroll yes
        for (int i = 0; i < Width; i++) {
          op3_src1[i].set_bits(inst.immediate2);
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
          vector_fetch_3_request_out.Push(request);

          value.set_bits(vector_fetch_3_response_in.Pop());
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
#if SUPPORT_MX
      if (inst.vector_op3 == VectorInstructions::vquantize_mx) {
        vquantize_mx<VectorType, ScaleType, Width>(res2, scale,
                                                   inst.immediate2);

#pragma hls_unroll yes
        for (int i = 0; i < Width; i++) {
          op3_src1[i] = scale;
        }

        mx_scale_out.Push(scale);
      }
#endif

      if (inst.vector_op3 == VectorInstructions::vdiv ||
          inst.vector_op3 == VectorInstructions::vquantize_mx) {
        vdiv<VectorType, Width>(res2, op3_src1, res3);
      } else {
        res3 = res2;
      }

      // Write outputs
      if (inst.vdest == VectorInstructions::to_output) {
        vector_op_unit_out.Push(res3);
      } else if (inst.vdest == VectorInstructions::to_reduce) {
        reduction_input.Push(res3);
      } else if (inst.vdest == VectorInstructions::to_accumulate) {
        accumulation_input.Push(res3);
      }
    }
  }

  void run_accumulation() {
    accumulation_inst_in.Reset();
    accumulation_input.ResetRead();
    accumulation_output.ResetWrite();

    wait();

    while (true) {
      VectorInstructions inst = accumulation_inst_in.Pop();

      Pack1D<VectorType, Width> outputs;

#pragma hls_unroll yes
      for (int i = 0; i < Width; i++) {
        outputs[i] = VectorType::zero();
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int i = 0; i < inst.reduce_count; i++) {
        Pack1D<VectorType, Width> op = accumulation_input.Pop();

        if (inst.reduce_op == VectorInstructions::radd) {
#pragma hls_unroll yes
          for (int j = 0; j < Width; j++) {
            outputs[j] += op[j];
          }
        } else if (inst.reduce_op == VectorInstructions::rmax) {
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
    reduction_inst_in.Reset();
    reduction_input.ResetRead();
    broadcast_input_0.ResetWrite();
    broadcast_count_0.ResetWrite();
    broadcast_input_1.ResetWrite();
    broadcast_count_1.ResetWrite();

    wait();

    while (true) {
      VectorInstructions inst = reduction_inst_in.Pop();

      Pack1D<VectorType, Width> res;
      VectorType output;

      int num_outputs = inst.rduplicate ? 1 : Width;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int i = 0; i < num_outputs; i++) {
        for (int j = 0; j < inst.reduce_count; j++) {
          Pack1D<VectorType, Width> op = reduction_input.Pop();

          if (inst.reduce_op == VectorInstructions::radd) {
            VectorType sum = treeadd(op);
            output = (j == 0) ? sum : output + sum;
          } else if (inst.reduce_op == VectorInstructions::rmax) {
            VectorType max = treemax(op);
            output = (output < max || j == 0) ? max : output;
          }
        }

        if (!inst.rduplicate) {
          res[i] = output;
        }

        DLOG("Reduction " << i << "/" << num_outputs << " : " << output);
      }

      if (inst.rsqrt) {
        output = output.sqrt();
      }

      if (inst.rreciprocal) {
        output = output.reciprocal();
      }

      // Duplicate the scalar result into a vector
      if (inst.rduplicate) {
#pragma hls_unroll yes
        for (int i = 0; i < Width; i++) {
          res[i] = output;
        }
      }

      if (inst.rdest == 0) {
        broadcast_count_0.Push(inst.immediate0);
        broadcast_input_0.Push(res);
      } else {
        broadcast_count_1.Push(inst.immediate0);
        broadcast_input_1.Push(res);
      }
    }
  }
};

template <typename VectorType, typename BufferType, typename ScaleType,
          int Width>
SC_MODULE(VectorUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  VectorParamsDeserializer CCS_INIT_S1(param_deserializer);

  Connections::Out<ac_int<16, false>> accumulation_buffer_read_address[2];
  Connections::In<Pack1D<BufferType, Width>> accumulation_buffer_read_data[2];
  Connections::SyncOut accumulation_buffer_done[2];
  Connections::Out<BufferWriteRequest<Pack1D<BufferType, Width>>>
      accumulation_buffer_write_request[2];
  Connections::Combinational<Pack1D<BufferType, Width>>
      accumulationBufferOutput;
  Connections::In<ac_int<64, false>> CCS_INIT_S1(serial_params_in);
  Connections::Combinational<VectorParams> CCS_INIT_S1(vector_params);
  Connections::Combinational<VectorInstructionConfig> CCS_INIT_S1(
      vector_instruction);

  VectorOpUnit<VectorType, BufferType, ScaleType, Width> CCS_INIT_S1(
      vector_op_unit);

  Connections::Combinational<VectorInstructions> CCS_INIT_S1(vector_op_insts);
  Connections::Combinational<VectorInstructions> CCS_INIT_S1(
      accumulation_insts);
  Connections::Combinational<VectorInstructions> CCS_INIT_S1(reduction_insts);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_0_request_out);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_0_response_in);
  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      vector_fetch_0_data);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_1_request_out);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_1_response_in);
  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      vector_fetch_1_data);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_2_request_out);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_fetch_2_response_in);
  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      vector_fetch_2_data);

  Connections::Out<MemoryRequest> CCS_INIT_S1(vector_fetch_3_request_out);
  Connections::In<ac_int<16, false>> CCS_INIT_S1(vector_fetch_3_response_in);

  Connections::Combinational<Pack1D<VectorType, Width>> CCS_INIT_S1(
      vector_unit_output);
  Connections::Combinational<ScaleType> CCS_INIT_S1(mx_scale);

  Connections::Out<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(vector_out);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      vector_address_out);
  Connections::Out<ac_int<ScaleType::width, false>> CCS_INIT_S1(scale_out);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(scale_address_out);

  Connections::SyncOut CCS_INIT_S1(start);
  Connections::SyncOut CCS_INIT_S1(done);

  VectorFetchUnit<VectorType, BufferType, Width, VU_INPUT_TYPES> CCS_INIT_S1(
      vector_fetcher);
  Connections::Combinational<VectorParams> CCS_INIT_S1(vector_fetch_params);

  OutputController<VectorType, ScaleType, Width, OUTPUT_DATATYPES> CCS_INIT_S1(
      output_controller);
  Connections::Combinational<VectorParams> CCS_INIT_S1(
      output_controller_params);

  SC_CTOR(VectorUnit) {
    param_deserializer.clk(clk);
    param_deserializer.rstn(rstn);
    param_deserializer.serialParamsIn(serial_params_in);
    param_deserializer.vectorParamsOut(vector_params);
    param_deserializer.vectorInstructionsOut(vector_instruction);

    vector_fetcher.clk(clk);
    vector_fetcher.rstn(rstn);
    vector_fetcher.params_in(vector_fetch_params);
    vector_fetcher.accumulation_buffer_read_address[0](
        accumulation_buffer_read_address[0]);
    vector_fetcher.accumulation_buffer_read_address[1](
        accumulation_buffer_read_address[1]);
    vector_fetcher.accumulation_buffer_done[0](accumulation_buffer_done[0]);
    vector_fetcher.accumulation_buffer_done[1](accumulation_buffer_done[1]);
    vector_fetcher.accumulation_buffer_read_data[0](
        accumulation_buffer_read_data[0]);
    vector_fetcher.accumulation_buffer_read_data[1](
        accumulation_buffer_read_data[1]);
    vector_fetcher.accumulationBufferOutput(accumulationBufferOutput);
    vector_fetcher.accumulation_buffer_write_request[0](
        accumulation_buffer_write_request[0]);
    vector_fetcher.accumulation_buffer_write_request[1](
        accumulation_buffer_write_request[1]);
    vector_fetcher.vector_fetch_0_request_out(vector_fetch_0_request_out);
    vector_fetcher.vector_fetch_0_resp_in(vector_fetch_0_response_in);
    vector_fetcher.vector_fetch_0_data_out(vector_fetch_0_data);
    vector_fetcher.vector_fetch_1_request_out(vector_fetch_1_request_out);
    vector_fetcher.vector_fetch_1_resp_in(vector_fetch_1_response_in);
    vector_fetcher.vector_fetch_1_data_out(vector_fetch_1_data);
    vector_fetcher.vector_fetch_2_request_out(vector_fetch_2_request_out);
    vector_fetcher.vector_fetch_2_resp_in(vector_fetch_2_response_in);
    vector_fetcher.vector_fetch_2_data_out(vector_fetch_2_data);

    vector_op_unit.clk(clk);
    vector_op_unit.rstn(rstn);
    vector_op_unit.vector_op_inst_in(vector_op_insts);
    vector_op_unit.accumulation_inst_in(accumulation_insts);
    vector_op_unit.reduction_inst_in(reduction_insts);
    vector_op_unit.accumulationBufferOutput(accumulationBufferOutput);
    vector_op_unit.vector_fetch_0_data_in(vector_fetch_0_data);
    vector_op_unit.vector_fetch_1_data_in(vector_fetch_1_data);
    vector_op_unit.vector_fetch_2_data_in(vector_fetch_2_data);
    vector_op_unit.vector_fetch_3_request_out(vector_fetch_3_request_out);
    vector_op_unit.vector_fetch_3_response_in(vector_fetch_3_response_in);
    vector_op_unit.vector_op_unit_out(vector_unit_output);
    vector_op_unit.mx_scale_out(mx_scale);

    output_controller.clk(clk);
    output_controller.rstn(rstn);
    output_controller.params_in(output_controller_params);
    output_controller.vector_in(vector_unit_output);
    output_controller.scale_in(mx_scale);
    output_controller.vector_out(vector_out);
    output_controller.vector_address_out(vector_address_out);
    output_controller.scale_out(scale_out);
    output_controller.scale_address_out(scale_address_out);
    output_controller.done(done);

    SC_THREAD(read_params);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(send_instructions);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void read_params() {
    vector_params.ResetRead();
    vector_fetch_params.ResetWrite();
    output_controller_params.ResetWrite();

    wait();

    while (true) {
      VectorParams params = vector_params.Pop();

      vector_fetch_params.Push(params);
      output_controller_params.Push(params);
    }
  }

  void send_instructions() {
    vector_instruction.ResetRead();
    vector_op_insts.ResetWrite();
    reduction_insts.ResetWrite();
    accumulation_insts.ResetWrite();

    start.Reset();

    wait();

    while (true) {
      VectorInstructionConfig vector_inst_config = vector_instruction.Pop();

      start.SyncPush();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (int loop = 0; loop < vector_inst_config.instLoopCount; loop++) {
        for (int i = 0; i < 8; i++) {
          VectorInstructions inst = vector_inst_config.inst[i];

          for (int count = 0; count < vector_inst_config.instCount[i];
               count++) {
            if (inst.op_type == VectorInstructions::vector) {
              vector_op_insts.Push(inst);
            } else if (inst.op_type == VectorInstructions::accumulation) {
              accumulation_insts.Push(inst);
            } else {
              reduction_insts.Push(inst);
            }
          }

          if (i >= vector_inst_config.instLen - 1) {
            break;
          }
        }
      }
    }
  }
};
