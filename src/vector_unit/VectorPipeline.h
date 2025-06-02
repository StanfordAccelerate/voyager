#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "../AccelTypes.h"
#include "../ArchitectureParams.h"
#include "Broadcaster.h"
#include "VectorOps.h"

template <typename VectorType, typename BufferType, int Width>
SC_MODULE(VectorPipeline) {
  sc_in<bool> clk;
  sc_in<bool> rstn;

  // Inputs
  Connections::In<VectorInstructions> instr;
  Connections::In<Pack1D<BufferType, Width>> matrix_unit_output;

#if DOUBLE_BUFFERED_ACCUM_BUFFER
  Connections::In<Pack1D<BufferType, Width>> accumulation_buffer_output;
#endif

#if SUPPORT_MVM
  Connections::In<Pack1D<VectorType, Width>> matrix_vector_unit_data;
#endif

  Connections::In<Pack1D<VectorType, Width>> vector_fetch_0_data;
  Connections::In<Pack1D<VectorType, Width>> vector_fetch_1_data;
  Connections::In<Pack1D<VectorType, Width>> vector_fetch_2_data;

  Connections::Out<MemoryRequest> vector_fetch_3_req;
  Connections::In<ac_int<16, false>> vector_fetch_3_resp;

  Connections::In<Pack1D<VectorType, Width>> accumulator_output;
  Connections::In<Pack1D<VectorType, Width>> reducer_output_0;
  Connections::In<Pack1D<VectorType, Width>> reducer_output_1;

  // Outputs to other submodules
  Connections::Out<VectorInstructions> quantizer_instr;
  Connections::Out<Pack1D<VectorType, Width>> quantizer_input;
  Connections::Out<Pack1D<VectorType, Width>> quantizer_scale;

  Connections::Out<Pack1D<VectorType, Width>> reducer_input;
  Connections::Out<Pack1D<VectorType, Width>> accumulator_input;

  SC_CTOR(VectorPipeline) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    instr.Reset();
    matrix_unit_output.Reset();
#if SUPPORT_MVM
    matrix_vector_unit_data.Reset();
#endif
    vector_fetch_0_data.Reset();
    vector_fetch_1_data.Reset();
    vector_fetch_2_data.Reset();
    vector_fetch_3_req.Reset();
    vector_fetch_3_resp.Reset();
    quantizer_instr.Reset();
    quantizer_input.Reset();
    quantizer_scale.Reset();
    accumulator_input.Reset();
    accumulator_output.Reset();
    reducer_input.Reset();
    reducer_output_0.Reset();
    reducer_output_1.Reset();
#if DOUBLE_BUFFERED_ACCUM_BUFFER
    accumulation_buffer_output.Reset();
#endif

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      VectorInstructions inst = instr.Pop();

      Pack1D<VectorType, Width> res0;
      Pack1D<VectorType, Width> res1;
      Pack1D<VectorType, Width> res2;

      // Vector unit inputs
      Pack1D<VectorType, Width> op0_src0;
      Pack1D<VectorType, Width> op0_src1;
      Pack1D<VectorType, Width> op2_src1;
      Pack1D<VectorType, Width> op3_src1;

      if (inst.vector_op0_src0 == VectorInstructions::from_matrix_unit ||
          inst.vector_op0_src1 == VectorInstructions::from_matrix_unit) {
        Pack1D<BufferType, Width> sa_output = matrix_unit_output.Pop();

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
#if DOUBLE_BUFFERED_ACCUM_BUFFER
      if (inst.vector_op0_src0 ==
              VectorInstructions::from_accumulation_buffer ||
          inst.vector_op0_src1 ==
              VectorInstructions::from_accumulation_buffer) {
        Pack1D<BufferType, Width> sa_output = accumulation_buffer_output.Pop();

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

        if (inst.vector_op0_src0 ==
            VectorInstructions::from_accumulation_buffer) {
          op0_src0 = temp;
        } else {
          op0_src1 = temp;
        }
      }
#endif

#if SUPPORT_MVM
      if (inst.vector_op0_src0 == VectorInstructions::from_matrix_vector_unit ||
          inst.vector_op0_src1 == VectorInstructions::from_matrix_vector_unit) {
        Pack1D<VectorType, Width> temp = matrix_vector_unit_data.Pop();
        if (inst.vector_op0_src0 ==
            VectorInstructions::from_matrix_vector_unit) {
          op0_src0 = temp;
        } else {
          op0_src1 = temp;
        }
      }
#endif

      if (inst.vector_op0_src0 == VectorInstructions::from_vector_fetch_0 ||
          inst.vector_op0_src1 == VectorInstructions::from_vector_fetch_0) {
        Pack1D<VectorType, Width> temp = vector_fetch_0_data.Pop();
        if (inst.vector_op0_src0 == VectorInstructions::from_vector_fetch_0) {
          op0_src0 = temp;
        } else {
          op0_src1 = temp;
        }
      }

      if (inst.vector_op0_src0 == VectorInstructions::from_vector_fetch_1 ||
          inst.vector_op0_src1 == VectorInstructions::from_vector_fetch_1) {
        Pack1D<VectorType, Width> temp = vector_fetch_1_data.Pop();
        if (inst.vector_op0_src0 == VectorInstructions::from_vector_fetch_1) {
          op0_src0 = temp;
        } else {
          op0_src1 = temp;
        }
      }

      if (inst.vector_op2_src1 == VectorInstructions::from_vector_fetch_2 ||
          inst.vector_op3_src1 == VectorInstructions::from_vector_fetch_2) {
        Pack1D<VectorType, Width> temp = vector_fetch_2_data.Pop();
        if (inst.vector_op2_src1 == VectorInstructions::from_vector_fetch_2) {
          op2_src1 = temp;
        } else {
          op3_src1 = temp;
        }
      }

      if (inst.vector_op0_src0 == VectorInstructions::from_accumulation ||
          inst.vector_op0_src1 == VectorInstructions::from_accumulation ||
          inst.vector_op2_src1 == VectorInstructions::from_accumulation) {
        Pack1D<VectorType, Width> temp = accumulator_output.Pop();
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
        Pack1D<VectorType, Width> temp = reducer_output_0.Pop();
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
        op2_src1 = reducer_output_1.Pop();
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

      // Stage 0: add, sub, mult
      if (inst.vector_op0 == VectorInstructions::vadd ||
          inst.vector_op0 == VectorInstructions::vsub) {
        if (inst.vector_op0 == VectorInstructions::vsub) {
#pragma hls_unroll yes
          for (int i = 0; i < Width; i++) {
            op0_src1[i] = op0_src1[i].negate();
          }
        }
        res0 = vadd<VectorType, Width>(op0_src0, op0_src1);
      } else if (inst.vector_op0 == VectorInstructions::vmult) {
        res0 = vmul<VectorType, Width>(op0_src0, op0_src1);
      } else {
        res0 = op0_src0;
      }

      // Stage 1: exp, abs, activations
      if (inst.vector_op1 == VectorInstructions::vexp) {
        res1 = vexp<VectorType, Width>(res0);
      } else if (inst.vector_op1 == VectorInstructions::vabs) {
        res1 = vabs<VectorType, Width>(res0);
      } else if (inst.vector_op1 == VectorInstructions::vrelu) {
        res1 = vrelu<VectorType, Width>(res0);
      } else if (inst.vector_op1 == VectorInstructions::vgelu) {
        res1 = vgelu<VectorType, Width>(res0);
      } else if (inst.vector_op1 == VectorInstructions::vsilu) {
        res1 = vsilu<VectorType, Width>(res0);
      } else if (inst.vector_op1 == VectorInstructions::vmap) {
        for (int i = 0; i < Width; i++) {
          DataTypes::bfloat16 value = res0[i];

          ac_int<32, false> address = value.bits_rep() * 2;
          MemoryRequest request = {inst.VMAP_OFFSET + address, 2};
          vector_fetch_3_req.Push(request);

          value.set_bits(vector_fetch_3_resp.Pop());
          res1[i] = value;
        }
      } else {
        res1 = res0;
      }

      // Stage 2: add, mult, square
      if (inst.vector_op2 == VectorInstructions::vadd) {
        res2 = vadd<VectorType, Width>(res1, op2_src1);
      } else if (inst.vector_op2 == VectorInstructions::vmult ||
                 inst.vector_op2 == VectorInstructions::vsquare) {
        if (inst.vector_op2 == VectorInstructions::vsquare) {
          op2_src1 = res1;
        }
        res2 = vmul<VectorType, Width>(res1, op2_src1);
      } else {
        res2 = res1;
      }

      // Write outputs
      if (inst.vdest == VectorInstructions::to_output) {
        quantizer_instr.Push(inst);
        quantizer_input.Push(res2);
        if (inst.vector_op3_src1 == VectorInstructions::from_vector_fetch_2) {
          quantizer_scale.Push(op3_src1);
        }
      } else if (inst.vdest == VectorInstructions::to_reduce) {
        reducer_input.Push(res2);
      } else if (inst.vdest == VectorInstructions::to_accumulate) {
        accumulator_input.Push(res2);
      }
    }
  }
};
