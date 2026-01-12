#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ArchitectureParams.h"
#include "MulAddTree.h"
#include "ParamsDeserializer.h"

template <typename Input, typename Weight, typename Psum, typename Output,
          int OutputWidth, typename OutputFinal>
SC_MODULE(DwCUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  DwCParamsDeserializer CCS_INIT_S1(param_deserializer);

  Connections::In<ac_int<64, false>> CCS_INIT_S1(serial_params_in);

  Connections::In<ac_int<UNROLLFACTOR * Input::width, false>> CCS_INIT_S1(
      input_resp);
  Connections::In<ac_int<DWC_KERNEL_SIZE * Weight::width, false>> CCS_INIT_S1(
      weight_resp);
  Connections::In<ac_int<Output::width, false>> CCS_INIT_S1(bias_resp);

  Connections::Out<MemoryRequest> CCS_INIT_S1(weight_req);
  Connections::Out<MemoryRequest> CCS_INIT_S1(bias_req);
  Connections::Out<MemoryRequest> CCS_INIT_S1(input_req);
#if SUPPORT_MX
  Connections::Out<MemoryRequest> CCS_INIT_S1(input_scale_req);
  Connections::Out<MemoryRequest> CCS_INIT_S1(weight_scale_req);
  Connections::In<ac_int<SCALE_DATATYPE::width, false>> CCS_INIT_S1(
      input_scale_resp);
  Connections::In<ac_int<DWC_KERNEL_SIZE * SCALE_DATATYPE::width, false>>
      CCS_INIT_S1(weight_scale_resp);
#endif

  Connections::Out<Pack1D<Output, OutputWidth>> CCS_INIT_S1(dwc_output);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      dwc_output_address);

  Connections::SyncOut CCS_INIT_S1(start);
  Connections::SyncOut CCS_INIT_S1(done);

  Connections::Combinational<DwCParams> CCS_INIT_S1(params_in);
  Connections::Combinational<DwCParams> CCS_INIT_S1(input_params);
  Connections::Combinational<DwCParams> CCS_INIT_S1(output_params);
  Connections::Combinational<DwCParams> CCS_INIT_S1(weight_params);
  Connections::Combinational<DwCParams> CCS_INIT_S1(input_buffer_params);
  Connections::Combinational<DwCParams> CCS_INIT_S1(execution_params);

  Connections::Combinational<ac_int<1, false>> CCS_INIT_S1(dwc_kernel_exe);
  Connections::Combinational<ac_int<1, false>> CCS_INIT_S1(dwc_kernel_exe_run);
  Connections::Combinational<ac_int<1, false>> CCS_INIT_S1(dwc_update_weight);
  Connections::Combinational<Pack1D<ac_int<1, false>, DWC_KERNEL_SIZE>>
      CCS_INIT_S1(dwc_elem_valid_out);

  Connections::Combinational<Pack1D<Input, DWC_KERNEL_SIZE>>
      mat_input[UNROLLFACTOR];
  Connections::Combinational<Pack1D<Weight, DWC_KERNEL_SIZE>>
      mat_weight[UNROLLFACTOR];
#if SUPPORT_MX
  Connections::Combinational<Pack1D<SCALE_DATATYPE, DWC_KERNEL_SIZE>>
      mat_input_scale[UNROLLFACTOR];
  Connections::Combinational<Pack1D<SCALE_DATATYPE, DWC_KERNEL_SIZE>>
      mat_weight_scale[UNROLLFACTOR];
  Connections::Combinational<ac_int<1, false>> CCS_INIT_S1(dwc_use_mx);
#endif
  Connections::Combinational<Output> mat_bias[UNROLLFACTOR];
  Connections::Combinational<ac_int<1, false>> mat_update_weight[UNROLLFACTOR];
  Connections::Combinational<Pack1D<ac_int<1, false>, DWC_KERNEL_SIZE>>
      mat_weight_mask[UNROLLFACTOR];
  Connections::Combinational<Output> mat_output[UNROLLFACTOR];

  ac_int<UNROLLFACTOR * Input::width, false> input_buffer[DWC_WIDTH]
                                                         [DWC_KERNEL_DIM - 1];
  ac_int<UNROLLFACTOR * Input::width, false>
      input_shift_old_reg[DWC_KERNEL_DIM - 1][DWC_KERNEL_DIM - 1];
  ac_int<UNROLLFACTOR * Input::width, false>
      input_shift_new_reg[DWC_KERNEL_DIM];
  ac_int<8, false> write_addr_shift_reg[DWC_KERNEL_DIM]
                                       [2];  // support up to 256, x y
  ac_int<1, false> write_addr_valid_shift_reg[DWC_KERNEL_DIM];
#if SUPPORT_MX
  ac_int<SCALE_DATATYPE::width, false> input_scale_buffer[DWC_WIDTH]
                                                         [DWC_KERNEL_DIM - 1];
  ac_int<SCALE_DATATYPE::width, false>
      input_scale_shift_old_reg[DWC_KERNEL_DIM - 1][DWC_KERNEL_DIM - 1];
  ac_int<SCALE_DATATYPE::width, false>
      input_scale_shift_new_reg[DWC_KERNEL_DIM];
#endif

#ifdef __SYNTHESIS__
  MulAddTree<Input, Weight, Psum, Output> mulAddTree[UNROLLFACTOR];
#endif

  static constexpr int LOOP_WIDTH = 10;

  SC_CTOR(DwCUnit) {
    param_deserializer.clk(clk);
    param_deserializer.rstn(rstn);
    param_deserializer.serial_params_in(serial_params_in);
    param_deserializer.dwc_params_out(params_in);

    MulAddTree<Input, Weight, Psum, Output>* mul_add_tree_ptr[UNROLLFACTOR];

    for (int i = 0; i < UNROLLFACTOR; i++) {
#ifdef __SYNTHESIS__
      mul_add_tree_ptr[i] = &mulAddTree[i];
#else
      mul_add_tree_ptr[i] = new MulAddTree<Input, Weight, Psum, Output>(
          sc_gen_unique_name("MulAddTree"));
#endif
      mul_add_tree_ptr[i]->clk(clk);
      mul_add_tree_ptr[i]->rstn(rstn);
      mul_add_tree_ptr[i]->weight_in(mat_weight[i]);
      mul_add_tree_ptr[i]->input_in(mat_input[i]);
#if SUPPORT_MX
      mul_add_tree_ptr[i]->weight_scale_in(mat_weight_scale[i]);
      mul_add_tree_ptr[i]->input_scale_in(mat_input_scale[i]);
#endif
      mul_add_tree_ptr[i]->update_weight(mat_update_weight[i]);
      mul_add_tree_ptr[i]->weight_mask(mat_weight_mask[i]);
      mul_add_tree_ptr[i]->bias_in(mat_bias[i]);
      mul_add_tree_ptr[i]->adder_out(mat_output[i]);
    }

    SC_THREAD(read_params);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(input_addr_gen);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(weight_bias_addr_gen);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(fill_weight_bias);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(input_buffer_write_addr_gen);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(out_addr_gen);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(get_output);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(kernel_execution_ctrl);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void read_params() {
    params_in.ResetRead();
    input_params.ResetWrite();
    output_params.ResetWrite();
    weight_params.ResetWrite();
    input_buffer_params.ResetWrite();
    execution_params.ResetWrite();

    start.Reset();

    wait();

    while (true) {
      const DwCParams params = params_in.Pop();

      start.SyncPush();

      input_params.Push(params);
      output_params.Push(params);
      weight_params.Push(params);
      input_buffer_params.Push(params);
      execution_params.Push(params);
    }
  }

  void input_addr_gen() {  // input addr gen
    input_req.Reset();
    input_params.ResetRead();
#if SUPPORT_MX
    input_scale_req.Reset();
#endif

    wait();

    while (true) {
      const DwCParams params = input_params.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][3];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][3];
      ac_int<LOOP_WIDTH, false> input_bounds[3];  // Y X C
      ac_int<7, true> padding[2][2];              // Y X
      ac_int<3, false> block_size;
      ac_int<LOOP_WIDTH, false> block_num;
      ac_int<1, false> use_mx;

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_bounds[i][j] = params.loops[i][j];
        }
      }
#pragma hls_unroll yes
      for (int i = 0; i < 3; i++) {
        input_bounds[i] = params.bounds[i];
      }
#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 2; j++) {
          padding[i][j] = params.padding[i][j];
        }
      }
      block_size = params.block_size;
      use_mx = params.use_mx;
      block_num = (input_bounds[2] + (1 << block_size) - 1) >> block_size;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][2] = 0;
           loop_counters[0][2] < loop_bounds[0][2];  // C1
           loop_counters[0][2]++) {
        for (loop_counters[0][1] = 0;
             loop_counters[0][1] < loop_bounds[0][1];  // X1
             loop_counters[0][1]++) {
          for (loop_counters[0][0] = 0;
               loop_counters[0][0] < loop_bounds[0][0];  // Y1 -> no tiling on Y
               loop_counters[0][0]++) {
            // everything below is for the register array
            ac_int<LOOP_WIDTH, false> Y1 = loop_counters[0][0];
            ac_int<LOOP_WIDTH, false> X1 = loop_counters[0][1];
            ac_int<LOOP_WIDTH, false> C1 = loop_counters[0][2];
            for (loop_counters[1][1] = 0;
                 loop_counters[1][1] < loop_bounds[1][1] + 2;  // X0
                 loop_counters[1][1]++) {
              ac_int<LOOP_WIDTH, false> X0 = loop_counters[1][1];
              ac_int<16, false> C_g = C1 * loop_bounds[1][2];
              ac_int<16, false> X_g = X0 + X1 * loop_bounds[1][1];

              if (X_g < padding[1][0]) {
                continue;  // skip if X_g is less than left padding
              }
              // we only need to fetch non-zero padding in the x dimension
              else {  // only push if not right padding
                if (X_g < input_bounds[1] + padding[1][0]) {
                  ac_int<32, false> address =
                      (Y1 * input_bounds[1] + X_g - padding[1][0]) *
                          input_bounds[2] +
                      C_g;
                  MemoryRequest request = {
                      params.input_offset + address * Input::width / 8,
                      UNROLLFACTOR * Input::width / 8};
                  input_req.Push(request);
#if SUPPORT_MX
                  if (use_mx) {
                    ac_int<32, false> address_scale =
                        (Y1 * input_bounds[1] + X_g - padding[1][0]) *
                            block_num +
                        (C_g >> block_size);
                    MemoryRequest request_scale = {
                        params.input_scale_offset +
                            address_scale * SCALE_DATATYPE::width / 8,
                        SCALE_DATATYPE::width / 8};
                    input_scale_req.Push(request_scale);
                  }
#endif
                }
              }

              if (loop_counters[1][1] >= loop_bounds[1][1] + 1 ||
                  X_g >= input_bounds[1] + padding[1][0] - 1) {
                break;
              }
            }
            if (loop_counters[0][0] >= loop_bounds[0][0] - 1) {
              break;
            }
          }
          if (loop_counters[0][1] >= loop_bounds[0][1] - 1) {
            break;
          }
        }
        if (loop_counters[0][2] >= loop_bounds[0][2] - 1) {
          break;
        }
      }
    }
  }

  void weight_bias_addr_gen() {
    weight_req.Reset();
    bias_req.Reset();
    weight_params.ResetRead();
#if SUPPORT_MX
    weight_scale_req.Reset();
#endif

    wait();

    while (true) {
      const DwCParams params = weight_params.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][3];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][3];
      ac_int<1, false> use_mx;

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_bounds[i][j] = params.loops[i][j];
        }
      }
      use_mx = params.use_mx;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][2] = 0;
           loop_counters[0][2] < loop_bounds[0][2];  // C1
           loop_counters[0][2]++) {
        for (loop_counters[1][2] = 0;
             loop_counters[1][2] < loop_bounds[1][2];  // C0
             loop_counters[1][2]++) {
          ac_int<32, false> C =
              loop_counters[0][2] * loop_bounds[1][2] + loop_counters[1][2];
          MemoryRequest weight_request = {
              params.weight_offset + C * DWC_KERNEL_SIZE * Weight::width / 8,
              DWC_KERNEL_SIZE * Weight::width / 8};
          weight_req.Push(weight_request);
#if SUPPORT_MX
          if (use_mx) {
            MemoryRequest weight_scale_request = {
                params.weight_scale_offset +
                    C * DWC_KERNEL_SIZE * SCALE_DATATYPE::width / 8,
                DWC_KERNEL_SIZE * SCALE_DATATYPE::width / 8};
            weight_scale_req.Push(weight_scale_request);
          }
#endif
          MemoryRequest bias_request = {
              params.bias_offset + C * Output::width / 8, Output::width / 8};
          bias_req.Push(bias_request);

          if (loop_counters[1][2] >= loop_bounds[1][2] - 1) {
            break;
          }
        }
        if (loop_counters[0][2] >= loop_bounds[0][2] - 1) {
          break;
        }
      }
    }
  }

  void fill_weight_bias() {
    weight_resp.Reset();
    bias_resp.Reset();
    dwc_update_weight.ResetRead();
    dwc_kernel_exe.ResetRead();
#if SUPPORT_MX
    weight_scale_resp.Reset();
    dwc_use_mx.ResetRead();
#endif

    for (int i = 0; i < UNROLLFACTOR; i++) {
      mat_weight[i].ResetWrite();
      mat_bias[i].ResetWrite();
      mat_update_weight[i].ResetWrite();
#if SUPPORT_MX
      mat_weight_scale[i].ResetWrite();
#endif
    }

    wait();

    ac_int<1, false> update_weight;
    ac_int<1, false> kernel_exe;
    ac_int<1, false> kernel_exe_first = 0;  // the first exe per C0
    ac_int<1, false> use_mx = 0;

    Pack1D<Weight, DWC_KERNEL_SIZE> weight_zero;
#pragma hls_unroll yes
    for (int i = 0; i < DWC_KERNEL_SIZE; i++) {
      weight_zero[i] = Weight::zero();
    }
#if SUPPORT_MX
    Pack1D<SCALE_DATATYPE, DWC_KERNEL_SIZE> weight_scale_one;
#pragma hls_unroll yes
    for (int i = 0; i < DWC_KERNEL_SIZE; i++) {
      weight_scale_one[i] = SCALE_DATATYPE::one();
    }
#endif

    ac_int<10, false> fill_weight_bias_counter = UNROLLFACTOR;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      if (fill_weight_bias_counter >= UNROLLFACTOR) {  // Execution
        update_weight = dwc_update_weight.Pop();
        kernel_exe = dwc_kernel_exe.Pop();
#if SUPPORT_MX
        use_mx = dwc_use_mx.Pop();
#endif

        if (update_weight) {
          assert(kernel_exe == 0);
          fill_weight_bias_counter = 0;
          kernel_exe_first = 1;
          continue;
        }

        if (kernel_exe) {
          if (kernel_exe_first == 1) {
#pragma hls_unroll yes
            for (int i = 0; i < UNROLLFACTOR; i++) {
              mat_update_weight[i].Push(1);
            }
            kernel_exe_first = 0;
          } else {
#pragma hls_unroll yes
            for (int i = 0; i < UNROLLFACTOR; i++) {
              mat_weight[i].Push(weight_zero);
              mat_bias[i].Push(Output::zero());
              mat_update_weight[i].Push(0);
#if SUPPORT_MX
              mat_weight_scale[i].Push(weight_scale_one);
#endif
            }
          }
        }
      } else {
        ac_int<DWC_KERNEL_SIZE * Weight::width, false> weight =
            weight_resp.Pop();
        Pack1D<Weight, DWC_KERNEL_SIZE> weight_tmp;
#if SUPPORT_MX
        ac_int<DWC_KERNEL_SIZE * SCALE_DATATYPE::width, false> weight_scale;
        Pack1D<SCALE_DATATYPE, DWC_KERNEL_SIZE> weight_scale_tmp;
        if (use_mx) {
          weight_scale = weight_scale_resp.Pop();
        }
#endif
        Output bias;
        bias.set_bits(bias_resp.Pop());

#pragma hls_unroll yes
        for (int j = 0; j < DWC_KERNEL_SIZE; j++) {
          weight_tmp[j].set_bits(
              weight.template slc<Weight::width>(j * Weight::width));
#if SUPPORT_MX
          if (use_mx) {
            weight_scale_tmp[j].set_bits(
                weight_scale.template slc<SCALE_DATATYPE::width>(
                    j * SCALE_DATATYPE::width));
          } else {
            weight_scale_tmp[j] = SCALE_DATATYPE::one();
          }
#endif
        }
        mat_weight[fill_weight_bias_counter].Push(weight_tmp);
        mat_bias[fill_weight_bias_counter].Push(bias);
#if SUPPORT_MX
        mat_weight_scale[fill_weight_bias_counter].Push(weight_scale_tmp);
#endif
        fill_weight_bias_counter++;
      }
    }
  }

  void input_buffer_write_addr_gen()  // the dwc execution
  {
    input_buffer_params.ResetRead();
    input_resp.Reset();
    dwc_elem_valid_out.ResetRead();
    dwc_kernel_exe_run.ResetRead();
#if SUPPORT_MX
    input_scale_resp.Reset();
#endif

    for (int i = 0; i < UNROLLFACTOR; i++) {
      mat_input[i].ResetWrite();
      mat_weight_mask[i].ResetWrite();
#if SUPPORT_MX
      mat_input_scale[i].ResetWrite();
#endif
    }

    for (int i = 0; i < DWC_KERNEL_DIM; i++) {
      write_addr_shift_reg[i][0] = 0;
      write_addr_shift_reg[i][1] = 0;
      write_addr_valid_shift_reg[i] = 0;
    }

    wait();

    while (true) {
      const DwCParams params = input_buffer_params.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][3];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][3];
      ac_int<LOOP_WIDTH, false> input_bounds[3];  // Y X C
      ac_int<7, true> padding[2][2];              // Y X

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_bounds[i][j] = params.loops[i][j];
        }
      }
#pragma hls_unroll yes
      for (int i = 0; i < 3; i++) {
        input_bounds[i] = params.bounds[i];
      }
#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 2; j++) {
          padding[i][j] = params.padding[i][j];
        }
      }
      ac_int<1, false> fast_forward_mode = params.fast_forward_mode;
      ac_int<1, false> use_mx = params.use_mx;

      Pack1D<Input, DWC_KERNEL_SIZE> input_per_kernel;
      ac_int<UNROLLFACTOR * Input::width, false> ff_reg;
#if SUPPORT_MX
      Pack1D<SCALE_DATATYPE, DWC_KERNEL_SIZE> input_scale_per_kernel;
      ac_int<SCALE_DATATYPE::width, false> ff_scale_reg;
#endif
      ac_int<1, false> ff_idx;
      ac_int<1, false> ff_vld = 0;

      ac_int<LOOP_WIDTH, false> y_start_waiting;
      if (padding[0][0] > DWC_KERNEL_DIM - 1) {
        y_start_waiting = padding[0][0] - (DWC_KERNEL_DIM - 1);
      } else {
        y_start_waiting = 0;
      }
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][2] = 0;
           loop_counters[0][2] < loop_bounds[0][2];  // C1
           loop_counters[0][2]++) {
        for (loop_counters[0][1] = 0;
             loop_counters[0][1] < loop_bounds[0][1];  // X1
             loop_counters[0][1]++) {
          ac_int<4, false> input_buffer_y_counter = 0;
          ac_int<1, false> input_buffer_idx =
              0;  // The virtual top row, since the new_reg is always the last
                  // row

          for (loop_counters[0][0] = 0;
               loop_counters[0][0] <
               loop_bounds[0][0] + y_start_waiting +
                   padding[0][1];  // Y1 -> no tiling on Y, extra execution for
                                   // the last row
               loop_counters[0][0]++) {
            // everything below is for the register array
            ac_int<LOOP_WIDTH, false> Y1 = loop_counters[0][0];
            ac_int<LOOP_WIDTH, false> X1 = loop_counters[0][1];
            ac_int<LOOP_WIDTH, false> C1 = loop_counters[0][2];
            for (loop_counters[1][1] = 0;
                 loop_counters[1][1] < loop_bounds[1][1] + 2;  // X0
                 loop_counters[1][1]++) {
              ac_int<LOOP_WIDTH, false> X0 = loop_counters[1][1];
              ac_int<16, false> C_g = C1 * loop_bounds[1][2];
              ac_int<16, false> X_g = X0 + X1 * loop_bounds[1][1];

              ac_int<1, false> kernel_exe = dwc_kernel_exe_run.Pop();
              Pack1D<ac_int<1, false>, DWC_KERNEL_SIZE> elem_valid =
                  dwc_elem_valid_out.Pop();

              if ((Y1 < loop_bounds[0][0] + y_start_waiting) &&
                  (Y1 >= y_start_waiting) &&
                  (X_g < input_bounds[1] + padding[1][0]) &&
                  (X_g >= padding[1][0])) {
                ac_int<UNROLLFACTOR * Input::width, false> input =
                    input_resp.Pop();
                input_shift_new_reg[DWC_KERNEL_DIM - 1] = input;
                write_addr_shift_reg[DWC_KERNEL_DIM - 1][0] = X0;
                write_addr_shift_reg[DWC_KERNEL_DIM - 1][1] = input_buffer_idx;
                write_addr_valid_shift_reg[DWC_KERNEL_DIM - 1] = 1;
#if SUPPORT_MX
                if (use_mx) {
                  ac_int<SCALE_DATATYPE::width, false> input_scale =
                      input_scale_resp.Pop();
                  input_scale_shift_new_reg[DWC_KERNEL_DIM - 1] = input_scale;
                }
#endif
              } else {
                write_addr_valid_shift_reg[DWC_KERNEL_DIM - 1] = 0;
              }

              ac_int<UNROLLFACTOR * Input::width, false>
                  input_buffer_read[DWC_KERNEL_DIM - 1];
#if SUPPORT_MX
              ac_int<SCALE_DATATYPE::width, false>
                  input_scale_buffer_read[DWC_KERNEL_DIM - 1];
#endif
#pragma hls_unroll yes
              for (int i = 0; i < DWC_KERNEL_DIM - 1; i++) {
                if (X1 == loop_bounds[0][1] - 1 && fast_forward_mode &&
                    i == ff_idx && ff_vld) {
                  input_buffer_read[ff_idx] = ff_reg;
#if SUPPORT_MX
                  input_scale_buffer_read[ff_idx] = ff_scale_reg;
#endif
                } else {
                  input_buffer_read[i] = input_buffer[X0][i];
#if SUPPORT_MX
                  input_scale_buffer_read[i] = input_scale_buffer[X0][i];
#endif
                }
              }

              ff_vld = 0;

              if (kernel_exe) {
#pragma hls_unroll yes
                for (int i = 0; i < UNROLLFACTOR; i++) {
#pragma hls_unroll yes
                  for (int r = 0; r < DWC_KERNEL_DIM; r++) {
#pragma hls_unroll yes
                    for (int c = 0; c < DWC_KERNEL_DIM; c++) {
                      if (elem_valid[r * DWC_KERNEL_DIM + c]) {
                        ac_int<1, false> input_buffer_idx_curr =
                            input_buffer_idx + r;
                        if (r == DWC_KERNEL_DIM - 1) {  // TODO
                          input_per_kernel[r * DWC_KERNEL_DIM + c].set_bits(
                              input_shift_new_reg[c].template slc<Input::width>(
                                  i * Input::width));
                        } else {
                          if (c != DWC_KERNEL_DIM - 1) {
                            input_per_kernel[r * DWC_KERNEL_DIM + c].set_bits(
                                input_shift_old_reg[input_buffer_idx_curr][c]
                                    .template slc<Input::width>(i *
                                                                Input::width));
                          } else {
                            input_per_kernel[r * DWC_KERNEL_DIM + c].set_bits(
                                input_buffer_read[input_buffer_idx_curr]
                                    .template slc<Input::width>(i *
                                                                Input::width));
                          }
                        }
                      } else {
                        input_per_kernel[r * DWC_KERNEL_DIM + c] =
                            Input::zero();
                      }
                    }
                  }

                  mat_input[i].Push(input_per_kernel);
                  mat_weight_mask[i].Push(elem_valid);
                }
#if SUPPORT_MX
#pragma hls_unroll yes
                for (int r = 0; r < DWC_KERNEL_DIM; r++) {
#pragma hls_unroll yes
                  for (int c = 0; c < DWC_KERNEL_DIM; c++) {
                    if (use_mx) {
                      if (elem_valid[r * DWC_KERNEL_DIM + c]) {
                        ac_int<1, false> input_buffer_idx_curr =
                            input_buffer_idx + r;
                        if (r == DWC_KERNEL_DIM - 1) {
                          input_scale_per_kernel[r * DWC_KERNEL_DIM + c]
                              .set_bits(
                                  input_scale_shift_new_reg[c]
                                      .template slc<SCALE_DATATYPE::width>(0));
                        } else {
                          if (c != DWC_KERNEL_DIM - 1) {
                            input_scale_per_kernel[r * DWC_KERNEL_DIM + c]
                                .set_bits(
                                    input_scale_shift_old_reg
                                        [input_buffer_idx_curr][c]
                                            .template slc<
                                                SCALE_DATATYPE::width>(0));
                          } else {
                            input_scale_per_kernel[r * DWC_KERNEL_DIM + c]
                                .set_bits(
                                    input_scale_buffer_read
                                        [input_buffer_idx_curr]
                                            .template slc<
                                                SCALE_DATATYPE::width>(0));
                          }
                        }
                      } else {
                        input_scale_per_kernel[r * DWC_KERNEL_DIM + c] =
                            SCALE_DATATYPE::zero();
                      }
                    } else {
                      input_scale_per_kernel[r * DWC_KERNEL_DIM + c] =
                          SCALE_DATATYPE::one();
                    }
                  }
                }
#pragma hls_unroll yes
                for (int i = 0; i < UNROLLFACTOR; i++) {
                  mat_input_scale[i].Push(input_scale_per_kernel);
                }
#endif
              }

              // write back the input_shift_reg
              if (write_addr_valid_shift_reg[0] == 1) {
                input_buffer[write_addr_shift_reg[0][0]]
                            [write_addr_shift_reg[0][1]] =
                                input_shift_new_reg[0];
#if SUPPORT_MX
                input_scale_buffer[write_addr_shift_reg[0][0]]
                                  [write_addr_shift_reg[0][1]] =
                                      input_scale_shift_new_reg[0];
#endif
                if (X1 == loop_bounds[0][1] - 1 && fast_forward_mode) {
                  ff_reg = input_shift_new_reg[0];
#if SUPPORT_MX
                  ff_scale_reg = input_scale_shift_new_reg[0];
#endif
                  ff_idx = write_addr_shift_reg[0][1];
                  ff_vld = 1;
                }
              }

              // shift the registers
#pragma hls_unroll yes
              for (int i = 0; i < DWC_KERNEL_DIM - 1; i++) {  // row
#pragma hls_unroll yes
                for (int j = 0; j < DWC_KERNEL_DIM - 2; j++) {  // column
                  input_shift_old_reg[i][j] = input_shift_old_reg[i][j + 1];
#if SUPPORT_MX
                  input_scale_shift_old_reg[i][j] =
                      input_scale_shift_old_reg[i][j + 1];
#endif
                }
              }
#pragma hls_unroll yes
              for (int i = 0; i < DWC_KERNEL_DIM - 1; i++) {  // row
                ac_int<1, false> input_buffer_idx_curr = input_buffer_idx + i;
                input_shift_old_reg[input_buffer_idx_curr][DWC_KERNEL_DIM - 2] =
                    input_buffer_read[input_buffer_idx_curr];
#if SUPPORT_MX
                input_scale_shift_old_reg
                    [input_buffer_idx_curr][DWC_KERNEL_DIM - 2] =
                        input_scale_buffer_read[input_buffer_idx_curr];
#endif
              }
#pragma hls_unroll yes
              for (int i = 0; i < DWC_KERNEL_DIM - 1; i++) {  // column
                input_shift_new_reg[i] = input_shift_new_reg[i + 1];
#if SUPPORT_MX
                input_scale_shift_new_reg[i] = input_scale_shift_new_reg[i + 1];
#endif
                write_addr_shift_reg[i][0] = write_addr_shift_reg[i + 1][0];
                write_addr_shift_reg[i][1] = write_addr_shift_reg[i + 1][1];
                write_addr_valid_shift_reg[i] =
                    write_addr_valid_shift_reg[i + 1];
              }

              if (loop_counters[1][1] >= loop_bounds[1][1] + 1 ||
                  X_g >= input_bounds[1] + padding[1][0] + padding[1][1] - 1) {
                break;
              }
            }

            if (input_buffer_y_counter == DWC_KERNEL_DIM - 1) {
              input_buffer_y_counter = 0;
            } else {
              input_buffer_y_counter++;
            }
            input_buffer_idx++;

            if (loop_counters[0][0] >= loop_bounds[0][0] + y_start_waiting +
                                           padding[0][1] - 1) {  // Careful here
              break;
            }
          }
          if (loop_counters[0][1] >= loop_bounds[0][1] - 1) {
            break;
          }
        }
        if (loop_counters[0][2] >= loop_bounds[0][2] - 1) {
          break;
        }
      }
    }
  }

  void out_addr_gen()  // Buffer read & write addr gen, output addr gen
  {
    dwc_output_address.Reset();
    output_params.ResetRead();

    done.Reset();

    wait();

    while (true) {
      const DwCParams params = output_params.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][3];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][3];
      ac_int<LOOP_WIDTH, false> input_bounds[3];  // Y X C
      ac_int<LOOP_WIDTH, false> Y_T;
      ac_int<LOOP_WIDTH, false> X_g_T;
      ac_int<LOOP_WIDTH, false> X0_T;

      ac_int<2, false> stride;

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_bounds[i][j] = params.loops[i][j];
        }
      }
#pragma hls_unroll yes
      for (int i = 0; i < 3; i++) {
        input_bounds[i] = params.bounds[i];
      }

      Y_T = params.outloops[0];
      X_g_T = params.outloops[1];
      X0_T = params.outloops[2];

      ac_int<1, false> output_end;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][2] = 0;
           loop_counters[0][2] < loop_bounds[0][2];  // C1
           loop_counters[0][2]++) {
        for (loop_counters[0][1] = 0;
             loop_counters[0][1] < loop_bounds[0][1];  // X1
             loop_counters[0][1]++) {
          for (loop_counters[0][0] = 0;
               loop_counters[0][0] < Y_T;  // Y1 -> no tiling on Y
               loop_counters[0][0]++) {
            // everything below is for the register array
            ac_int<LOOP_WIDTH, false> Y1 = loop_counters[0][0];
            ac_int<LOOP_WIDTH, false> X1 = loop_counters[0][1];
            ac_int<LOOP_WIDTH, false> C1 = loop_counters[0][2];
            for (loop_counters[1][1] = 0; loop_counters[1][1] < X0_T;  // X0
                 loop_counters[1][1]++) {
              ac_int<LOOP_WIDTH, false> X0 = loop_counters[1][1];
              ac_int<16, false> C_g = C1 * loop_bounds[1][2];
              ac_int<16, false> X_g = X0 + X1 * X0_T;

              ac_int<32, false> address =
                  (Y1 * X_g_T + X_g) * input_bounds[2] + C_g;
              ac_int<ADDRESS_WIDTH, false> output_address = address;
              dwc_output_address.Push(output_address);

              if (loop_counters[1][1] == X0_T - 1 || X_g == X_g_T - 1) {
                break;
              }
            }
            if (loop_counters[0][0] >= Y_T - 1) {
              break;
            }
          }
          if (loop_counters[0][1] >= loop_bounds[0][1] - 1) {
            break;
          }
        }
        if (loop_counters[0][2] >= loop_bounds[0][2] - 1) {
          break;
        }
      }

      done.SyncPush();
    }
  }

  void get_output() {
    dwc_output.Reset();
    for (int i = 0; i < UNROLLFACTOR; i++) {
      mat_output[i].ResetRead();
    }

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      Pack1D<Output, OutputWidth> output;
#pragma hls_unroll yes
      for (int i = 0; i < OutputWidth; i++) {
        if (i < UNROLLFACTOR) {
          output[i] = mat_output[i].Pop();
        } else {
          output[i] = Output::zero();
        }
      }
      dwc_output.Push(output);
    }
  }

  void kernel_execution_ctrl()  // Buffer read & write addr gen, output addr gen
  {
    execution_params.ResetRead();
    dwc_update_weight.ResetWrite();
    dwc_kernel_exe.ResetWrite();
    dwc_kernel_exe_run.ResetWrite();
    dwc_elem_valid_out.ResetWrite();
#if SUPPORT_MX
    dwc_use_mx.ResetWrite();
#endif

    wait();

    while (true) {
      const DwCParams params = execution_params.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][3];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][3];
      ac_int<LOOP_WIDTH, false> input_bounds[3];  // Y X C

      ac_int<4, false> stride;
      ac_int<4, false> stride_x;
      ac_int<4, false> stride_y;
      ac_int<7, true> padding[2][2];  // Y X

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_bounds[i][j] = params.loops[i][j];
        }
      }
#pragma hls_unroll yes
      for (int i = 0; i < 3; i++) {
        input_bounds[i] = params.bounds[i];
      }
#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 2; j++) {
          padding[i][j] = params.padding[i][j];
        }
      }
      stride = params.stride;
      ac_int<LOOP_WIDTH, false> y_read_waiting;
      ac_int<2, false> y_exe_delay;
      if (padding[0][0] > DWC_KERNEL_DIM - 1) {
        y_read_waiting = padding[0][0] - (DWC_KERNEL_DIM - 1);
        y_exe_delay = 0;
      } else {
        y_read_waiting = 0;
        y_exe_delay = DWC_KERNEL_DIM - 1 - padding[0][0];
      }
      Pack1D<ac_int<1, false>, DWC_KERNEL_SIZE> elem_valid;
      Pack1D<ac_int<1, false>, DWC_KERNEL_SIZE> elem_valid_zero;
      ac_int<1, false> update_weight;
      ac_int<1, false> kernel_exe;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][2] = 0;
           loop_counters[0][2] < loop_bounds[0][2];  // C1
           loop_counters[0][2]++) {
        for (loop_counters[0][1] = 0;
             loop_counters[0][1] < loop_bounds[0][1];  // X1
             loop_counters[0][1]++) {
          stride_y = 0;

          for (loop_counters[0][0] = 0;
               loop_counters[0][0] <
               loop_bounds[0][0] + y_read_waiting +
                   padding[0][1];  // Y1 -> no tiling on Y, extra execution for
                                   // the last row
               loop_counters[0][0]++) {
            // everything below is for the register array
            ac_int<LOOP_WIDTH, false> Y1 = loop_counters[0][0];
            ac_int<LOOP_WIDTH, false> X1 = loop_counters[0][1];
            ac_int<LOOP_WIDTH, false> C1 = loop_counters[0][2];
            stride_x = 0;

            for (loop_counters[1][1] = 0;
                 loop_counters[1][1] < loop_bounds[1][1] + 2;
                 loop_counters[1][1]++) {
              ac_int<LOOP_WIDTH, false> X0 = loop_counters[1][1];
              ac_int<16, false> C_g = C1 * loop_bounds[1][2];
              ac_int<16, false> X_g = X0 + X1 * loop_bounds[1][1];

              ac_int<LOOP_WIDTH, false> X_g_true = X_g - 2;
              ac_int<LOOP_WIDTH, false> Y_g_true = Y1 - y_exe_delay;

              if (Y1 == 0 && X0 == 0 && X1 == 0) {
                update_weight = 1;
                dwc_update_weight.Push(update_weight);
              } else {
                update_weight = 0;
                dwc_update_weight.Push(update_weight);
              }

              if ((Y1 < y_exe_delay || X0 <= 1) ||
                  (stride_x != 0 || stride_y != 0)) {
                kernel_exe = 0;
                dwc_kernel_exe.Push(kernel_exe);
                dwc_kernel_exe_run.Push(kernel_exe);
              } else {
                kernel_exe = 1;
                dwc_kernel_exe.Push(kernel_exe);
                dwc_kernel_exe_run.Push(kernel_exe);
              }
#if SUPPORT_MX
              dwc_use_mx.Push(params.use_mx);
#endif

              // spdlog::debug(
              //     "DwCUnit: kernel execution ctrl: Y1: {}, X0: {}, X_g: {}, "
              //     "C1: {}, C_g: {}, execute {}, update {}\n",
              //     Y1.to_int(), X0.to_int(), X_g.to_int(), C1.to_int(),
              //     C_g.to_int(), kernel_exe.to_int(), update_weight.to_int());

              if (Y1 >= y_exe_delay && X0 > 1) {
#pragma hls_unroll yes
                for (int i = 0; i < DWC_KERNEL_DIM; i++) {
#pragma hls_unroll yes
                  for (int j = 0; j < DWC_KERNEL_DIM; j++) {
                    if (X_g_true + i < padding[1][0]) {
                      elem_valid[i + j * DWC_KERNEL_DIM] = 0;
                    } else if (X_g_true + i >=
                               input_bounds[1] + padding[1][0]) {
                      elem_valid[i + j * DWC_KERNEL_DIM] = 0;
                    } else if (Y_g_true + j < padding[0][0]) {
                      elem_valid[i + j * DWC_KERNEL_DIM] = 0;
                    } else if (Y_g_true + j >=
                               loop_bounds[0][0] + padding[0][0]) {
                      elem_valid[i + j * DWC_KERNEL_DIM] = 0;
                    } else {
                      elem_valid[i + j * DWC_KERNEL_DIM] = 1;
                    }
                  }
                }
                dwc_elem_valid_out.Push(elem_valid);
              } else {
                dwc_elem_valid_out.Push(elem_valid_zero);
              }

              if (X0 > 1 && stride != 1) {
                if (stride_x == stride - 1) {
                  stride_x = 0;
                } else {
                  stride_x++;
                }
              }
              if (loop_counters[1][1] >= loop_bounds[1][1] + 1 ||
                  X_g >= input_bounds[1] + padding[1][0] + padding[1][1] - 1) {
                break;
              }
            }

            if (Y1 >= y_exe_delay && stride != 1) {
              if (stride_y == stride - 1) {
                stride_y = 0;
              } else {
                stride_y++;
              }
            }
            if (loop_counters[0][0] >= loop_bounds[0][0] + y_read_waiting +
                                           padding[0][1] - 1) {  // Careful here
              break;
            }
          }
          if (loop_counters[0][1] >= loop_bounds[0][1] - 1) {
            break;
          }
        }
        if (loop_counters[0][2] >= loop_bounds[0][2] - 1) {
          break;
        }
      }
    }
  }
};
