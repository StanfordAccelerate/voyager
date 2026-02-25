#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "ArchitectureParams.h"
#include "Params.h"
#include "ParamsDeserializer.h"
#include "Utils.h"
#include "connections/connections.h"

#pragma hls_design ccore
template <typename Input, typename Weight, typename Scale, typename Output>
Output multiply_accumulate(const Input input, const Weight weight,
                           const Scale weight_scale, const Output psum) {
  return psum + (Output)input * (Output)weight * (Output)weight_scale;
}

template <typename WeightTypeTuple, typename Input, typename Weight,
          typename Meta, typename Output, typename Scale, int port_width,
          int width, int bs, int vu_width>
struct SpMMUnit;

template <typename... WeightTypes, typename Input, typename Weight,
          typename Meta, typename Output, typename Scale, int port_width,
          int width, int bs, int vu_width>
struct SpMMUnit<std::tuple<WeightTypes...>, Input, Weight, Meta, Output, Scale,
                port_width, width, bs, vu_width> : public sc_module {
  static constexpr int LOOP_WIDTH = 16;
  using loop_t = ac_int<LOOP_WIDTH, false>;

#ifdef CLOCK_PERIOD
  static constexpr double clock_period = CLOCK_PERIOD;
#else
  static constexpr double clock_period = 5.0;  // Default to 5 ns if not defined
#endif

  static constexpr int FEEDBACK_DEPTH = (clock_period < 5) ? 6 : 4;
  static constexpr int FEEDBACK_LAST = FEEDBACK_DEPTH - 1;

  static constexpr int NUM_META = port_width / Meta::width;
  static constexpr int NUM_DATA = port_width / Input::width;

#if SUPPORT_MX
  static constexpr int SCALE_PORT_WIDTH = Scale::width * OC_DIMENSION;
  typedef ac_int<SCALE_PORT_WIDTH, false> SCALE_PORT_TYPE;
#endif

  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  MatrixParamsDeserializer<2, 1> CCS_INIT_S1(params_deserializer);
  Connections::In<ac_int<64, false>> CCS_INIT_S1(serial_params_in);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(params_in);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(
      fetch_input_indptr_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(
      process_input_indptr_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(
      fetch_input_indices_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(fetch_input_data_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(fetch_weight_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(process_weight_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(
      fetch_weight_scale_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(
      write_weight_scale_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(read_weight_scale_param);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(send_output_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(run_accumulation_param);

  Connections::Out<MemoryRequest> CCS_INIT_S1(input_indptr_req);
  Connections::In<ac_int<port_width, false>> CCS_INIT_S1(input_indptr_resp);
  Connections::Combinational<Pack1D<Meta, NUM_META>> CCS_INIT_S1(
      input_indptr_pack);

  // FIFOs to avoid stalling between rows
  Connections::Fifo<Meta, 2> CCS_INIT_S1(input_indptr_fifo);
  Connections::Fifo<Meta, 2> CCS_INIT_S1(fetch_input_indices_start_fifo);
  Connections::Fifo<Meta, 2> CCS_INIT_S1(fetch_input_indices_end_fifo);
  Connections::Fifo<Meta, 2> CCS_INIT_S1(process_input_indices_start_fifo);
  Connections::Fifo<Meta, 2> CCS_INIT_S1(process_input_indices_end_fifo);
  Connections::Fifo<Meta, 2> CCS_INIT_S1(fetch_input_data_start_fifo);
  Connections::Fifo<Meta, 2> CCS_INIT_S1(fetch_input_data_end_fifo);
  Connections::Fifo<Meta, 2> CCS_INIT_S1(process_input_data_start_fifo);
  Connections::Fifo<Meta, 2> CCS_INIT_S1(process_input_data_end_fifo);
  Connections::Fifo<Meta, 2> CCS_INIT_S1(fetch_weight_nnz_fifo);
  Connections::Fifo<Meta, 2> CCS_INIT_S1(process_weight_nnz_fifo);

  Connections::Combinational<Meta> CCS_INIT_S1(input_indptr_enq);
  Connections::Combinational<Meta> CCS_INIT_S1(input_indptr_deq);
  Connections::Combinational<Meta> CCS_INIT_S1(fetch_input_indices_start_enq);
  Connections::Combinational<Meta> CCS_INIT_S1(fetch_input_indices_start_deq);
  Connections::Combinational<Meta> CCS_INIT_S1(fetch_input_indices_end_enq);
  Connections::Combinational<Meta> CCS_INIT_S1(fetch_input_indices_end_deq);
  Connections::Combinational<Meta> CCS_INIT_S1(process_input_indices_start_enq);
  Connections::Combinational<Meta> CCS_INIT_S1(process_input_indices_start_deq);
  Connections::Combinational<Meta> CCS_INIT_S1(process_input_indices_end_enq);
  Connections::Combinational<Meta> CCS_INIT_S1(process_input_indices_end_deq);
  Connections::Combinational<Meta> CCS_INIT_S1(fetch_input_data_start_enq);
  Connections::Combinational<Meta> CCS_INIT_S1(fetch_input_data_start_deq);
  Connections::Combinational<Meta> CCS_INIT_S1(fetch_input_data_end_enq);
  Connections::Combinational<Meta> CCS_INIT_S1(fetch_input_data_end_deq);
  Connections::Combinational<Meta> CCS_INIT_S1(process_input_data_start_enq);
  Connections::Combinational<Meta> CCS_INIT_S1(process_input_data_start_deq);
  Connections::Combinational<Meta> CCS_INIT_S1(process_input_data_end_enq);
  Connections::Combinational<Meta> CCS_INIT_S1(process_input_data_end_deq);
  Connections::Combinational<Meta> CCS_INIT_S1(fetch_weight_nnz_enq);
  Connections::Combinational<Meta> CCS_INIT_S1(fetch_weight_nnz_deq);
  Connections::Combinational<Meta> CCS_INIT_S1(process_weight_nnz_enq);
  Connections::Combinational<Meta> CCS_INIT_S1(process_weight_nnz_deq);

  Connections::Out<MemoryRequest> CCS_INIT_S1(input_indices_req);
  Connections::In<ac_int<port_width, false>> CCS_INIT_S1(input_indices_resp);

  Connections::Fifo<Pack1D<Meta, NUM_META>, 2> CCS_INIT_S1(
      fetch_weight_input_indices_fifo);
  Connections::Combinational<Pack1D<Meta, NUM_META>> CCS_INIT_S1(
      fetch_weight_input_indices_enq);
  Connections::Combinational<Pack1D<Meta, NUM_META>> CCS_INIT_S1(
      fetch_weight_input_indices_deq);
#if SUPPORT_MX
  Connections::Fifo<Pack1D<Meta, NUM_META>, 2> CCS_INIT_S1(
      read_weight_scale_input_indices_fifo);
  Connections::Combinational<Pack1D<Meta, NUM_META>> CCS_INIT_S1(
      read_weight_scale_input_indices_enq);
  Connections::Combinational<Pack1D<Meta, NUM_META>> CCS_INIT_S1(
      read_weight_scale_input_indices_deq);
#endif

  Connections::Out<MemoryRequest> CCS_INIT_S1(input_data_req);
  Connections::In<ac_int<port_width, false>> CCS_INIT_S1(input_data_resp);

  Connections::Fifo<Pack1D<Input, NUM_DATA>, 2> CCS_INIT_S1(
      input_data_packed_fifo);
  Connections::Combinational<Pack1D<Input, NUM_DATA>> CCS_INIT_S1(
      input_data_packed_enq);
  Connections::Combinational<Pack1D<Input, NUM_DATA>> CCS_INIT_S1(
      input_data_packed_deq);
  Connections::Combinational<Input> CCS_INIT_S1(input_data);

  Connections::Out<MemoryRequest> CCS_INIT_S1(weight_req);
  Connections::In<ac_int<port_width, false>> CCS_INIT_S1(weight_resp);
  Connections::Combinational<Pack1D<Weight, width>> CCS_INIT_S1(weight_data);

#if SUPPORT_MX
  Connections::Out<MemoryRequest> CCS_INIT_S1(weight_scale_req);
  Connections::In<ac_int<port_width, false>> CCS_INIT_S1(weight_scale_resp);
  Connections::Combinational<SCALE_PORT_TYPE> CCS_INIT_S1(weight_scale_data);
  // TODO: parameterisze buffer depth
  DoubleBuffer<32, SCALE_DATATYPE::width * OC_DIMENSION> CCS_INIT_S1(
      weight_scale_buffer);
  Connections::Combinational<BufferWriteRequest<SCALE_PORT_TYPE>>
      weight_scale_write_req[2];
  Connections::Combinational<BufferReadRequest> weight_scale_read_req[2];
  Connections::Combinational<ac_int<SCALE_PORT_WIDTH, false>> CCS_INIT_S1(
      weight_scale_read_resp);
  Connections::Fifo<Meta, 2> CCS_INIT_S1(read_weight_scale_nnz_fifo);
  Connections::Combinational<Meta> CCS_INIT_S1(read_weight_scale_nnz_enq);
  Connections::Combinational<Meta> CCS_INIT_S1(read_weight_scale_nnz_deq);

#endif

  Connections::Combinational<Pack1D<Output, width>> CCS_INIT_S1(
      accumulation_out);
  Connections::Out<Pack1D<Output, vu_width>> CCS_INIT_S1(spmm_unit_output);

  Connections::SyncOut CCS_INIT_S1(start);
  Connections::SyncOut CCS_INIT_S1(done);

  SC_CTOR(SpMMUnit) {
    params_deserializer.clk(clk);
    params_deserializer.rstn(rstn);
    params_deserializer.serial_params_in(serial_params_in);
    params_deserializer.params_out[0](params_in);

#if SUPPORT_MX
    weight_scale_buffer.clk(clk);
    weight_scale_buffer.rstn(rstn);
    for (int i = 0; i < 2; i++) {
      weight_scale_buffer.write_request[i](weight_scale_write_req[i]);
      weight_scale_buffer.read_request[i](weight_scale_read_req[i]);
    }
    weight_scale_buffer.output(weight_scale_data);
#endif

    input_indptr_fifo.clk(clk);
    input_indptr_fifo.rst(rstn);
    input_indptr_fifo.enq(input_indptr_enq);
    input_indptr_fifo.deq(input_indptr_deq);

    fetch_input_indices_start_fifo.clk(clk);
    fetch_input_indices_start_fifo.rst(rstn);
    fetch_input_indices_start_fifo.enq(fetch_input_indices_start_enq);
    fetch_input_indices_start_fifo.deq(fetch_input_indices_start_deq);

    fetch_input_indices_end_fifo.clk(clk);
    fetch_input_indices_end_fifo.rst(rstn);
    fetch_input_indices_end_fifo.enq(fetch_input_indices_end_enq);
    fetch_input_indices_end_fifo.deq(fetch_input_indices_end_deq);

    process_input_indices_start_fifo.clk(clk);
    process_input_indices_start_fifo.rst(rstn);
    process_input_indices_start_fifo.enq(process_input_indices_start_enq);
    process_input_indices_start_fifo.deq(process_input_indices_start_deq);

    process_input_indices_end_fifo.clk(clk);
    process_input_indices_end_fifo.rst(rstn);
    process_input_indices_end_fifo.enq(process_input_indices_end_enq);
    process_input_indices_end_fifo.deq(process_input_indices_end_deq);

    fetch_input_data_start_fifo.clk(clk);
    fetch_input_data_start_fifo.rst(rstn);
    fetch_input_data_start_fifo.enq(fetch_input_data_start_enq);
    fetch_input_data_start_fifo.deq(fetch_input_data_start_deq);

    fetch_input_data_end_fifo.clk(clk);
    fetch_input_data_end_fifo.rst(rstn);
    fetch_input_data_end_fifo.enq(fetch_input_data_end_enq);
    fetch_input_data_end_fifo.deq(fetch_input_data_end_deq);

    process_input_data_start_fifo.clk(clk);
    process_input_data_start_fifo.rst(rstn);
    process_input_data_start_fifo.enq(process_input_data_start_enq);
    process_input_data_start_fifo.deq(process_input_data_start_deq);

    process_input_data_end_fifo.clk(clk);
    process_input_data_end_fifo.rst(rstn);
    process_input_data_end_fifo.enq(process_input_data_end_enq);
    process_input_data_end_fifo.deq(process_input_data_end_deq);

    fetch_weight_nnz_fifo.clk(clk);
    fetch_weight_nnz_fifo.rst(rstn);
    fetch_weight_nnz_fifo.enq(fetch_weight_nnz_enq);
    fetch_weight_nnz_fifo.deq(fetch_weight_nnz_deq);

    process_weight_nnz_fifo.clk(clk);
    process_weight_nnz_fifo.rst(rstn);
    process_weight_nnz_fifo.enq(process_weight_nnz_enq);
    process_weight_nnz_fifo.deq(process_weight_nnz_deq);

    fetch_weight_input_indices_fifo.clk(clk);
    fetch_weight_input_indices_fifo.rst(rstn);
    fetch_weight_input_indices_fifo.enq(fetch_weight_input_indices_enq);
    fetch_weight_input_indices_fifo.deq(fetch_weight_input_indices_deq);

    input_data_packed_fifo.clk(clk);
    input_data_packed_fifo.rst(rstn);
    input_data_packed_fifo.enq(input_data_packed_enq);
    input_data_packed_fifo.deq(input_data_packed_deq);

#if SUPPORT_MX
    read_weight_scale_nnz_fifo.clk(clk);
    read_weight_scale_nnz_fifo.rst(rstn);
    read_weight_scale_nnz_fifo.enq(read_weight_scale_nnz_enq);
    read_weight_scale_nnz_fifo.deq(read_weight_scale_nnz_deq);

    read_weight_scale_input_indices_fifo.clk(clk);
    read_weight_scale_input_indices_fifo.rst(rstn);
    read_weight_scale_input_indices_fifo.enq(
        read_weight_scale_input_indices_enq);
    read_weight_scale_input_indices_fifo.deq(
        read_weight_scale_input_indices_deq);
#endif

    SC_THREAD(fetch_input_indptr);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(process_input_indptr);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(fetch_input_indices);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(process_input_indices);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(fetch_input_data);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(process_input_data);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(fetch_weight);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(process_weight);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
#if SUPPORT_MX
    SC_THREAD(fetch_weight_scales);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(write_weight_scales);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(read_weight_scales);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
#endif
    SC_THREAD(run_accumulation);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(send_outputs);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(send_params);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void fetch_input_indptr() {
    fetch_input_indptr_param.ResetRead();
    input_indptr_req.Reset();
    input_indptr_resp.Reset();
    input_indptr_pack.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = fetch_input_indptr_param.Pop();

      // we only care about the K and X loops
      loop_t loop_counters[2][2];
      loop_t loop_bounds[2][2];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 2; j++) {
          loop_bounds[i][j] = params.loops[i][j] - 1;
        }
      }

      loop_t K1 = params.loops[0][params.weight_loop_idx[0]];
      loop_t K0 = params.loops[1][params.weight_loop_idx[1]];

      loop_t X0 = params.loops[1][params.x_loop_idx[1]];
      loop_t X1 = params.loops[0][params.x_loop_idx[0]];

      // varable to keep track of the range of x covered by the current indptr
      // fetch
      loop_t current_max_x = 0;
      loop_t current_min_x = 0;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
          for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
            for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
              loop_t x1 = loop_counters[0][params.x_loop_idx[0]];
              loop_t x0 = loop_counters[1][params.x_loop_idx[1]];
              loop_t x = x1 * X0 + x0;
              if (x > current_max_x || x < current_min_x || x == 0) {
                send_input_request<Meta, NUM_META>(params.spmm_indptr_offset, x,
                                                   input_indptr_req);
                current_min_x = x;
                current_max_x = x + NUM_META - 2;
                ac_int<Meta::width * NUM_META, false> bits = 0;
                process_matrix_input<Meta, NUM_META, port_width,
                                     Meta::width * NUM_META>(input_indptr_resp,
                                                             bits);
                Pack1D<Meta, NUM_META> indptrs;
#pragma hls_unroll yes
                for (int i = 0; i < NUM_META; i++) {
                  indptrs[i].set_bits(
                      bits.template slc<Meta::width>(i * Meta::width));
                }
                input_indptr_pack.Push(indptrs);
              }
              if (loop_counters[1][1] == loop_bounds[1][1]) break;
            }
            if (loop_counters[1][0] == loop_bounds[1][0]) break;
          }
          if (loop_counters[0][1] == loop_bounds[0][1]) break;
        }
        if (loop_counters[0][0] == loop_bounds[0][0]) break;
      }
    }
  }

  void process_input_indptr() {
    process_input_indptr_param.ResetRead();
    input_indptr_pack.ResetRead();
    input_indptr_enq.ResetWrite();
    fetch_input_indices_start_enq.ResetWrite();
    fetch_input_indices_end_enq.ResetWrite();
    process_input_indices_start_enq.ResetWrite();
    process_input_indices_end_enq.ResetWrite();
    fetch_input_data_start_enq.ResetWrite();
    fetch_input_data_end_enq.ResetWrite();
    process_input_data_start_enq.ResetWrite();
    process_input_data_end_enq.ResetWrite();
    fetch_weight_nnz_enq.ResetWrite();
    process_weight_nnz_enq.ResetWrite();
    read_weight_scale_nnz_enq.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = process_input_indptr_param.Pop();

      // we only care about the K and X loops
      loop_t loop_counters[2][2];
      loop_t loop_bounds[2][2];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 2; j++) {
          loop_bounds[i][j] = params.loops[i][j] - 1;
        }
      }

      loop_t K1 = params.loops[0][params.weight_loop_idx[0]];
      loop_t K0 = params.loops[1][params.weight_loop_idx[1]];

      loop_t X0 = params.loops[1][params.x_loop_idx[1]];
      loop_t X1 = params.loops[0][params.x_loop_idx[0]];

      // varable to keep track of the range of x covered by the current indptr
      // fetch
      loop_t current_max_x = 0;
      loop_t current_min_x = 0;

      Pack1D<Meta, NUM_META> indptrs;

      Meta indptr_tile_offset;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
          for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
            for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
              loop_t x1 = loop_counters[0][params.x_loop_idx[0]];
              loop_t x0 = loop_counters[1][params.x_loop_idx[1]];
              loop_t x = x1 * X0 + x0;
              if (x > current_max_x || x < current_min_x || x == 0) {
                indptrs = input_indptr_pack.Pop();
                current_min_x = x;
                current_max_x = x + NUM_META - 2;
                if (x == 0) {
                  indptr_tile_offset = indptrs[0];
                }
              }
              Meta start_index =
                  indptrs[x - current_min_x] - indptr_tile_offset;
              Meta end_index =
                  indptrs[x - current_min_x + 1] - indptr_tile_offset;
              Meta nnz = end_index - start_index;
              input_indptr_enq.Push(nnz);
              fetch_input_indices_start_enq.Push(start_index);
              fetch_input_indices_end_enq.Push(end_index);
              process_input_indices_start_enq.Push(start_index);
              process_input_indices_end_enq.Push(end_index);
              fetch_weight_nnz_enq.Push(nnz);
              process_weight_nnz_enq.Push(nnz);
              read_weight_scale_nnz_enq.Push(nnz);
              fetch_input_data_start_enq.Push(start_index);
              fetch_input_data_end_enq.Push(end_index);
              process_input_data_start_enq.Push(start_index);
              process_input_data_end_enq.Push(end_index);
              if (loop_counters[1][1] == loop_bounds[1][1]) break;
            }
            if (loop_counters[1][0] == loop_bounds[1][0]) break;
          }
          if (loop_counters[0][1] == loop_bounds[0][1]) break;
        }
        if (loop_counters[0][0] == loop_bounds[0][0]) break;
      }
    }
  }

  void fetch_input_indices() {
    fetch_input_indices_param.ResetRead();
    input_indices_req.Reset();
    fetch_input_indices_start_deq.ResetRead();
    fetch_input_indices_end_deq.ResetRead();

    wait();

    while (true) {
      const MatrixParams params = fetch_input_indices_param.Pop();

      // we only care about the K and X loops
      loop_t loop_counters[2][2];
      loop_t loop_bounds[2][2];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 2; j++) {
          loop_bounds[i][j] = params.loops[i][j] - 1;
        }
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
          for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
            for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
              Meta start = fetch_input_indices_start_deq.Pop();
              Meta end = fetch_input_indices_end_deq.Pop();

              Meta nnz = end - start;
              // the row is entirely empty
              if (nnz.int_val != 0) {
                loop_t num_packs_bound =
                    (nnz.int_val + NUM_META - 1) / NUM_META - 1;
                for (loop_t i = 0;; i++) {
                  ac_int<32, false> address = start.int_val + i * NUM_META;
                  send_input_request<Meta, NUM_META>(
                      params.spmm_indices_offset, address, input_indices_req);
                  if (i == num_packs_bound) break;
                }
              }
              if (loop_counters[1][1] == loop_bounds[1][1]) break;
            }
            if (loop_counters[1][0] == loop_bounds[1][0]) break;
          }
          if (loop_counters[0][1] == loop_bounds[0][1]) break;
        }
        if (loop_counters[0][0] == loop_bounds[0][0]) break;
      }
    }
  }

  void process_input_indices() {
    input_indices_resp.Reset();
    fetch_weight_input_indices_enq.ResetWrite();
    read_weight_scale_input_indices_enq.ResetWrite();
    process_input_indices_start_deq.ResetRead();
    process_input_indices_end_deq.ResetRead();

    wait();

    while (true) {
      Meta start = process_input_indices_start_deq.Pop();
      Meta end = process_input_indices_end_deq.Pop();

      Meta length = end - start;
      // the row is entirely empty
      if (length.int_val == 0) {
        continue;
      }

      loop_t num_packs = (length.int_val + NUM_META - 1) / NUM_META;
      loop_t num_packs_bound = num_packs - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t i = 0;; i++) {
        ac_int<Meta::width * NUM_META, false> bits = 0;
        process_matrix_input<Meta, NUM_META, port_width,
                             Meta::width * NUM_META>(input_indices_resp, bits);

        Pack1D<Meta, NUM_META> indices;
#pragma hls_unroll yes
        for (int j = 0; j < NUM_META; j++) {
          indices[j].set_bits(bits.template slc<Meta::width>(j * Meta::width));
        }
        fetch_weight_input_indices_enq.Push(indices);
        read_weight_scale_input_indices_enq.Push(indices);
        if (i == num_packs_bound) break;
      }
    }
  }

  void fetch_input_data() {
    fetch_input_data_param.ResetRead();
    input_data_req.Reset();
    fetch_input_data_start_deq.ResetRead();
    fetch_input_data_end_deq.ResetRead();

    wait();

    while (true) {
      const MatrixParams params = fetch_input_data_param.Pop();

      // we only care about the K and X loops
      loop_t loop_counters[2][2];
      loop_t loop_bounds[2][2];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 2; j++) {
          loop_bounds[i][j] = params.loops[i][j] - 1;
        }
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
          for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
            for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
              Meta start = fetch_input_data_start_deq.Pop();
              Meta end = fetch_input_data_end_deq.Pop();

              Meta nnz = end - start;

              // do nothing if the row is entirely empty
              if (nnz.int_val != 0) {
                loop_t num_packs_bound =
                    (nnz.int_val + NUM_DATA - 1) / NUM_DATA - 1;

                for (loop_t i = 0;; i++) {
                  ac_int<32, false> address = start.int_val + i * NUM_DATA;
                  send_input_request<Input, NUM_DATA>(params.spmm_data_offset,
                                                      address, input_data_req);
                  if (i == num_packs_bound) break;
                }
              }
              if (loop_counters[1][1] == loop_bounds[1][1]) break;
            }
            if (loop_counters[1][0] == loop_bounds[1][0]) break;
          }
          if (loop_counters[0][1] == loop_bounds[0][1]) break;
        }
        if (loop_counters[0][0] == loop_bounds[0][0]) break;
      }
    }
  }

  void process_input_data() {
    input_data_resp.Reset();
    input_data_packed_enq.ResetWrite();
    process_input_data_start_deq.ResetRead();
    process_input_data_end_deq.ResetRead();

    wait();

    while (true) {
      Meta start = process_input_data_start_deq.Pop();
      Meta end = process_input_data_end_deq.Pop();

      Meta length = end - start;
      // the row is entirely empty
      if (length.int_val == 0) {
        continue;
      }

      loop_t num_packs = (length.int_val + NUM_DATA - 1) / NUM_DATA;
      loop_t num_packs_bound = num_packs - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_t i = 0;; i++) {
        ac_int<Input::width * NUM_DATA, false> bits = 0;
        process_matrix_input<Input, NUM_DATA, port_width,
                             Input::width * NUM_DATA>(input_data_resp, bits);
        Pack1D<Input, NUM_DATA> data;
#pragma hls_unroll yes
        for (int j = 0; j < NUM_DATA; j++) {
          data[j].set_bits(bits.template slc<Input::width>(j * Input::width));
        }
        input_data_packed_enq.Push(data);
        if (i == num_packs_bound) break;
      }
    }
  }

  void fetch_weight() {
    fetch_weight_param.ResetRead();
    fetch_weight_input_indices_deq.ResetRead();
    weight_req.Reset();
    fetch_weight_nnz_deq.ResetRead();
#if SUPPORT_MX
    weight_scale_req.Reset();
#endif

    wait();

    while (true) {
      const MatrixParams params = fetch_weight_param.Pop();

      // we only care about the K and X loops
      loop_t loop_counters[2][2];
      loop_t loop_bounds[2][2];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 2; j++) {
          loop_bounds[i][j] = params.loops[i][j] - 1;
        }
      }

      loop_t K1 = params.loops[0][params.weight_loop_idx[0]];
      loop_t K0 = params.loops[1][params.weight_loop_idx[1]];
      loop_t K = K1 * K0 * width;

      loop_t X0 = params.loops[1][params.x_loop_idx[1]];
      loop_t X1 = params.loops[0][params.x_loop_idx[0]];

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
          for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
            for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
              loop_t k1 = loop_counters[0][params.weight_loop_idx[0]];
              loop_t k0 = loop_counters[1][params.weight_loop_idx[1]];
              loop_t k = k1 * K0 * width + k0 * width;

              Meta nnz = fetch_weight_nnz_deq.Pop();
              loop_t num_packs_bound =
                  (nnz.int_val + NUM_META - 1) / NUM_META - 1;

              // skip if no weights to fetch for this row
              if (nnz.int_val != 0) {
                for (loop_t x = 0;; x++) {
                  // index here is row index (y dimension)
                  Pack1D<Meta, NUM_META> indices =
                      fetch_weight_input_indices_deq.Pop();
                  for (int j = 0; j < NUM_META; j++) {
                    ac_int<32, false> address = indices[j].int_val * K + k;
                    (fetch_matrix_input<WeightTypes, width, WeightTypes...>(
                         params.weight_dtype, params.weight_offset, address,
                         weight_req),
                     ...);
                    if (x * NUM_META + j == nnz.int_val - 1) break;
                  }
                  if (x == num_packs_bound) break;
                }
              }
              if (loop_counters[1][1] == loop_bounds[1][1]) break;
            }
            if (loop_counters[1][0] == loop_bounds[1][0]) break;
          }
          if (loop_counters[0][1] == loop_bounds[0][1]) break;
        }
        if (loop_counters[0][0] == loop_bounds[0][0]) break;
      }
    }
  }

  void process_weight() {
    process_weight_param.ResetRead();
    process_weight_nnz_deq.ResetRead();
    weight_resp.Reset();
    weight_data.ResetWrite();

    wait();

    while (true) {
      MatrixParams params = process_weight_param.Pop();

      // we only care about the K and X loops
      loop_t loop_counters[2][2];
      loop_t loop_bounds[2][2];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 2; j++) {
          loop_bounds[i][j] = params.loops[i][j] - 1;
        }
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
          for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
            for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
              Meta nnz = process_weight_nnz_deq.Pop();
              loop_t nnz_bound = nnz.int_val - 1;

              // skip if no weights to process for this row
              if (nnz.int_val != 0) {
                constexpr int buffer_width = Weight::width * width;
                for (loop_t i = 0;; i++) {
                  ac_int<buffer_width, false> bits = 0;
                  bool success =
                      (process_matrix_input<WeightTypes, width, port_width,
                                            buffer_width, WeightTypes...>(
                           params.weight_dtype, weight_resp, bits) ||
                       ...);

                  Pack1D<Weight, width> weights;
#pragma hls_unroll yes
                  for (int j = 0; j < width; j++) {
                    auto data =
                        bits.template slc<Weight::width>(j * Weight::width);
                    weights[j] = decode_input<Weight, NUM_CODEBOOK_ENTRIES,
                                              WeightTypes...>(
                        data, params.weight_dtype, params.use_weight_codebook,
                        params.weight_code);
                  }
                  weight_data.Push(weights);
                  if (i == nnz_bound) break;
                }
              }
              if (loop_counters[1][1] == loop_bounds[1][1]) break;
            }
            if (loop_counters[1][0] == loop_bounds[1][0]) break;
          }
          if (loop_counters[0][1] == loop_bounds[0][1]) break;
        }
        if (loop_counters[0][0] == loop_bounds[0][0]) break;
      }
    }
  }

#if SUPPORT_MX
  void fetch_weight_scales() {
    fetch_weight_scale_param.ResetRead();
    weight_scale_req.Reset();

    wait();

    while (true) {
      MatrixParams params = fetch_weight_scale_param.Pop();

      // we only care about the K and X loops
      loop_t loop_counters[2][2];
      loop_t loop_bounds[2][2];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 2; j++) {
          loop_bounds[i][j] = params.loops[i][j] - 1;
        }
      }

      loop_t K1 = params.loops[0][params.weight_loop_idx[0]];
      loop_t K0 = params.loops[1][params.weight_loop_idx[1]];
      loop_t K = K1 * K0;
      loop_t k1_offset = K0 * width;

      loop_t X0 = params.loops[1][params.x_loop_idx[1]];
      loop_t X1 = params.loops[0][params.x_loop_idx[0]];

      // we are fetching scales for all C in the weight matrix in each fill
      loop_t c_offset = K * width;
      loop_t C =
          params.weight_addr_loops[0][params.weight_addr_reduction_loop_idx[0]];
      // replacing inner x loop with C loop for fetching weight scales
      loop_bounds[1][params.x_loop_idx[1]] = C - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
          for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
            for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
              loop_t c = loop_counters[1][params.x_loop_idx[1]];
              loop_t k1 = loop_counters[0][params.weight_loop_idx[0]];
              loop_t k0 = loop_counters[1][params.weight_loop_idx[1]];
              ac_int<32, false> address =
                  c * c_offset + k1 * k1_offset + k0 * width;
              send_input_request<Scale, width>(params.weight_scale_offset,
                                               address, weight_scale_req);
              if (loop_counters[1][1] == loop_bounds[1][1]) break;
            }
            if (loop_counters[1][0] == loop_bounds[1][0]) break;
          }
          if (loop_counters[0][1] == loop_bounds[0][1]) break;
        }
        if (loop_counters[0][0] == loop_bounds[0][0]) break;
      }
    }
  }

  void write_weight_scales() {
    write_weight_scale_param.ResetRead();
    weight_scale_resp.Reset();
    weight_scale_write_req[0].ResetWrite();
    weight_scale_write_req[1].ResetWrite();

    bool bank_sel = 0;

    wait();

    while (true) {
      MatrixParams params = write_weight_scale_param.Pop();

      // we only care about the K and X loops
      loop_t loop_counters[2][2];
      loop_t loop_bounds[2][2];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 2; j++) {
          loop_bounds[i][j] = params.loops[i][j] - 1;
        }
      }

      loop_t K1 = params.loops[0][params.weight_loop_idx[0]];
      loop_t K0 = params.loops[1][params.weight_loop_idx[1]];

      loop_t X0 = params.loops[1][params.x_loop_idx[1]];
      loop_t X1 = params.loops[0][params.x_loop_idx[0]];

      // we are fetching scales for all C in the weight matrix in each fill
      loop_t C =
          params.weight_addr_loops[0][params.weight_addr_reduction_loop_idx[0]];
      // replacing inner x loop with C loop for fetching weight scales
      loop_bounds[1][params.x_loop_idx[1]] = C - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
          for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
            for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
              loop_t c = loop_counters[1][params.x_loop_idx[1]];
              loop_t k0 = loop_counters[1][params.weight_loop_idx[1]];
              ac_int<32, false> address = c * K0 + k0;
              ac_int<Scale::width * width, false> data;
              process_matrix_input<Scale, width, port_width,
                                   Scale::width * width>(weight_scale_resp,
                                                         data);

              BufferWriteRequest<SCALE_PORT_TYPE> req;
              req.address = address;
              req.data = data;
              req.last = k0 == loop_bounds[1][params.weight_loop_idx[1]] &&
                         c == loop_bounds[1][params.x_loop_idx[1]];
              weight_scale_write_req[bank_sel].Push(req);
              if (loop_counters[1][1] == loop_bounds[1][1]) break;
            }
            if (loop_counters[1][0] == loop_bounds[1][0]) break;
          }
          bank_sel = !bank_sel;
          if (loop_counters[0][1] == loop_bounds[0][1]) break;
        }
        if (loop_counters[0][0] == loop_bounds[0][0]) break;
      }
    }
  }

  void read_weight_scales() {
    read_weight_scale_param.ResetRead();
    read_weight_scale_nnz_deq.ResetRead();
    read_weight_scale_input_indices_deq.ResetRead();
    weight_scale_read_req[0].ResetWrite();
    weight_scale_read_req[1].ResetWrite();

    bool bank_sel = 0;

    wait();

    while (true) {
      const MatrixParams params = read_weight_scale_param.Pop();

      // we only care about the K and X loops
      loop_t loop_counters[2][2];
      loop_t loop_bounds[2][2];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 2; j++) {
          loop_bounds[i][j] = params.loops[i][j] - 1;
        }
      }

      loop_t k0_bound = loop_bounds[1][params.weight_loop_idx[1]];
      loop_t x0_bound = loop_bounds[1][params.x_loop_idx[1]];

      loop_t K1 = params.loops[0][params.weight_loop_idx[0]];
      loop_t K0 = params.loops[1][params.weight_loop_idx[1]];
      loop_t K = K1 * K0 * width;

      loop_t X0 = params.loops[1][params.x_loop_idx[1]];
      loop_t X1 = params.loops[0][params.x_loop_idx[0]];

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
          for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
            for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
              loop_t k1 = loop_counters[0][params.weight_loop_idx[0]];
              loop_t k0 = loop_counters[1][params.weight_loop_idx[1]];
              loop_t x0 = loop_counters[1][params.x_loop_idx[1]];
              loop_t k = k1 * K0 * width + k0 * width;
              Meta nnz = read_weight_scale_nnz_deq.Pop();
              loop_t num_packs_bound =
                  (nnz.int_val + NUM_META - 1) / NUM_META - 1;

              // skip if no weights to fetch for this row
              if (nnz.int_val != 0) {
                for (loop_t x = 0;; x++) {
                  // index here is row index (y dimension)
                  Pack1D<Meta, NUM_META> indices =
                      read_weight_scale_input_indices_deq.Pop();
                  for (int j = 0; j < NUM_META; j++) {
                    ac_int<32, false> address =
                        indices[j].int_val / bs * K0 + k0;
                    BufferReadRequest req;
                    req.address = address;
                    req.last = (x * NUM_META + j == nnz.int_val - 1) &&
                               (k0 == k0_bound) && (x0 == x0_bound);
                    weight_scale_read_req[bank_sel].Push(req);
                    if (x * NUM_META + j == nnz.int_val - 1) break;
                  }
                  if (x == num_packs_bound) break;
                }
              } else {
                // send a dummy request so if the last nnz is zero we can still
                // switch banks properly
                BufferReadRequest req;
                req.address = 0XFFFF;
                req.last = (k0 == k0_bound) && (x0 == x0_bound);
                weight_scale_read_req[bank_sel].Push(req);
              }
              if (loop_counters[1][1] == loop_bounds[1][1]) break;
            }
            if (loop_counters[1][0] == loop_bounds[1][0]) break;
          }
          bank_sel = !bank_sel;
          if (loop_counters[0][1] == loop_bounds[0][1]) break;
        }
        if (loop_counters[0][0] == loop_bounds[0][0]) break;
      }
    }
  }
#endif

  void run_accumulation() {
    run_accumulation_param.ResetRead();
    input_indptr_deq.ResetRead();
    input_data_packed_deq.ResetRead();
    weight_data.ResetRead();
#if SUPPORT_MX
    weight_scale_data.ResetRead();
#endif
    accumulation_out.ResetWrite();

    wait();

    while (true) {
      MatrixParams params = run_accumulation_param.Pop();

      // we only care about the K and X loops
      loop_t loop_counters[2][2];
      loop_t loop_bounds[2][2];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 2; j++) {
          loop_bounds[i][j] = params.loops[i][j] - 1;
        }
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
          for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
            for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
              loop_t nnz_bound = input_indptr_deq.Pop().int_val;
              Pack1D<Output, width> acc_old[FEEDBACK_DEPTH];

#pragma hls_unroll yes
              for (int i = 0; i < FEEDBACK_DEPTH; i++) {
                acc_old[i] = Pack1D<Output, width>::zero();
              }
              loop_t input_data_idx = 0;
              Pack1D<Input, NUM_DATA> input_data_pack;
              // even if the current row is empty, we need to pop from the
              // weight scale so that the dummy request that is used to trigger
              // the bank switch is consumed and discarded
              for (loop_t nnz = 0;; nnz++) {
                Pack1D<Scale, width> weight_scale = Pack1D<Scale, width>::one();
                SCALE_PORT_TYPE scale_data;
#if SUPPORT_MX
                // nnz != nnz_bound to avoid reading an additional scale at the
                // end of the iteration nnz == 0 to enforce reading at least
                // once even for empty rows
                if (params.is_mx_op && (nnz != nnz_bound) || (nnz == 0)) {
                  scale_data = weight_scale_data.Pop();
#pragma hls_unroll yes
                  for (int i = 0; i < width; i++) {
                    weight_scale[i].set_bits(
                        scale_data.template slc<Scale::width>(i *
                                                              Scale::width));
                  }
                }
#endif
                if (nnz == nnz_bound) break;
                if (input_data_idx == 0) {
                  input_data_pack = input_data_packed_deq.Pop();
                }
                Input input = input_data_pack[input_data_idx];
                input_data_idx = (input_data_idx + 1) % NUM_DATA;
                Pack1D<Weight, width> weights = weight_data.Pop();
                Pack1D<Output, width> psums;
#pragma hls_unroll yes
                for (int i = 0; i < width; i++) {
                  psums[i] =
                      multiply_accumulate(input, weights[i], weight_scale[i],
                                          acc_old[FEEDBACK_LAST][i]);
                }

#pragma hls_unroll yes
                for (int k = FEEDBACK_LAST; k > 0; k--) {
                  acc_old[k] = acc_old[k - 1];
                }
                acc_old[0] = psums;
              }

              Pack1D<Output, width> outputs;
#pragma hls_unroll yes
              for (int i = 0; i < width; i++) {
                Pack1D<Output, FEEDBACK_DEPTH> col;
#pragma hls_unroll yes
                for (int j = 0; j < FEEDBACK_DEPTH; j++) {
                  col[j] = acc_old[j][i];
                }
                outputs[i] = add_tree(col);
              }
              accumulation_out.Push(outputs);
              if (loop_counters[1][1] == loop_bounds[1][1]) break;
            }
            if (loop_counters[1][0] == loop_bounds[1][0]) break;
          }
          if (loop_counters[0][1] == loop_bounds[0][1]) break;
        }
        if (loop_counters[0][0] == loop_bounds[0][0]) break;
      }
    }
  }

  void send_outputs() {
    send_output_params.ResetRead();
    accumulation_out.ResetRead();
    spmm_unit_output.Reset();
    done.Reset();

    wait();

    while (true) {
      const MatrixParams params = send_output_params.Pop();

      // we only care about the K and X loops
      loop_t loop_counters[2][2];
      loop_t loop_bounds[2][2];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 2; j++) {
          loop_bounds[i][j] = params.loops[i][j] - 1;
        }
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
          for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
            for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
              auto acc = accumulation_out.Pop();
              for (int i = 0; i < width / vu_width; i++) {
                Pack1D<Output, vu_width> outputs;
#pragma hls_unroll yes
                for (int j = 0; j < vu_width; j++) {
                  outputs[j] = acc[i * vu_width + j];
                }
                spmm_unit_output.Push(outputs);
              }
              if (loop_counters[1][1] == loop_bounds[1][1]) break;
            }
            if (loop_counters[1][0] == loop_bounds[1][0]) break;
          }
          if (loop_counters[0][1] == loop_bounds[0][1]) break;
        }
        if (loop_counters[0][0] == loop_bounds[0][0]) break;
      }
      done.SyncPush();
    }
  }

  void send_params() {
    params_in.ResetRead();
    fetch_input_indptr_param.ResetWrite();
    process_input_indptr_param.ResetWrite();
    fetch_input_indices_param.ResetWrite();
    fetch_input_data_param.ResetWrite();
    fetch_weight_param.ResetWrite();
    process_weight_param.ResetWrite();
    fetch_weight_scale_param.ResetWrite();
    write_weight_scale_param.ResetWrite();
    read_weight_scale_param.ResetWrite();
    send_output_params.ResetWrite();
    run_accumulation_param.ResetWrite();
    start.Reset();

    wait();

    while (true) {
      const MatrixParams params = params_in.Pop();

      start.SyncPush();

      fetch_input_indptr_param.Push(params);
      process_input_indptr_param.Push(params);
      fetch_input_indices_param.Push(params);
      fetch_input_data_param.Push(params);
      fetch_weight_param.Push(params);
#if SUPPORT_MX
      if (params.is_mx_op) {
        fetch_weight_scale_param.Push(params);
        write_weight_scale_param.Push(params);
        read_weight_scale_param.Push(params);
      }
#endif
      process_weight_param.Push(params);
      run_accumulation_param.Push(params);
      send_output_params.Push(params);
    }
  }
};
