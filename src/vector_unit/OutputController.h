#pragma once

#include <systemc.h>

#include "../AccelTypes.h"
#include "../ArchitectureParams.h"
#include "VectorOps.h"

template <typename VectorType, typename ScaleType, int width,
          typename... OutputTypes>
SC_MODULE(OutputController) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<VectorParams> CCS_INIT_S1(params_in);
  Connections::Combinational<VectorParams> CCS_INIT_S1(write_address_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(
      write_scale_address_params);
  Connections::Combinational<ac_int<32, false>> CCS_INIT_S1(write_scale_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(signal_done_params);

  Connections::In<Pack1D<VectorType, width>> CCS_INIT_S1(vector_in);
  Connections::In<ScaleType> CCS_INIT_S1(scale_in);

#if SUPPORT_DWC
  Connections::In<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(dwc_address_in);
  Connections::Combinational<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      dwc_scale_address);
#endif

#if SUPPORT_SPMM
  using Meta = SPMM_META_DATATYPE;

  Connections::In<CsrDataAndIndices<VectorType, Meta, width>> CCS_INIT_S1(
      csr_data_and_indices);
  Connections::In<Pack1D<Meta, width>> CCS_INIT_S1(csr_indptr);

  Connections::Combinational<VectorParams> CCS_INIT_S1(process_csr_data_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(
      process_csr_indices_params);
  Connections::Combinational<VectorParams> CCS_INIT_S1(
      process_csr_indptr_params);

  using write_req_t = CsrWriteRequest<ac_int<OC_PORT_WIDTH, false>>;
  Connections::Combinational<write_req_t> CCS_INIT_S1(csr_data_write_req);
  Connections::Combinational<write_req_t> CCS_INIT_S1(csr_indptr_write_req);
#endif

  Connections::Out<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      vector_output_data);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      vector_output_addr);
  Connections::Out<ac_int<ScaleType::width, false>> CCS_INIT_S1(
      mx_scale_output_data);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      mx_scale_output_addr);
  Connections::Out<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(
      sparse_tensor_output_data);
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> CCS_INIT_S1(
      sparse_tensor_output_addr);

  Connections::SyncChannel CCS_INIT_S1(write_vector_output_done);
#if SUPPORT_MX
  Connections::SyncChannel CCS_INIT_S1(write_mx_scale_done);
#endif
#if SUPPORT_SPMM
  Connections::SyncChannel CCS_INIT_S1(process_csr_data_and_indices_done);
  Connections::SyncChannel CCS_INIT_S1(process_csr_indptr_done);
#endif
  Connections::SyncOut CCS_INIT_S1(done);

  static constexpr int LOOP_WIDTH = 16;

  SC_CTOR(OutputController) {
    SC_THREAD(write_vector_output);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(write_vector_address);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
#if SUPPORT_MX
    SC_THREAD(write_mx_scale);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(write_scale_address);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
#endif
#if SUPPORT_SPMM
    SC_THREAD(process_csr_data_and_indices);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(process_csr_indptr);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(write_sparse_tensor);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
#endif
    SC_THREAD(signal_done);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void write_vector_output() {
    params_in.Reset();
    write_address_params.ResetWrite();
    write_scale_params.ResetWrite();
    write_scale_address_params.ResetWrite();
#if SUPPORT_SPMM
    process_csr_data_params.ResetWrite();
    process_csr_indptr_params.ResetWrite();
#endif
    vector_in.Reset();
    vector_output_data.Reset();
    write_vector_output_done.ResetWrite();
    signal_done_params.ResetWrite();

    wait();

    while (true) {
      VectorParams params = params_in.Pop();
      write_address_params.Push(params);
#if SUPPORT_SPMM
      if (params.has_sparse_output) {
        process_csr_data_params.Push(params);
        process_csr_indptr_params.Push(params);
      }
#endif
      ac_int<32, false> loop_bound =
          params.output_loops[0][0] * params.output_loops[0][1] *
              params.output_loops[0][2] * params.output_loops[1][0] *
              params.output_loops[1][1] * params.output_loops[1][2] -
          1;

      if (params.quantize_output_mx) {
        write_scale_params.Push(loop_bound);
        write_scale_address_params.Push(params);
      }

      signal_done_params.Push(params);

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (ac_int<32, false> i = 0;; i++) {
        Pack1D<VectorType, width> outputs = vector_in.Pop();

        bool found = (send_output_data<OutputTypes, width, VectorType,
                                       OC_PORT_WIDTH, OutputTypes...>(
                          params.output_dtype, params.use_output_codebook,
                          outputs, vector_output_data) ||
                      ...);
#ifndef __SYNTHESIS__
        if (!found) {
          std::cerr << "Error: output type '" << params.output_dtype
                    << "' is not valid.\n";
        }
#endif
        if (i == loop_bound) break;
      }

      write_vector_output_done.SyncPush();
    }
  }

  void write_vector_address() {
    write_address_params.ResetRead();
    vector_output_addr.Reset();
#if SUPPORT_DWC
    dwc_address_in.Reset();
    dwc_scale_address.ResetWrite();
#endif

    wait();

    while (true) {
      VectorParams params = write_address_params.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][3];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_bounds[i][j] = params.output_loops[i][j] - 1;
        }
      }

      ac_int<LOOP_WIDTH, false> Y1 =
          params.output_loops[0][params.output_y_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> X1 =
          params.output_loops[0][params.output_x_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> K2 =
          params.output_loops[0][params.output_k_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> Y0 =
          params.output_loops[1][params.output_y_loop_idx[1]];
      ac_int<LOOP_WIDTH, false> X0 =
          params.output_loops[1][params.output_x_loop_idx[1]];
      ac_int<16, false> K1 =
          params.output_loops[1][params.output_k_loop_idx[1]] * width;

      ac_int<16, false> X = X0 * X1;
      ac_int<16, false> K = K2 * K1;

      ac_int<16, false> loop_bound_4 = params.output_loops[1][2];
      ac_int<16, false> loop_bound_3 = params.output_loops[1][1] * loop_bound_4;
      ac_int<16, false> loop_bound_2 = params.output_loops[1][0] * loop_bound_3;
      ac_int<16, false> loop_bound_1 = params.output_loops[0][2] * loop_bound_2;
      ac_int<16, false> loop_bound_0 = params.output_loops[0][1] * loop_bound_1;

#if SUPPORT_DWC
      // TODO: remove this and handle properly using vector params
      if (params.is_dwc) {
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        while (true) {
          ac_int<ADDRESS_WIDTH, false> address = dwc_address_in.Pop();
          bool found = (send_output_address<OutputTypes, width, OC_PORT_WIDTH,
                                            OutputTypes...>(
                            params.output_dtype, params.vector_output_offset,
                            address, vector_output_addr) ||
                        ...);

#ifndef __SYNTHESIS__
          if (!found) {
            std::cerr << "Error: output type '" << params.output_dtype
                      << "' is not valid.\n";
          }
#endif
#if SUPPORT_MX
          if (params.quantize_output_mx) {
            dwc_scale_address.Push(params.mx_scale_offset +
                                   address / width * ScaleType::width / 8);
          }
#endif
        }
      } else
#endif
      {
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
          for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
            for (loop_counters[0][2] = 0;; loop_counters[0][2]++) {
              for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
                for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
                  for (loop_counters[1][2] = 0;; loop_counters[1][2]++) {
                    ac_int<32, false> address;
                    if (params.output_mode == 1) {
                      ac_int<LOOP_WIDTH, false> y1 =
                          loop_counters[0][params.output_y_loop_idx[0]];
                      ac_int<LOOP_WIDTH, false> x1 =
                          loop_counters[0][params.output_x_loop_idx[0]];
                      ac_int<LOOP_WIDTH, false> k2 =
                          loop_counters[0][params.output_k_loop_idx[0]];
                      ac_int<LOOP_WIDTH, false> y0 =
                          loop_counters[1][params.output_y_loop_idx[1]];
                      ac_int<LOOP_WIDTH, false> x0 =
                          loop_counters[1][params.output_x_loop_idx[1]];
                      ac_int<LOOP_WIDTH, false> k1 =
                          loop_counters[1][params.output_k_loop_idx[1]];

                      ac_int<32, false> y = y1 * Y0 + y0;
                      ac_int<32, false> x = x1 * X0 + x0;
                      ac_int<32, false> k = k2 * K1 + k1 * width;

                      if (params.transpose_for_scores) {
                        // k / head_size * (X * head_size) + x * head_size
                        // + k % head_size
                        ac_int<16, false> mask =
                            (1 << params.head_size_lg2) - 1;
                        address = (((k >> params.head_size_lg2) * X)
                                   << params.head_size_lg2) +
                                  (x << params.head_size_lg2) + (k & mask);
                      } else {
                        address = y * X * K + x * K + k;
                      }
                    } else if (params.output_mode == 2) {
                      address = (loop_counters[0][0] * loop_bound_0 +
                                 loop_counters[0][1] * loop_bound_1 +
                                 loop_counters[0][2] * loop_bound_2 +
                                 loop_counters[1][0] * loop_bound_3 +
                                 loop_counters[1][1] * loop_bound_4 +
                                 loop_counters[1][2]) *
                                width;
                    }

                    bool found =
                        (send_output_address<OutputTypes, width, OC_PORT_WIDTH,
                                             OutputTypes...>(
                             params.output_dtype, params.vector_output_offset,
                             address, vector_output_addr) ||
                         ...);

#ifndef __SYNTHESIS__
                    if (!found) {
                      std::cerr << "Error: output type '" << params.output_dtype
                                << "' is not valid.\n";
                    }
#endif
                    if (loop_counters[1][2] == loop_bounds[1][2]) break;
                  }
                  if (loop_counters[1][1] == loop_bounds[1][1]) break;
                }
                if (loop_counters[1][0] == loop_bounds[1][0]) break;
              }
              if (loop_counters[0][2] == loop_bounds[0][2]) break;
            }
            if (loop_counters[0][1] == loop_bounds[0][1]) break;
          }
          if (loop_counters[0][0] == loop_bounds[0][0]) break;
        }
      }
    }
  }
#if SUPPORT_MX
  void write_mx_scale() {
    write_scale_params.ResetRead();
    scale_in.Reset();
    mx_scale_output_data.Reset();
    write_mx_scale_done.ResetWrite();

    wait();

    while (true) {
      ac_int<32, false> loop_bound = write_scale_params.Pop();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (ac_int<32, false> i = 0;; i++) {
        ScaleType scale = scale_in.Pop();
        mx_scale_output_data.Push(scale.bits_rep());
        if (i == loop_bound) break;
      }

      write_mx_scale_done.SyncPush();
    }
  }

  void write_scale_address() {
    write_scale_address_params.ResetRead();
    mx_scale_output_addr.Reset();
#if SUPPORT_DWC
    dwc_scale_address.ResetRead();
#endif

    wait();

    while (true) {
      VectorParams params = write_scale_address_params.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][3];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][3];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 3; j++) {
          loop_bounds[i][j] = params.output_loops[i][j] - 1;
        }
      }

      ac_int<LOOP_WIDTH, false> Y1 =
          params.output_loops[0][params.output_y_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> X1 =
          params.output_loops[0][params.output_x_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> K2 =
          params.output_loops[0][params.output_k_loop_idx[0]];
      ac_int<LOOP_WIDTH, false> Y0 =
          params.output_loops[1][params.output_y_loop_idx[1]];
      ac_int<LOOP_WIDTH, false> X0 =
          params.output_loops[1][params.output_x_loop_idx[1]];
      ac_int<16, false> K1 =
          params.output_loops[1][params.output_k_loop_idx[1]];

      ac_int<16, false> X = X0 * X1;
      ac_int<16, false> K = K2 * K1;

      ac_int<16, false> loop_bound_4 = params.output_loops[1][2];
      ac_int<16, false> loop_bound_3 = params.output_loops[1][1] * loop_bound_4;
      ac_int<16, false> loop_bound_2 = params.output_loops[1][0] * loop_bound_3;
      ac_int<16, false> loop_bound_1 = params.output_loops[0][2] * loop_bound_2;
      ac_int<16, false> loop_bound_0 = params.output_loops[0][1] * loop_bound_1;

#if SUPPORT_DWC
      // TODO: remove this and handle properly using vector params
      if (params.is_dwc) {
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        while (true) {
          ac_int<ADDRESS_WIDTH, false> address = dwc_scale_address.Pop();
          mx_scale_output_addr.Push(address);
        }
      } else
#endif
      {
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
          for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
            for (loop_counters[0][2] = 0;; loop_counters[0][2]++) {
              for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
                for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
                  for (loop_counters[1][2] = 0;; loop_counters[1][2]++) {
                    ac_int<32, false> address;
                    if (params.output_mode == 1) {
                      ac_int<LOOP_WIDTH, false> y1 =
                          loop_counters[0][params.output_y_loop_idx[0]];
                      ac_int<LOOP_WIDTH, false> x1 =
                          loop_counters[0][params.output_x_loop_idx[0]];
                      ac_int<LOOP_WIDTH, false> k2 =
                          loop_counters[0][params.output_k_loop_idx[0]];
                      ac_int<LOOP_WIDTH, false> y0 =
                          loop_counters[1][params.output_y_loop_idx[1]];
                      ac_int<LOOP_WIDTH, false> x0 =
                          loop_counters[1][params.output_x_loop_idx[1]];
                      ac_int<LOOP_WIDTH, false> k1 =
                          loop_counters[1][params.output_k_loop_idx[1]];

                      ac_int<32, false> y = y1 * Y0 + y0;
                      ac_int<32, false> x = x1 * X0 + x0;
                      ac_int<32, false> k = k2 * K1 + k1;
                      address = y * X * K + x * K + k;
                    } else if (params.output_mode == 2) {
                      address = loop_counters[0][0] * loop_bound_0 +
                                loop_counters[0][1] * loop_bound_1 +
                                loop_counters[0][2] * loop_bound_2 +
                                loop_counters[1][0] * loop_bound_3 +
                                loop_counters[1][1] * loop_bound_4 +
                                loop_counters[1][2];
                    }

                    mx_scale_output_addr.Push(params.mx_scale_offset +
                                              address * ScaleType::width / 8);

                    if (loop_counters[1][2] == loop_bounds[1][2]) break;
                  }
                  if (loop_counters[1][1] == loop_bounds[1][1]) break;
                }
                if (loop_counters[1][0] == loop_bounds[1][0]) break;
              }
              if (loop_counters[0][2] == loop_bounds[0][2]) break;
            }
            if (loop_counters[0][1] == loop_bounds[0][1]) break;
          }
          if (loop_counters[0][0] == loop_bounds[0][0]) break;
        }
      }
    }
  }
#endif
#if SUPPORT_SPMM
  template <typename T, int pack_width>
  void write_sparse_data(const Pack1D<T, pack_width>& pack,
                         ac_int<32, false> offset,
                         Connections::Combinational<write_req_t>& channel) {
    constexpr int total_bits = T::width * pack_width;
    constexpr int num_words = (total_bits + OC_PORT_WIDTH - 1) / OC_PORT_WIDTH;

    ac_int<total_bits, false> bits;
    bits = BitsToType<decltype(bits)>(TypeToBits(pack));

    for (int i = 0; i < num_words; i++) {
      ac_int<32, false> address = offset + i * (OC_PORT_WIDTH / 8);
      write_req_t write_req = {
          .address = address,
          .data = bits.template slc<OC_PORT_WIDTH>(i * OC_PORT_WIDTH),
      };
      channel.Push(write_req);
    }
  }

  void process_csr_data_and_indices() {
    process_csr_data_params.ResetRead();
    csr_data_and_indices.Reset();
    csr_data_write_req.ResetWrite();
    process_csr_data_and_indices_done.ResetWrite();

    wait();

    while (true) {
      VectorParams params = process_csr_data_params.Pop();

      ac_int<32, false> csr_data_address = params.csr_data_offset;
      ac_int<32, false> csr_indices_address = params.csr_indices_offset;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (true) {
        auto data_and_indices = csr_data_and_indices.Pop();

        write_sparse_data<VectorType, width>(
            data_and_indices.data, csr_data_address, csr_data_write_req);

        write_sparse_data<Meta, width>(data_and_indices.indices,
                                       csr_indices_address, csr_data_write_req);

        csr_data_address += width * VectorType::width / 8;
        csr_indices_address += width * Meta::width / 8;

        if (data_and_indices.is_last) {
          break;
        }
      }

      process_csr_data_and_indices_done.SyncPush();
    }
  }

  void process_csr_indptr() {
    process_csr_indptr_params.ResetRead();
    csr_indptr.Reset();
    csr_indptr_write_req.ResetWrite();
    process_csr_indptr_done.ResetWrite();

    wait();

    while (true) {
      VectorParams params = process_csr_indptr_params.Pop();

      auto Y1 = params.output_loops[0][params.output_y_loop_idx[0]];
      auto X1 = params.output_loops[0][params.output_x_loop_idx[0]];
      auto Y0 = params.output_loops[1][params.output_y_loop_idx[1]];
      auto X0 = params.output_loops[1][params.output_x_loop_idx[1]];

      ac_int<16, false> X = Y1 * Y0 * X1 * X0;
      ac_int<16, false> loop_bound = (X + 1 + width - 1) / width - 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (ac_int<16, false> x = 0;; x++) {
        auto indptr = csr_indptr.Pop();
        ac_int<32, false> address =
            params.csr_indptr_offset + x * (width * Meta::width / 8);
        write_sparse_data<Meta, width>(indptr, address, csr_indptr_write_req);

        if (x == loop_bound) break;
      }

      process_csr_indptr_done.SyncPush();
    }
  }

  void write_sparse_tensor() {
    csr_data_write_req.ResetRead();
    csr_indptr_write_req.ResetRead();
    sparse_tensor_output_data.Reset();
    sparse_tensor_output_addr.Reset();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      write_req_t write_req;
      if (csr_data_write_req.PopNB(write_req)) {
        sparse_tensor_output_data.Push(write_req.data);
        sparse_tensor_output_addr.Push(write_req.address);
      } else if (csr_indptr_write_req.PopNB(write_req)) {
        sparse_tensor_output_data.Push(write_req.data);
        sparse_tensor_output_addr.Push(write_req.address);
      } else {
        wait();
      }
    }
  }
#endif

  void signal_done() {
    done.Reset();
    signal_done_params.ResetRead();
    write_vector_output_done.ResetRead();
#if SUPPORT_MX
    write_mx_scale_done.ResetRead();
#endif
#if SUPPORT_SPMM
    process_csr_data_and_indices_done.ResetRead();
    process_csr_indptr_done.ResetRead();
#endif

    wait();

    while (true) {
      VectorParams params = signal_done_params.Pop();

      write_vector_output_done.SyncPop();

#if SUPPORT_MX
      if (params.quantize_output_mx) {
        write_mx_scale_done.SyncPop();
      }
#endif

#if SUPPORT_SPMM
      if (params.has_sparse_output) {
        process_csr_data_and_indices_done.SyncPop();
        process_csr_indptr_done.SyncPop();
      }
#endif

      done.SyncPush();
    }
  }
};
