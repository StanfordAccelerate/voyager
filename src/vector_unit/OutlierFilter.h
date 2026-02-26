#pragma once

#include <mc_connections.h>
#include <systemc.h>

template <typename Vector, typename Meta, int width>
SC_MODULE(OutlierFilter) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<OutlierFilterConfig> CCS_INIT_S1(config_in);
  Connections::In<Pack1D<Vector, width>> CCS_INIT_S1(data_in);

  Connections::Out<Pack1D<Vector, width>> CCS_INIT_S1(data_out);
  Connections::Out<CsrDataAndIndices<Vector, Meta, width>> CCS_INIT_S1(
      csr_data_and_indices_out);
  Connections::Out<Pack1D<Meta, width>> CCS_INIT_S1(csr_indptr_out);

  SC_CTOR(OutlierFilter) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    config_in.Reset();
    data_in.Reset();
    data_out.Reset();
    csr_data_and_indices_out.Reset();
    csr_indptr_out.Reset();

    wait();

    while (true) {
      auto config = config_in.Pop();

      Vector threshold;
      threshold.set_bits(config.outlier_threshold);

      CsrDataAndIndices<Vector, Meta, width> output;
      auto indptr = Pack1D<Meta, width>::zero();
      ac_int<32, false> nnz = 0;

      indptr[0] = config.indptr_offset;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (ac_int<16, false> x = 0;; ++x) {
        for (ac_int<16, false> k = 0;; ++k) {
          auto data = data_in.Pop();

          auto outliers = Pack1D<Vector, width>::zero();
          auto filtered = Pack1D<Vector, width>::zero();
          ac_int<width, false> flag = 0;

#pragma hls_unroll yes
          for (int i = 0; i < width; i++) {
            if (data[i].abs() > threshold) {
              outliers[i] = data[i];
              flag[i] = 1;
            } else {
              filtered[i] = data[i];
            }
          }

          data_out.Push(filtered);

          flag.reverse();
          bool all_sign;
          auto pos = flag.leading_sign(all_sign);

          while (!all_sign) {
            int index = nnz % width;
            output.data[index] = outliers[pos];
            output.indices[index] = k * width + pos;
            nnz++;
            flag[width - 1 - pos] = 0;

            if (nnz % width == 0 && nnz != 0) {
              csr_data_and_indices_out.Push(output);
              output = CsrDataAndIndices<Vector, Meta, width>();
            }

            pos = flag.leading_sign(all_sign);
          }

          if (k == config.dense_input_shape[1] - 1) break;
        }

        int indptr_idx = x % width;
        ac_int<32, false> current_index = config.indptr_offset + nnz;
        if (indptr_idx == width - 1) {
          csr_indptr_out.Push(indptr);
          indptr = Pack1D<Meta, width>::zero();
          indptr[0] = current_index;
        } else {
          indptr[indptr_idx + 1] = current_index;
        }

        if (x == config.dense_input_shape[0] - 1) break;
      }

      // Flush remaining data
      output.is_last = true;
      csr_data_and_indices_out.Push(output);
      csr_indptr_out.Push(indptr);
    }
  }
};
