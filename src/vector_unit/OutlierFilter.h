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

      auto indptr = Pack1D<Meta, width>::zero();
      ac_int<32, false> nnz = 0;

      // Keep 2 * width to avoid overflow
      Pack1D<Vector, 2 * width> csr_data;
      Pack1D<Meta, 2 * width> csr_indices;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (ac_int<16, false> x = 0;; ++x) {
        for (ac_int<16, false> k = 0;; ++k) {
          auto data = data_in.Pop();

          auto outliers = Pack1D<Vector, width>::zero();
          auto filtered = Pack1D<Vector, width>::zero();

#pragma hls_unroll yes
          for (int i = 0; i < width; i++) {
            if (data[i] > threshold) {
              outliers[i] = data[i];
            } else {
              filtered[i] = data[i];
            }
          }

          data_out.Push(filtered);

#ifndef __SYNTHESIS__
          std::cerr << "OutlierFilter: x=" << x << " k=" << k << " nnz=" << nnz
                    << " threshold=" << threshold << "\n";
          std::cerr << "  data:     " << data << "\n";
          std::cerr << "  filtered: " << filtered << "\n";
          std::cerr << "  outliers: " << outliers << "\n";
#endif

#pragma hls_unroll yes
          for (int k0 = 0; k0 < width; k0++) {
            int index = nnz % (2 * width);
            if (outliers[k0] != Vector::zero()) {
              csr_data[index] = outliers[k0];
              csr_indices[index] = k * width + k0;
              nnz++;
            }
          }

#ifndef __SYNTHESIS__
          std::cerr << "  nnz after: " << nnz << "\n";
          std::cerr << "  csr_data: " << csr_data << "\n";
          std::cerr << "  csr_indices: " << csr_indices << "\n";
#endif

          int nnz_in_block = nnz % (2 * width);

          if (nnz_in_block > width) {
            Pack1D<Vector, width> csr_data_block;
            Pack1D<Meta, width> csr_indices_block;

#pragma hls_unroll yes
            for (int i = 0; i < width; i++) {
              csr_data_block[i] = csr_data[i];
              csr_indices_block[i] = csr_indices[i];
            }

            CsrDataAndIndices<Vector, Meta, width> output = {
                .data = csr_data_block,
                .indices = csr_indices_block,
                .is_last = false,
            };

            csr_data_and_indices_out.Push(output);

            // Shift remaining data to the front
#pragma hls_unroll yes
            for (int i = 0; i < width; i++) {
              csr_data[i] = csr_data[i + width];
              csr_indices[i] = csr_indices[i + width];
            }
          }

          if (k == config.dense_input_shape[1] - 1) break;
        }

        int indptr_idx = x % width;
        if (indptr_idx == width - 1) {
          csr_indptr_out.Push(indptr);
          indptr = Pack1D<Meta, width>::zero();
          indptr[0] = nnz;
        } else {
          indptr[indptr_idx + 1] = nnz;
        }

        if (x == config.dense_input_shape[0] - 1) break;
      }

      // Flush remaining data
      int remaining = nnz % width;

      Pack1D<Vector, width> csr_data_block;
      Pack1D<Meta, width> csr_indices_block;

#pragma hls_unroll yes
      for (int i = 0; i < width; i++) {
        csr_data_block[i] = i < remaining ? csr_data[i] : Vector::zero();
        csr_indices_block[i] = i < remaining ? csr_indices[i] : Meta::zero();
      }

      CsrDataAndIndices<Vector, Meta, width> output = {
          .data = csr_data_block,
          .indices = csr_indices_block,
          .is_last = true,
      };

      csr_data_and_indices_out.Push(output);

      // Flush final indptr
      csr_indptr_out.Push(indptr);
    }
  }
};
