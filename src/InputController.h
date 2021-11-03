#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"

template <typename DTYPE, int NROWS>
SC_MODULE(InputController) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Params> paramsIn;

  Connections::Out<MemoryRequest> CCS_INIT_S1(addressRequest);
  Connections::In<Pack1D<DTYPE, NROWS> > CCS_INIT_S1(dataResponse);

  Connections::Out<int> writeAddress[2];
  Connections::Out<Pack1D<DTYPE, NROWS> > writeData[2];
  Connections::Out<int> writeControl[2];
  Connections::Out<int> readAddress[2];
  Connections::Out<int> readControl[2];

  Connections::Combinational<Params> CCS_INIT_S1(fetcherParams);
  Connections::Combinational<Params> CCS_INIT_S1(writerParams);
  Connections::Combinational<Params> CCS_INIT_S1(readerParams);

  SC_CTOR(InputController) {
    SC_THREAD(read_params);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(fetcher);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(reader);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(writer);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void fetcher() {
    addressRequest.Reset();
    fetcherParams.ResetRead();

    wait();

    while (true) {
      Params params = fetcherParams.Pop();

      int loop_counters[2][3];

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0; loop_counters[0][0] < params.loops[0][0];
           loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0; loop_counters[0][1] < params.loops[0][1];
             loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0;
               loop_counters[0][2] < params.loops[0][2];
               loop_counters[0][2]++) {
            // inner memory
            for (loop_counters[1][params.reductionLoopIndex[1]] = 0;
                 loop_counters[1][params.reductionLoopIndex[1]] <
                 params.loops[1][params.reductionLoopIndex[1]];
                 loop_counters[1][params.reductionLoopIndex[1]]++) {
              for (loop_counters[1][params.inputLoopIndex[1]] = 0;
                   loop_counters[1][params.inputLoopIndex[1]] <
                   params.loops[1][params.inputLoopIndex[1]];
                   loop_counters[1][params.inputLoopIndex[1]]++) {
                int m0 = loop_counters[1][params.inputLoopIndex[1]];
                int m1 = loop_counters[0][params.inputLoopIndex[0]];
                int M0 = params.loops[1][params.inputLoopIndex[1]];
                int N1 = params.loops[1][params.reductionLoopIndex[1]];
                int n1 = loop_counters[1][params.reductionLoopIndex[1]];

                int baseAddress = (m0 + m1 * M0) * (N1 * NROWS) + n1 * NROWS;
                int burstSize = NROWS;
                MemoryRequest memRequest = {params.INPUT_OFFSET + baseAddress,
                                            burstSize};
                CCS_LOG(memRequest);
                addressRequest.Push(memRequest);
              }
            }
          }
        }
      }
    }
  }

  void writer() {
    writerParams.ResetRead();
    dataResponse.Reset();

    writeControl[0].Reset();
    writeControl[1].Reset();
    writeAddress[0].Reset();
    writeAddress[1].Reset();
    writeData[0].Reset();
    writeData[1].Reset();

    wait();

    while (true) {
      Params params = writerParams.Pop();

      bool bankSel = 0;

      int loop_counters[2][3];
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0; loop_counters[0][0] < params.loops[0][0];
           loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0; loop_counters[0][1] < params.loops[0][1];
             loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0;
               loop_counters[0][2] < params.loops[0][2];
               loop_counters[0][2]++) {
            // inner memory
            for (loop_counters[1][params.reductionLoopIndex[1]] = 0;
                 loop_counters[1][params.reductionLoopIndex[1]] <
                 params.loops[1][params.reductionLoopIndex[1]];
                 loop_counters[1][params.reductionLoopIndex[1]]++) {
              writeControl[bankSel].Push(
                  params.loops[1][params.inputLoopIndex[1]]);

              for (loop_counters[1][params.inputLoopIndex[1]] = 0;
                   loop_counters[1][params.inputLoopIndex[1]] <
                   params.loops[1][params.inputLoopIndex[1]];
                   loop_counters[1][params.inputLoopIndex[1]]++) {
                Pack1D<DTYPE, NROWS> data = dataResponse.Pop();

                int m0 = loop_counters[1][params.inputLoopIndex[1]];

                writeAddress[bankSel].Push(m0);
                writeData[bankSel].Push(data);
              }

              bankSel = !bankSel;
            }
          }
        }
      }
    }
  }

  void reader() {
    readerParams.ResetRead();

    readControl[0].Reset();
    readControl[1].Reset();
    readAddress[0].Reset();
    readAddress[1].Reset();

    wait();

    while (true) {
      Params params = readerParams.Pop();

      bool bankSel = 0;

      int loop_counters[2][3];
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0; loop_counters[0][0] < params.loops[0][0];
           loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0; loop_counters[0][1] < params.loops[0][1];
             loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0;
               loop_counters[0][2] < params.loops[0][2];
               loop_counters[0][2]++) {
            // inner memory
            for (loop_counters[1][params.reductionLoopIndex[1]] = 0;
                 loop_counters[1][params.reductionLoopIndex[1]] <
                 params.loops[1][params.reductionLoopIndex[1]];
                 loop_counters[1][params.reductionLoopIndex[1]]++) {
              readControl[bankSel].Push(
                  params.loops[1][params.weightLoopIndex[1]] *
                  params.loops[1][params.inputLoopIndex[1]]);

              for (loop_counters[1][params.weightLoopIndex[1]] = 0;
                   loop_counters[1][params.weightLoopIndex[1]] <
                   params.loops[1][params.weightLoopIndex[1]];
                   loop_counters[1][params.weightLoopIndex[1]]++) {
                for (loop_counters[1][params.inputLoopIndex[1]] = 0;
                     loop_counters[1][params.inputLoopIndex[1]] <
                     params.loops[1][params.inputLoopIndex[1]];
                     loop_counters[1][params.inputLoopIndex[1]]++) {
                  readAddress[bankSel].Push(
                      loop_counters[1][params.inputLoopIndex[1]]);
                }
              }
              bankSel = !bankSel;
            }
          }
        }
      }
    }
  }

  void read_params() {
    paramsIn.Reset();
    fetcherParams.ResetWrite();
    writerParams.ResetWrite();
    readerParams.ResetWrite();

    wait();

    while (true) {
      Params params = paramsIn.Pop();

      fetcherParams.Push(params);
      writerParams.Push(params);
      readerParams.Push(params);
    }
  }
};
