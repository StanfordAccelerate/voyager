#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "ArchitectureParams.h"
#include "DoubleBuffer.h"
#include "DualPortBuffer.h"
#include "InputController.h"
#include "InputScaleController.h"
#include "OutputController.h"
#include "ParamsDeserializer.h"
#include "WeightScaleController.h"
#include "mc_scverify.h"

#if MATRIX_BACKEND == MATRIX_BACKEND_CIM
#include "cim/CIMProcessor.h"
#include "cim/CIMWeightController.h"
#else
#include "MatrixProcessor.h"
#include "WeightController.h"
#endif

SC_MODULE(MatrixUnit) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

#if SUPPORT_MX
  static constexpr int PARAMS_MODULE_COUNT = 6;
  static constexpr int SCALE_PORT_WIDTH = SCALE_DATATYPE::width * OC_DIMENSION;
  typedef ac_int<SCALE_PORT_WIDTH, false> SCALE_PORT_TYPE;
#else
  static constexpr int PARAMS_MODULE_COUNT = 4;
#endif

#if DOUBLE_BUFFERED_ACCUM_BUFFER
  static constexpr int ACCUM_BUFFER_BANKS = 2;
#else
  static constexpr int ACCUM_BUFFER_BANKS = 1;
#endif

#if MATRIX_BACKEND == MATRIX_BACKEND_CIM
  using ActiveMatrixProcessor = CIMProcessor<
      InputTypeList, WeightTypeList, SA_INPUT_TYPE, SA_WEIGHT_TYPE,
      ACCUM_DATATYPE, ACCUM_BUFFER_DATATYPE, SCALE_DATATYPE, IC_DIMENSION,
      OC_DIMENSION, ACCUM_BUFFER_SIZE, CIM_MACRO_INPUT_LANES,
      CIM_MACRO_OUTPUT_LANES, CIM_WEIGHT_SETS, CIM_BASE_A_WIDTH,
      CIM_BASE_B_WIDTH, CIM_BASE_C_WIDTH, CIM_MACRO_WRITE_INPUT_LANES,
      CIM_MAC_LATENCY, CIM_MODE, CIM_SIGNED, CIM_TILE_INPUT_AXIS_ELEMENTS,
      CIM_TILE_OUTPUT_AXIS_ELEMENTS, CIM_INPUT_AXIS_TILES,
      CIM_OUTPUT_AXIS_TILES, CIM_A_PORT_TILES, CIM_B_PORT_TILES,
      CIM_C_PORT_TILES, CIM_C_BEAT_LAYOUT, CIM_ARRAY_RESULT_SLOTS,
      CIM_LOCAL_ACCUM_CONTEXTS>;
  using ActiveWeightController = CIMWeightController<
      WeightTypeList, ACCUM_BUFFER_DATATYPE, IC_DIMENSION, OC_DIMENSION,
      OC_PORT_WIDTH, ActiveMatrixProcessor::WEIGHT_ROW_WIDTH,
      ActiveMatrixProcessor::WEIGHT_WRITE_WIDTH, CIM_WEIGHT_SETS>;
#else
  using ActiveMatrixProcessor =
      MatrixProcessor<InputTypeList, WeightTypeList, SA_INPUT_TYPE,
                      SA_WEIGHT_TYPE, ACCUM_DATATYPE, ACCUM_BUFFER_DATATYPE,
                      SCALE_DATATYPE, IC_DIMENSION, OC_DIMENSION,
                      ACCUM_BUFFER_SIZE>;
  using ActiveWeightController =
      WeightController<WeightTypeList, ACCUM_BUFFER_DATATYPE, IC_DIMENSION,
                       OC_DIMENSION, OC_PORT_WIDTH, WEIGHT_BUFFER_WIDTH>;
#endif

  MatrixParamsDeserializer<0, PARAMS_MODULE_COUNT> CCS_INIT_S1(
      params_deserializer);
  Connections::In<ac_int<64, false>> CCS_INIT_S1(serial_params_in);
  Connections::Combinational<MatrixParams> matrix_params[PARAMS_MODULE_COUNT];

  InputController<InputTypeList, IC_DIMENSION, IC_PORT_WIDTH,
                  INPUT_BUFFER_WIDTH>
      CCS_INIT_S1(input_controller);

  DoubleBuffer<INPUT_BUFFER_SIZE, INPUT_BUFFER_WIDTH> CCS_INIT_S1(input_buffer);
  Connections::Out<MemoryRequest> CCS_INIT_S1(input_req);
  Connections::In<ac_int<IC_PORT_WIDTH, false>> CCS_INIT_S1(input_resp);
  Connections::Combinational<
      BufferWriteRequest<ac_int<INPUT_BUFFER_WIDTH, false>>>
      input_buffer_write_req[2];
  Connections::Combinational<BufferReadRequest> input_buffer_read_req[2];
  Connections::Combinational<ac_int<INPUT_BUFFER_WIDTH, false>> CCS_INIT_S1(
      window_buffer_in);
  Connections::Combinational<ac_int<INPUT_BUFFER_WIDTH, false>> CCS_INIT_S1(
      window_buffer_out);

#if SUPPORT_MX
  InputScaleController<SCALE_DATATYPE, IC_DIMENSION> CCS_INIT_S1(
      input_scale_controller);
  DoubleBuffer<INPUT_BUFFER_SIZE, SCALE_DATATYPE::width> CCS_INIT_S1(
      input_scale_buffer);
  Connections::Out<MemoryRequest> CCS_INIT_S1(input_scale_req);
  Connections::In<ac_int<SCALE_DATATYPE::width, false>> CCS_INIT_S1(
      input_scale_resp);
  Connections::Combinational<
      BufferWriteRequest<ac_int<SCALE_DATATYPE::width, false>>>
      input_scale_write_req[2];
  Connections::Combinational<BufferReadRequest> input_scale_read_req[2];
  Connections::Combinational<ac_int<SCALE_DATATYPE::width, false>> CCS_INIT_S1(
      input_scale_read_resp);
#endif

  ActiveWeightController CCS_INIT_S1(weight_controller);

#if MATRIX_BACKEND == MATRIX_BACKEND_CIM
  Connections::Combinational<
      ac_int<ActiveMatrixProcessor::WEIGHT_WRITE_WIDTH, false>>
      CCS_INIT_S1(weight_channel);
  Connections::Combinational<cim::WeightDescriptor> CCS_INIT_S1(
      weight_descriptor_channel);
#else
  DoubleBuffer<WEIGHT_BUFFER_SIZE, WEIGHT_BUFFER_WIDTH> CCS_INIT_S1(
      weight_buffer);
  Connections::Combinational<
      BufferWriteRequest<ac_int<WEIGHT_BUFFER_WIDTH, false>>>
      weight_buffer_write_req[2];
  Connections::Combinational<BufferReadRequest> weight_buffer_read_req[2];
  Connections::Combinational<ac_int<WEIGHT_BUFFER_WIDTH, false>> CCS_INIT_S1(
      weight_buffer_read_resp);
#endif
  Connections::Out<MemoryRequest> CCS_INIT_S1(weight_req);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(weight_resp);
  Connections::Out<MemoryRequest> CCS_INIT_S1(bias_req);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(bias_resp);

#if SUPPORT_MX
  WeightScaleController<SCALE_DATATYPE, IC_DIMENSION, OC_DIMENSION,
                        OC_PORT_WIDTH>
      CCS_INIT_S1(weight_scale_controller);
  DoubleBuffer<WEIGHT_BUFFER_SIZE / IC_DIMENSION,
               SCALE_DATATYPE::width * OC_DIMENSION>
      CCS_INIT_S1(weight_scale_buffer);
  Connections::Out<MemoryRequest> CCS_INIT_S1(weight_scale_req);
  Connections::In<ac_int<OC_PORT_WIDTH, false>> CCS_INIT_S1(weight_scale_resp);
  Connections::Combinational<BufferWriteRequest<SCALE_PORT_TYPE>>
      weight_scale_write_req[2];
  Connections::Combinational<BufferReadRequest> weight_scale_read_req[2];
  Connections::Combinational<ac_int<SCALE_PORT_WIDTH, false>> CCS_INIT_S1(
      weight_scale_read_resp);
#endif

  ActiveMatrixProcessor CCS_INIT_S1(matrix_processor);
  Connections::Combinational<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>>
      CCS_INIT_S1(bias_data);

  DualPortBuffer<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>, ACCUM_BUFFER_SIZE>
      CCS_INIT_S1(accumulation_buffer);
  Connections::Combinational<ac_int<16, false>>
      accumulation_buffer_mu_read_address[ACCUM_BUFFER_BANKS];
  Connections::Combinational<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>>
      accumulation_buffer_mu_read_data[ACCUM_BUFFER_BANKS];
  Connections::Combinational<
      BufferWriteRequest<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>>>
      accumulation_buffer_mu_write_request[ACCUM_BUFFER_BANKS];

#if DOUBLE_BUFFERED_ACCUM_BUFFER
  Connections::SyncChannel accumulation_buffer_mu_done[ACCUM_BUFFER_BANKS];

  Connections::Combinational<ac_int<16, false>>
      accumulation_buffer_vu_read_address[2];
  Connections::Combinational<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>>
      accumulation_buffer_vu_read_data[2];
  // Write request from Vector Unit, unused for now
  Connections::Combinational<
      BufferWriteRequest<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>>>
      accumulation_buffer_vu_write_request[2];
  Connections::SyncChannel accumulation_buffer_vu_done[2];
#endif

  MatrixUnitOutputController<ACCUM_BUFFER_DATATYPE, OC_DIMENSION, OC_PORT_WIDTH,
                             MU_OUTPUT_TYPES>
      CCS_INIT_S1(output_controller);

  Connections::Combinational<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>>
      CCS_INIT_S1(matrix_processor_output);

  Connections::Out<Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>> output_channel;
  Connections::Out<ac_int<OC_PORT_WIDTH, false>> output_data;
  Connections::Out<ac_int<ADDRESS_WIDTH, false>> output_addr;

  Connections::SyncOut CCS_INIT_S1(start);
  Connections::SyncOut CCS_INIT_S1(done);
#if ENABLE_PERF_COUNTERS
  sc_in<MatrixPerformance::CounterIndex> CCS_INIT_S1(perf_counter_select);
  sc_out<MatrixPerformance::Counter> CCS_INIT_S1(perf_counter_value);
  sc_signal<MatrixPerformance::Counter>
      processor_perf_counters[MatrixPerformance::PROCESSOR_COUNTER_COUNT];
  sc_signal<MatrixPerformance::Counter> input_perf_reads[2],
      input_perf_writes[2];
#if MATRIX_BACKEND == MATRIX_BACKEND_SYSTOLIC
  sc_signal<MatrixPerformance::Counter> weight_perf_reads[2],
      weight_perf_writes[2];
#endif
#if SUPPORT_MX
  // Scale SRAMs are separate from the reported matrix data-buffer traffic.
  sc_signal<MatrixPerformance::Counter> input_scale_perf_reads[2],
      input_scale_perf_writes[2];
  sc_signal<MatrixPerformance::Counter> weight_scale_perf_reads[2],
      weight_scale_perf_writes[2];
#endif
  sc_signal<MatrixPerformance::Counter> accum_perf_reads[ACCUM_BUFFER_BANKS],
      accum_perf_writes[ACCUM_BUFFER_BANKS];
  sc_signal<MatrixPerformance::Counter>
      perf_snapshot[MatrixPerformance::PERFORMANCE_COUNTER_COUNT];
  sc_signal<MatrixPerformance::SnapshotSequence> perf_snapshot_sequence;
#endif

  SC_CTOR(MatrixUnit) {
    params_deserializer.clk(clk);
    params_deserializer.rstn(rstn);
    params_deserializer.serial_params_in(serial_params_in);
    for (int i = 0; i < PARAMS_MODULE_COUNT; i++) {
      params_deserializer.params_out[i](matrix_params[i]);
    }

    input_controller.clk(clk);
    input_controller.rstn(rstn);
    input_controller.input_req(input_req);
    input_controller.input_resp(input_resp);
    input_controller.params_in(matrix_params[0]);
    input_controller.window_buffer_in(window_buffer_in);
    input_controller.window_buffer_out(window_buffer_out);

    input_buffer.clk(clk);
    input_buffer.rstn(rstn);
    for (int i = 0; i < 2; i++) {
      input_controller.write_request[i](input_buffer_write_req[i]);
      input_controller.read_request[i](input_buffer_read_req[i]);

      input_buffer.write_request[i](input_buffer_write_req[i]);
      input_buffer.read_request[i](input_buffer_read_req[i]);
#if ENABLE_PERF_COUNTERS
      input_buffer.perf_reads[i](input_perf_reads[i]);
      input_buffer.perf_writes[i](input_perf_writes[i]);
#endif
    }
    input_buffer.output(window_buffer_in);

#if SUPPORT_MX
    input_scale_controller.clk(clk);
    input_scale_controller.rstn(rstn);
    input_scale_controller.scale_req(input_scale_req);
    input_scale_controller.scale_resp(input_scale_resp);
    input_scale_controller.params_in(matrix_params[4]);

    input_scale_buffer.clk(clk);
    input_scale_buffer.rstn(rstn);
    for (int i = 0; i < 2; i++) {
      input_scale_controller.write_request[i](input_scale_write_req[i]);
      input_scale_controller.read_request[i](input_scale_read_req[i]);

      input_scale_buffer.write_request[i](input_scale_write_req[i]);
      input_scale_buffer.read_request[i](input_scale_read_req[i]);
#if ENABLE_PERF_COUNTERS
      input_scale_buffer.perf_reads[i](input_scale_perf_reads[i]);
      input_scale_buffer.perf_writes[i](input_scale_perf_writes[i]);
#endif
    }
    input_scale_buffer.output(input_scale_read_resp);
#endif

    weight_controller.clk(clk);
    weight_controller.rstn(rstn);
    weight_controller.weight_req(weight_req);
    weight_controller.weight_resp(weight_resp);
    weight_controller.params_in(matrix_params[1]);
    weight_controller.bias_req(bias_req);
    weight_controller.bias_resp(bias_resp);
    weight_controller.bias_data(bias_data);

#if MATRIX_BACKEND == MATRIX_BACKEND_CIM
    weight_controller.weight_channel(weight_channel);
    weight_controller.weight_descriptor_channel(weight_descriptor_channel);
#else
    weight_buffer.clk(clk);
    weight_buffer.rstn(rstn);
    for (int i = 0; i < 2; i++) {
      weight_controller.write_request[i](weight_buffer_write_req[i]);
      weight_controller.read_request[i](weight_buffer_read_req[i]);

      weight_buffer.write_request[i](weight_buffer_write_req[i]);
      weight_buffer.read_request[i](weight_buffer_read_req[i]);
#if ENABLE_PERF_COUNTERS
      weight_buffer.perf_reads[i](weight_perf_reads[i]);
      weight_buffer.perf_writes[i](weight_perf_writes[i]);
#endif
    }
    weight_buffer.output(weight_buffer_read_resp);
#endif

#if SUPPORT_MX
    weight_scale_controller.clk(clk);
    weight_scale_controller.rstn(rstn);
    weight_scale_controller.weight_scale_req(weight_scale_req);
    weight_scale_controller.weight_scale_resp(weight_scale_resp);
    weight_scale_controller.params_in(matrix_params[5]);

    weight_scale_buffer.clk(clk);
    weight_scale_buffer.rstn(rstn);
    for (int i = 0; i < 2; i++) {
      weight_scale_controller.write_request[i](weight_scale_write_req[i]);
      weight_scale_controller.read_request[i](weight_scale_read_req[i]);

      weight_scale_buffer.write_request[i](weight_scale_write_req[i]);
      weight_scale_buffer.read_request[i](weight_scale_read_req[i]);
#if ENABLE_PERF_COUNTERS
      weight_scale_buffer.perf_reads[i](weight_scale_perf_reads[i]);
      weight_scale_buffer.perf_writes[i](weight_scale_perf_writes[i]);
#endif
    }
    weight_scale_buffer.output(weight_scale_read_resp);
#endif

    matrix_processor.clk(clk);
    matrix_processor.rstn(rstn);
    matrix_processor.input_channel(window_buffer_out);
#if MATRIX_BACKEND == MATRIX_BACKEND_CIM
    matrix_processor.weight_channel(weight_channel);
    matrix_processor.weight_descriptor_channel(weight_descriptor_channel);
#else
    matrix_processor.weight_channel(weight_buffer_read_resp);
#endif
    matrix_processor.bias_channel(bias_data);
    matrix_processor.params_in(matrix_params[2]);
    matrix_processor.start(start);
#if ENABLE_PERF_COUNTERS
    for (int i = 0; i < MatrixPerformance::PROCESSOR_COUNTER_COUNT; i++)
      matrix_processor.perf_counters[i](processor_perf_counters[i]);
#endif

    for (int i = 0; i < ACCUM_BUFFER_BANKS; i++) {
      matrix_processor.accumulation_buffer_read_address[i](
          accumulation_buffer_mu_read_address[i]);
      matrix_processor.accumulation_buffer_read_data[i](
          accumulation_buffer_mu_read_data[i]);
      matrix_processor.accumulation_buffer_write_request[i](
          accumulation_buffer_mu_write_request[i]);
#if DOUBLE_BUFFERED_ACCUM_BUFFER
      matrix_processor.accumulation_buffer_done[i](
          accumulation_buffer_mu_done[i]);
#endif
    }

    matrix_processor.output_channel(matrix_processor_output);

    accumulation_buffer.clk(clk);
    accumulation_buffer.rstn(rstn);

    for (int i = 0; i < ACCUM_BUFFER_BANKS; i++) {
#if ENABLE_PERF_COUNTERS
      accumulation_buffer.perf_reads[i](accum_perf_reads[i]);
      accumulation_buffer.perf_writes[i](accum_perf_writes[i]);
#endif
      accumulation_buffer.read_address[i * 2](
          accumulation_buffer_mu_read_address[i]);
      accumulation_buffer.read_data[i * 2](accumulation_buffer_mu_read_data[i]);
      accumulation_buffer.write_request[i * 2](
          accumulation_buffer_mu_write_request[i]);
#if DOUBLE_BUFFERED_ACCUM_BUFFER
      accumulation_buffer.done[i * 2](accumulation_buffer_mu_done[i]);
#endif
    }

#if DOUBLE_BUFFERED_ACCUM_BUFFER
    for (int i = 0; i < ACCUM_BUFFER_BANKS; i++) {
      accumulation_buffer.read_address[i * 2 + 1](
          accumulation_buffer_vu_read_address[i]);
      accumulation_buffer.read_data[i * 2 + 1](
          accumulation_buffer_vu_read_data[i]);
      accumulation_buffer.write_request[i * 2 + 1](
          accumulation_buffer_vu_write_request[i]);
      accumulation_buffer.done[i * 2 + 1](accumulation_buffer_vu_done[i]);
    }
#endif

#if SUPPORT_MX
    matrix_processor.input_scale_channel(input_scale_read_resp);
    matrix_processor.weight_scale_channel(weight_scale_read_resp);
#endif

    output_controller.clk(clk);
    output_controller.rstn(rstn);
    output_controller.params_in(matrix_params[3]);
    output_controller.matrix_processor_output(matrix_processor_output);
#if DOUBLE_BUFFERED_ACCUM_BUFFER
    for (int i = 0; i < ACCUM_BUFFER_BANKS; i++) {
      output_controller.accumulation_buffer_read_address[i](
          accumulation_buffer_vu_read_address[i]);
      output_controller.accumulation_buffer_read_data[i](
          accumulation_buffer_vu_read_data[i]);
      output_controller.accumulation_buffer_done[i](
          accumulation_buffer_vu_done[i]);
    }
#endif
    output_controller.vector_unit_input_data(output_channel);
    output_controller.matrix_unit_output_data(output_data);
    output_controller.matrix_unit_output_addr(output_addr);
    output_controller.done(done);

#if ENABLE_PERF_COUNTERS
    SC_THREAD(snapshot_performance);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
    SC_METHOD(read_performance_counter);
    sensitive << perf_counter_select << perf_snapshot_sequence;
    for (int i = 0; i < MatrixPerformance::PERFORMANCE_COUNTER_COUNT; i++)
      sensitive << perf_snapshot[i];
#endif
  }

#if ENABLE_PERF_COUNTERS
  // Matrix-unit retirement includes output drain and every accumulation-bank
  // reader. Sampling never changes the completion handshake.
  void snapshot_performance() {
    using namespace MatrixPerformance;
    SnapshotSequence sequence = 0;
    perf_snapshot_sequence.write(0);
#pragma hls_unroll yes
    for (int i = 0; i < PERFORMANCE_COUNTER_COUNT; i++)
      perf_snapshot[i].write(0);
    wait();
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      if (done.vld.read() && done.rdy.read()) {
#pragma hls_unroll yes
        for (int i = 0; i < PROCESSOR_COUNTER_COUNT; i++)
          perf_snapshot[i].write(processor_perf_counters[i].read());
        Counter input_reads = 0, input_writes = 0;
        Counter accum_reads = 0, accum_writes = 0;
#pragma hls_unroll yes
        for (int bank = 0; bank < 2; bank++) {
          input_reads += input_perf_reads[bank].read();
          input_writes += input_perf_writes[bank].read();
        }
#pragma hls_unroll yes
        for (int bank = 0; bank < ACCUM_BUFFER_BANKS; bank++) {
          accum_reads += accum_perf_reads[bank].read();
          accum_writes += accum_perf_writes[bank].read();
        }
        perf_snapshot[storage_index(INPUT_BUFFER_READS)].write(input_reads);
        perf_snapshot[storage_index(INPUT_BUFFER_WRITES)].write(input_writes);
        perf_snapshot[storage_index(ACCUM_BUFFER_READS)].write(accum_reads);
        perf_snapshot[storage_index(ACCUM_BUFFER_WRITES)].write(accum_writes);
#if MATRIX_BACKEND == MATRIX_BACKEND_SYSTOLIC
        Counter weight_reads = 0, weight_writes = 0;
#pragma hls_unroll yes
        for (int bank = 0; bank < 2; ++bank) {
          weight_reads += weight_perf_reads[bank].read();
          weight_writes += weight_perf_writes[bank].read();
        }
        perf_snapshot[storage_index(WEIGHT_BUFFER_READS)].write(weight_reads);
        perf_snapshot[storage_index(WEIGHT_BUFFER_WRITES)].write(weight_writes);
#endif
        perf_snapshot_sequence.write(++sequence);
      }
      wait();
    }
  }

  void read_performance_counter() {
    using namespace MatrixPerformance;
    Counter value = 0;
    if (perf_counter_select.read() == SNAPSHOT_SEQUENCE)
      value = perf_snapshot_sequence.read();
#pragma hls_unroll yes
    for (int i = 0; i < PERFORMANCE_COUNTER_COUNT; i++)
      if (perf_counter_select.read() == PROCESSOR_ACTIVE_CYCLES + i)
        value = perf_snapshot[i].read();
    perf_counter_value.write(value);
  }
#endif
};
