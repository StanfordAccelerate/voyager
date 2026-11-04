#pragma once

#include <ac_int.h>
#include <mc_connections.h>
#include <systemc.h>

#include <type_traits>

#ifndef __SYNTHESIS__
#include <deque>
#include <sstream>
#include <stdexcept>
#include <vector>
#endif

#include "AccelTypes.h"
#include "ArchitectureParams.h"
#include "CIMArray.h"
#include "CIMConfig.h"
#include "CIMTypes.h"
#include "PackUtils.h"
#include "Params.h"
#include "PerfMonitor.h"

// Execute a lowered matrix schedule on a resident-weight CIM array
//
// WeightController owns weight order and lifetime: it streams set payloads plus
// descriptors that declare an ordered set sequence and its replay count. The
// loader assigns each sequence to the physical set ring, while the issue path
// only walks [replay][set], waits for first-use readiness, and releases a set
// after its final replay. The result path reduces IC/FX/FY contributions
// through MatrixUnit's accumulation buffer. The first two geometry arguments
// retain MatrixProcessor's template positions and specify the array input
// and output lane counts.
template <typename InputTypeTuple, typename WeightTypeTuple, typename Input,
          typename Weight, typename Psum, typename Buffer, typename Scale,
          int input_lanes, int output_lanes, int buffer_size,
          int MACRO_INPUT_LANES, int MACRO_OUTPUT_LANES, int WEIGHT_SETS,
          int BASE_A_WIDTH, int BASE_B_WIDTH, int BASE_C_WIDTH,
          int MACRO_WRITE_INPUT_LANES, int MAC_LATENCY, int MODE, bool SIGNED,
          int TILE_INPUT_AXIS_ELEMENTS, int TILE_OUTPUT_AXIS_ELEMENTS,
          int INPUT_AXIS_TILES, int OUTPUT_AXIS_TILES, int A_PORT_TILES,
          int B_PORT_TILES, int C_PORT_TILES, int C_BEAT_LAYOUT,
          int RESULT_SLOTS_PER_OUTPUT_TILE = INPUT_AXIS_TILES,
          int LOCAL_ACCUM_CONTEXTS = CIM_LOCAL_ACCUM_CONTEXTS>
SC_MODULE(CIMProcessor) {
 public:
  static constexpr int INPUT_LANES = input_lanes;
  static constexpr int OUTPUT_LANES = output_lanes;

  // Derive array input, weight, and accumulation widths from their datatypes.
  static constexpr int A_WIDTH = Input::width;
  static constexpr int B_WIDTH = Weight::width;
  static constexpr int C_WIDTH = Psum::width;

  using Array =
      CIMArray<MACRO_INPUT_LANES, MACRO_OUTPUT_LANES, WEIGHT_SETS, BASE_A_WIDTH,
               BASE_B_WIDTH, BASE_C_WIDTH, MACRO_WRITE_INPUT_LANES, MAC_LATENCY,
               MODE, A_WIDTH, B_WIDTH, C_WIDTH, SIGNED,
               TILE_INPUT_AXIS_ELEMENTS, TILE_OUTPUT_AXIS_ELEMENTS,
               INPUT_AXIS_TILES, OUTPUT_AXIS_TILES, A_PORT_TILES, B_PORT_TILES,
               C_PORT_TILES, C_BEAT_LAYOUT, RESULT_SLOTS_PER_OUTPUT_TILE>;
  using ABeat = typename Array::ABeat;
  using CBeat = typename Array::CBeat;
  using Set = typename Array::Set;
  using MACRequest = typename Array::MACRequest;
  using WriteRequest = typename Array::WriteRequest;
  using AccumulationWriteRequest =
      BufferWriteRequest<Pack1D<Buffer, OUTPUT_LANES>>;

  static constexpr int LOOP_WIDTH = 10;
  static constexpr int LOOP_LEVEL_COUNT = 2;
  static constexpr int LOOP_SLOT_COUNT = 6;
  static constexpr int LOCAL_ACCUM_CONTEXT_WIDTH =
      ac::nbits<(LOCAL_ACCUM_CONTEXTS > 1 ? LOCAL_ACCUM_CONTEXTS - 1 : 1)>::val;
  using LocalAccumContext = ac_int<LOCAL_ACCUM_CONTEXT_WIDTH, false>;

  // Carry arithmetic inputs and storage selection into the accumulation
  // datapath
  //
  // A matrix command is one accepted MatrixParams command and its full semantic
  // loop. The boundary bit resets every local accum context
  // before the command's first operation
  struct AccumulationMetadata {
    Pack1D<Psum, OUTPUT_LANES> result;
    Pack1D<Buffer, OUTPUT_LANES> initial_value;
    ac_int<1, false> starts_reduction;
    ac_int<1, false> finishes_reduction;
    ac_int<1, false> uses_local_accum_context;
    ac_int<1, false> reads_accumulation_buffer;
    LocalAccumContext local_accum_context;
    ac_int<1, false> buffer_bank;
    ac_int<1, false> begins_matrix_command;

    static const unsigned int width = Pack1D<Psum, OUTPUT_LANES>::width +
                                      Pack1D<Buffer, OUTPUT_LANES>::width + 6 +
                                      LocalAccumContext::width;

    template <unsigned int Size>
    void Marshall(Marshaller<Size>& m) {
      m & result;
      m & initial_value;
      m & starts_reduction;
      m & finishes_reduction;
      m & uses_local_accum_context;
      m & reads_accumulation_buffer;
      m & local_accum_context;
      m & buffer_bank;
      m & begins_matrix_command;
    }

    inline friend void sc_trace(sc_trace_file* tf,
                                const AccumulationMetadata& metadata,
                                const std::string& name) {
      sc_trace(tf, metadata.result, name + ".result");
      sc_trace(tf, metadata.initial_value, name + ".initial_value");
      sc_trace(tf, metadata.starts_reduction, name + ".starts_reduction");
      sc_trace(tf, metadata.finishes_reduction, name + ".finishes_reduction");
      sc_trace(tf, metadata.uses_local_accum_context,
               name + ".uses_local_accum_context");
      sc_trace(tf, metadata.reads_accumulation_buffer,
               name + ".reads_accumulation_buffer");
      sc_trace(tf, metadata.local_accum_context, name + ".local_accum_context");
      sc_trace(tf, metadata.buffer_bank, name + ".buffer_bank");
      sc_trace(tf, metadata.begins_matrix_command,
               name + ".begins_matrix_command");
    }

    inline friend std::ostream& operator<<(
        std::ostream& os, const AccumulationMetadata& metadata) {
      os << metadata.result << " ";
      os << metadata.initial_value << " ";
      os << metadata.starts_reduction << " ";
      os << metadata.finishes_reduction << " ";
      os << metadata.uses_local_accum_context << " ";
      os << metadata.reads_accumulation_buffer << " ";
      os << metadata.local_accum_context << " ";
      os << metadata.buffer_bank << " ";
      os << metadata.begins_matrix_command;
      return os;
    }

    inline friend bool operator==(const AccumulationMetadata& lhs,
                                  const AccumulationMetadata& rhs) {
      return lhs.result == rhs.result &&
             lhs.initial_value == rhs.initial_value &&
             lhs.starts_reduction == rhs.starts_reduction &&
             lhs.finishes_reduction == rhs.finishes_reduction &&
             lhs.uses_local_accum_context == rhs.uses_local_accum_context &&
             lhs.reads_accumulation_buffer == rhs.reads_accumulation_buffer &&
             lhs.local_accum_context == rhs.local_accum_context &&
             lhs.buffer_bank == rhs.buffer_bank &&
             lhs.begins_matrix_command == rhs.begins_matrix_command;
    }
  };

  // One array B-port write
  static constexpr int WEIGHT_WRITE_WIDTH = Array::BBeat::width;
  // One logical B row matches MatrixProcessor's weight-channel convention
  static constexpr int WEIGHT_ROW_WIDTH = OUTPUT_LANES * B_WIDTH;
  // Narrowing the B port splits one row into more array writes
  static constexpr int WEIGHT_BEATS_PER_ROW = OUTPUT_AXIS_TILES / B_PORT_TILES;

  static constexpr int ACCUM_TO_WB_FIFO_DEPTH = cim::ACCUM_TO_WB_FIFO_DEPTH;

  // Carry only the accumulated value; write_back rederives retirement actions
  using AccumulationResult = Pack1D<Buffer, OUTPUT_LANES>;

  static constexpr int OUTPUT_FIFO_DEPTH = cim::OUTPUT_FIFO_DEPTH;
  static constexpr int RESIDENT_SET_COUNT = WEIGHT_SETS;
#if DOUBLE_BUFFERED_ACCUM_BUFFER
  static constexpr int ACCUM_BUFFER_BANKS = 2;
#else
  static constexpr int ACCUM_BUFFER_BANKS = 1;
#endif

  static_assert(std::is_same<Input, Int<Input::width, SIGNED>>::value,
                "CIMProcessor input must be an integer matching CIM_SIGNED");
  static_assert(std::is_same<Weight, Int<Weight::width, SIGNED>>::value,
                "CIMProcessor weight must be an integer matching CIM_SIGNED");
  static_assert(!SUPPORT_MX,
                "CIMProcessor currently does not define microscaling across "
                "the input axis");
  static_assert(!SUPPORT_CODEBOOK_QUANT,
                "CIMProcessor currently does not decode codebook operands");
  // Both macro modes use the same request protocol. The array's ready/credit
  // handshake controls when the processor can issue a MAC. CIMElement derives
  // the issue interval and result latency from the operand widths, base
  // widths, and macro mode, and checks the bit-serial accumulator capacity.
  static_assert(MODE == 0 || MODE == 1,
                "CIMProcessor supports bit-parallel (0) and bit-serial (1) CIM "
                "macros");
  static_assert(C_BEAT_LAYOUT == CIM_C_BEAT_OUTPUT_MAJOR,
                "CIMProcessor currently requires output-major C beats");
  static_assert(
      A_PORT_TILES == INPUT_AXIS_TILES,
      "CIMProcessor currently sends the complete input axis in one A beat");
  static_assert(C_PORT_TILES == OUTPUT_AXIS_TILES,
                "CIMProcessor requires one complete reduced MAC result to fit "
                "in one C beat");
  static_assert(MACRO_WRITE_INPUT_LANES == 1,
                "CIMProcessor currently stores one CIM weight row per request");
  static_assert(WEIGHT_SETS > 0, "CIMProcessor requires a resident weight set");
  static_assert(WEIGHT_SETS <= 0xFFFF,
                "CIM resident set count exceeds schedule metadata");
  static_assert(LOCAL_ACCUM_CONTEXTS > 0,
                "CIMProcessor requires at least one local accum context");
  static_assert(LOCAL_ACCUM_CONTEXTS <= buffer_size,
                "local accum contexts cannot exceed live-output capacity");
  static_assert(INPUT_LANES == Array::INPUT_LANES,
                "CIMProcessor INPUT_LANES must match the CIM input axis");
  // InputController explicitly defines packing, boundary, and
  // replication-unroll rules for these extents
  static_assert(INPUT_LANES == 4 || INPUT_LANES == 8 || INPUT_LANES == 16 ||
                    INPUT_LANES == 32 || INPUT_LANES == 64,
                "CIMProcessor currently requires an InputController-supported "
                "input extent");
  static_assert(OUTPUT_LANES == Array::OUTPUT_LANES,
                "CIMProcessor OUTPUT_LANES must match the CIM output axis");
  static_assert(buffer_size > 0,
                "CIM accumulation buffer depth must be positive");
  static_assert(INPUT_BUFFER_WIDTH == ABeat::width,
                "One input-buffer word must contain one complete CIM A beat");
  static_assert(OUTPUT_AXIS_TILES % B_PORT_TILES == 0,
                "One logical B row must split into whole B-port beats");
  static_assert(INPUT_LANES % MACRO_WRITE_INPUT_LANES == 0,
                "One resident set must hold a whole number of write blocks");
  // The lane order inside a beat is a convention, but its size is not: pin the
  // width to the array geometry so a beat cannot silently hold the wrong lanes
  static_assert(WEIGHT_WRITE_WIDTH == MACRO_WRITE_INPUT_LANES * B_PORT_TILES *
                                          Array::TILE_OUTPUT_LANES * B_WIDTH,
                "A beat must hold exactly the lanes one B-port write covers");
  static_assert(WEIGHT_ROW_WIDTH == WEIGHT_WRITE_WIDTH * WEIGHT_BEATS_PER_ROW /
                                        MACRO_WRITE_INPUT_LANES,
                "Beats must tile one logical B row exactly");

  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<ac_int<INPUT_BUFFER_WIDTH, false>> CCS_INIT_S1(input_channel);
  // Retain MatrixProcessor's weight_channel protocol at physical B-beat width
  Connections::In<ac_int<WEIGHT_WRITE_WIDTH, false>> CCS_INIT_S1(
      weight_channel);
  Connections::In<cim::WeightDescriptor> CCS_INIT_S1(weight_descriptor_channel);
  Connections::In<Pack1D<Buffer, OUTPUT_LANES>> CCS_INIT_S1(bias_channel);
  Connections::In<MatrixParams> CCS_INIT_S1(params_in);

  Connections::Out<Pack1D<Buffer, OUTPUT_LANES>> CCS_INIT_S1(output_channel);
  Connections::Out<ac_int<16, false>>
      accumulation_buffer_read_address[ACCUM_BUFFER_BANKS];
  Connections::In<Pack1D<Buffer, OUTPUT_LANES>>
      accumulation_buffer_read_data[ACCUM_BUFFER_BANKS];
  Connections::Out<AccumulationWriteRequest>
      accumulation_buffer_write_request[ACCUM_BUFFER_BANKS];

#if DOUBLE_BUFFERED_ACCUM_BUFFER
  Connections::SyncOut accumulation_buffer_done[ACCUM_BUFFER_BANKS];
#endif

  Connections::SyncOut CCS_INIT_S1(start);
#if ENABLE_PERF_COUNTERS
  sc_out<MatrixPerformance::Counter>
      perf_counters[MatrixPerformance::PROCESSOR_COUNTER_COUNT];
#endif

 private:
  // Size tags for every queued descriptor plus the active compute descriptor
  static constexpr int WEIGHT_DESCRIPTOR_FIFO_DEPTH = WEIGHT_SETS;
  static constexpr int MAX_LIVE_WEIGHT_DESCRIPTORS =
      WEIGHT_DESCRIPTOR_FIFO_DEPTH + 1;
  static constexpr int WEIGHT_DESCRIPTOR_TAG_WIDTH =
      ac::nbits<MAX_LIVE_WEIGHT_DESCRIPTORS - 1>::val;
  using WeightDescriptorTag = ac_int<WEIGHT_DESCRIPTOR_TAG_WIDTH, false>;

  // Bind one logical set sequence to a contiguous physical ring allocation
  struct ScheduledWeightDescriptor {
    cim::WeightDescriptor logical;
    Set first_set;
    WeightDescriptorTag descriptor_tag;

    static const unsigned int width =
        cim::WeightDescriptor::width + Set::width + WeightDescriptorTag::width;

    template <unsigned int Size>
    void Marshall(Marshaller<Size>& m) {
      m & logical;
      m & first_set;
      m & descriptor_tag;
    }

    inline friend void sc_trace(sc_trace_file* tf,
                                const ScheduledWeightDescriptor& descriptor,
                                const std::string& name) {
      sc_trace(tf, descriptor.logical, name + ".logical");
      sc_trace(tf, descriptor.first_set, name + ".first_set");
      sc_trace(tf, descriptor.descriptor_tag, name + ".descriptor_tag");
    }

    inline friend std::ostream& operator<<(
        std::ostream& os, const ScheduledWeightDescriptor& descriptor) {
      os << descriptor.logical << " " << descriptor.first_set << " "
         << descriptor.descriptor_tag;
      return os;
    }

    inline friend bool operator==(const ScheduledWeightDescriptor& lhs,
                                  const ScheduledWeightDescriptor& rhs) {
      return lhs.logical == rhs.logical && lhs.first_set == rhs.first_set &&
             lhs.descriptor_tag == rhs.descriptor_tag;
    }
  };

  // Tag one per-set state transition with its descriptor ownership
  struct ResidentSetEvent {
    Set set;
    WeightDescriptorTag descriptor_tag;

    static const unsigned int width = Set::width + WeightDescriptorTag::width;

    template <unsigned int Size>
    void Marshall(Marshaller<Size>& m) {
      m & set;
      m & descriptor_tag;
    }

    inline friend void sc_trace(sc_trace_file* tf,
                                const ResidentSetEvent& event,
                                const std::string& name) {
      sc_trace(tf, event.set, name + ".set");
      sc_trace(tf, event.descriptor_tag, name + ".descriptor_tag");
    }

    inline friend std::ostream& operator<<(std::ostream& os,
                                           const ResidentSetEvent& event) {
      os << event.set << " " << event.descriptor_tag;
      return os;
    }

    inline friend bool operator==(const ResidentSetEvent& lhs,
                                  const ResidentSetEvent& rhs) {
      return lhs.set == rhs.set && lhs.descriptor_tag == rhs.descriptor_tag;
    }
  };

  // Track whether one set accepts a fill or is retained for descriptor replays
  enum ResidentSetState { SET_FREE = 0, SET_READY = 1 };
  using ResidentSetStateBits = ac_int<1, false>;

  Array CCS_INIT_S1(cim_array);
  Connections::Combinational<MACRequest, Connections::SYN_PORT> CCS_INIT_S1(
      mac_request_channel);
  Connections::Combinational<WriteRequest> CCS_INIT_S1(write_request_channel);
  Connections::Combinational<CBeat, Connections::SYN_PORT> CCS_INIT_S1(
      result_channel);

  // Isolate CIM result retirement from the accumulation pipeline
  Connections::Combinational<Pack1D<Psum, OUTPUT_LANES>> CCS_INIT_S1(
      result_to_accum_channel);

  // Decouple accumulator reads from their ordered response/addition stage
  Connections::Fifo<AccumulationMetadata, cim::ACCUM_METADATA_FIFO_DEPTH>
      CCS_INIT_S1(accumulation_metadata_fifo);
  Connections::Combinational<AccumulationMetadata> CCS_INIT_S1(
      accumulation_metadata_enq);
  Connections::Combinational<AccumulationMetadata> CCS_INIT_S1(
      accumulation_metadata_deq);

  // Match MatrixProcessor's accumulation-to-write-back decoupling
  Connections::Fifo<AccumulationResult, ACCUM_TO_WB_FIFO_DEPTH> CCS_INIT_S1(
      accum_to_wb_fifo);
  Connections::Combinational<AccumulationResult> CCS_INIT_S1(accum_to_wb_enq);
  Connections::Combinational<AccumulationResult> CCS_INIT_S1(accum_to_wb_deq);

#ifndef __SYNTHESIS__
  // Diagnose read-after-write hazards at the SRAM feedback boundary in
  // simulation. This reference exists only to localize stale partial-sum reads,
  // not to replace final-output checking or synthesize hardware dependency
  // checks
  struct AccumulationReadCheck {
    unsigned address;
    unsigned bank;
    Pack1D<Buffer, OUTPUT_LANES> previous;
  };
  std::vector<Pack1D<Buffer, OUTPUT_LANES>>
      simulation_partial_sums[ACCUM_BUFFER_BANKS];
  std::deque<AccumulationReadCheck> simulation_accumulation_reads;
#endif

  // Match MatrixProcessor's final-output decoupling after accumulation
  Connections::Fifo<Pack1D<Buffer, OUTPUT_LANES>, OUTPUT_FIFO_DEPTH>
      CCS_INIT_S1(accum_output_fifo);
  Connections::Combinational<Pack1D<Buffer, OUTPUT_LANES>> CCS_INIT_S1(
      accum_output_enq);

  // Buffer one ring traversal of scheduled descriptor assignments
  Connections::Fifo<ScheduledWeightDescriptor, WEIGHT_DESCRIPTOR_FIFO_DEPTH>
      CCS_INIT_S1(scheduled_weight_descriptor_fifo);
  Connections::Combinational<ScheduledWeightDescriptor> CCS_INIT_S1(
      scheduled_weight_descriptor_enq);
  Connections::Combinational<ScheduledWeightDescriptor> CCS_INIT_S1(
      scheduled_weight_descriptor_deq);

  // Buffer descriptors independently for the continuous weight-load loop
  Connections::Fifo<ScheduledWeightDescriptor, WEIGHT_DESCRIPTOR_FIFO_DEPTH>
      CCS_INIT_S1(loader_weight_descriptor_fifo);
  Connections::Combinational<ScheduledWeightDescriptor> CCS_INIT_S1(
      loader_weight_descriptor_enq);
  Connections::Combinational<ScheduledWeightDescriptor> CCS_INIT_S1(
      loader_weight_descriptor_deq);

  // Centralize per-set state updates from the loader and compute scheduler
  sc_signal<ResidentSetStateBits> resident_set_state[WEIGHT_SETS];
  sc_signal<WeightDescriptorTag> resident_set_owner_tag[WEIGHT_SETS];
  Connections::Combinational<ResidentSetEvent> CCS_INIT_S1(set_ready_channel);
  Connections::Combinational<ResidentSetEvent> CCS_INIT_S1(set_release_channel);

  Connections::Fifo<MatrixParams, 1> CCS_INIT_S1(
      process_accumulation_params_fifo);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(
      process_accumulation_params_enq);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(
      process_accumulation_params_deq);

  Connections::Fifo<MatrixParams, 1> CCS_INIT_S1(collect_results_params_fifo);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(
      collect_results_params_enq);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(
      collect_results_params_deq);

  Connections::Fifo<MatrixParams, 1> CCS_INIT_S1(write_back_params_fifo);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(write_back_params_enq);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(write_back_params_deq);

#if ENABLE_PERF_COUNTERS
  sc_signal<bool> perf_completion_toggle;
  sc_signal<MatrixPerformance::Counter> perf_mac_wait_weight_set_load_cycles;
#endif

 public:
  // Construct the array and the independent issue/result controllers
  SC_CTOR(CIMProcessor) {
    cim_array.clk(clk);
    cim_array.rstn(rstn);
    cim_array.mac_request_channel(mac_request_channel);
    cim_array.write_request_channel(write_request_channel);
    cim_array.result_channel(result_channel);

    accum_to_wb_fifo.clk(clk);
    accum_to_wb_fifo.rst(rstn);
    accum_to_wb_fifo.enq(accum_to_wb_enq);
    accum_to_wb_fifo.deq(accum_to_wb_deq);

    accum_output_fifo.clk(clk);
    accum_output_fifo.rst(rstn);
    accum_output_fifo.enq(accum_output_enq);
    accum_output_fifo.deq(output_channel);

    accumulation_metadata_fifo.clk(clk);
    accumulation_metadata_fifo.rst(rstn);
    accumulation_metadata_fifo.enq(accumulation_metadata_enq);
    accumulation_metadata_fifo.deq(accumulation_metadata_deq);

    scheduled_weight_descriptor_fifo.clk(clk);
    scheduled_weight_descriptor_fifo.rst(rstn);
    scheduled_weight_descriptor_fifo.enq(scheduled_weight_descriptor_enq);
    scheduled_weight_descriptor_fifo.deq(scheduled_weight_descriptor_deq);

    loader_weight_descriptor_fifo.clk(clk);
    loader_weight_descriptor_fifo.rst(rstn);
    loader_weight_descriptor_fifo.enq(loader_weight_descriptor_enq);
    loader_weight_descriptor_fifo.deq(loader_weight_descriptor_deq);

    process_accumulation_params_fifo.clk(clk);
    process_accumulation_params_fifo.rst(rstn);
    process_accumulation_params_fifo.enq(process_accumulation_params_enq);
    process_accumulation_params_fifo.deq(process_accumulation_params_deq);

    collect_results_params_fifo.clk(clk);
    collect_results_params_fifo.rst(rstn);
    collect_results_params_fifo.enq(collect_results_params_enq);
    collect_results_params_fifo.deq(collect_results_params_deq);

    write_back_params_fifo.clk(clk);
    write_back_params_fifo.rst(rstn);
    write_back_params_fifo.enq(write_back_params_enq);
    write_back_params_fifo.deq(write_back_params_deq);

    SC_THREAD(set_scoreboard);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(issue_operations);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(schedule_weight_descriptors);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(load_weights);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(collect_results);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(issue_accumulation_reads);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(complete_accumulation);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(write_back);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

#if ENABLE_PERF_COUNTERS
    SC_THREAD(monitor_performance);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
#endif
  }

 private:
  // Return the operation volume; fixed-unit FX@L2 occupies omitted slot 5
  static ac_int<32, false> total_operations(const MatrixParams& params) {
    return params.loops[0][0] * params.loops[0][1] * params.loops[0][2] *
           params.loops[0][3] * params.loops[0][4] * params.loops[1][0] *
           params.loops[1][1] * params.loops[1][2] * params.loops[1][3] *
           params.loops[1][4] * params.loops[1][5];
  }

  // Reset both physical counter levels before walking one semantic schedule
  static void reset_loop_counters(
      ac_int<LOOP_WIDTH, false> loop_counters[2][6]) {
#pragma hls_unroll yes
    for (int level = 0; level < LOOP_LEVEL_COUNT; level++) {
#pragma hls_unroll yes
      for (int slot = 0; slot < LOOP_SLOT_COUNT; slot++) {
        loop_counters[level][slot] = 0;
      }
    }
  }

  // Advance the shared two-level loop-counter representation by one operation
  static void advance_loop_counters(
      ac_int<LOOP_WIDTH, false> loop_counters[2][6],
      const MatrixParams& params) {
    loop_counters[LOOP_LEVEL_COUNT - 1][LOOP_SLOT_COUNT - 1]++;
#pragma hls_unroll yes
    for (int level = LOOP_LEVEL_COUNT - 1; level >= 0; level--) {
#pragma hls_unroll yes
      for (int slot = LOOP_SLOT_COUNT - 1; slot >= 0; slot--) {
        if (loop_counters[level][slot] == params.loops[level][slot]) {
          loop_counters[level][slot] = 0;
          if (slot > 0) {
            loop_counters[level][slot - 1]++;
          } else if (level > 0) {
            loop_counters[level - 1][LOOP_SLOT_COUNT - 1]++;
          }
        }
      }
    }
  }

  // Return whether counters are at a resident-weight-set run boundary
  //
  // The boundary starts a run before issuing a MAC and ends it after counters
  // advance; reused L1 output dimensions must all wrap to zero
  static bool weight_set_boundary(
      const MatrixParams& params,
      const ac_int<LOOP_WIDTH, false> loop_counters[2][6],
      bool l1_ox_reuses_weights, bool l1_oy_reuses_weights) {
    return (!l1_ox_reuses_weights ||
            loop_counters[1][params.x_loop_idx[1]] == 0) &&
           (!l1_oy_reuses_weights ||
            loop_counters[1][params.y_loop_idx[1]] == 0);
  }

  // Count SRAM addresses for partial sums and buffered final outputs
  //
  // An outer output dimension contributes only when a reduction loop
  // encloses it because those coordinates remain live across later terms
  static ac_int<32, false> accumulation_footprint(const MatrixParams& params) {
    const auto l2_ic = params.loops[0][params.reduction_loop_idx[0]];
    const auto l2_fy = params.loops[0][params.fy_loop_idx[0]];
    const bool includes_outer_ox =
        (l2_ic > 1 && params.reduction_loop_idx[0] < params.x_loop_idx[0]) ||
        (l2_fy > 1 && params.fy_loop_idx[0] < params.x_loop_idx[0]);
    const bool includes_outer_oy =
        (l2_ic > 1 && params.reduction_loop_idx[0] < params.y_loop_idx[0]) ||
        (l2_fy > 1 && params.fy_loop_idx[0] < params.y_loop_idx[0]);
    ac_int<32, false> x_extent = params.loops[1][params.x_loop_idx[1]];
    ac_int<32, false> y_extent = params.loops[1][params.y_loop_idx[1]];
    if (includes_outer_ox) x_extent *= params.loops[0][params.x_loop_idx[0]];
    if (includes_outer_oy) y_extent *= params.loops[0][params.y_loop_idx[0]];
    return params.loops[1][params.weight_loop_idx[1]] * y_extent * x_extent;
  }

  // Commit one ordered weight-channel beat to the selected resident set
  // WEIGHT CHANNEL CONTRACT. Width is checked, but stream and lane order are
  // conventions shared with WeightController and must change on both sides
  //
  // One resident set arrives in row-major order, with output-axis spans inside
  // each input-axis row:
  //
  //   [ input_index=0 span=0 ][ input_index=0 span=1 ] ... [
  //   input_index=INPUT_LANES-1 span=last ]
  //
  //   bit 0                              WEIGHT_WRITE_WIDTH - 1
  //   |                                                         |
  //   [ port tile 0 ][ port tile 1 ] ... [ port tile B_PORT_TILES-1 ]
  //       [out0][out1]... [out0][out1]...  TILE_OUTPUT_LANES per tile
  //
  //   request.data[port_tile][0][tile_output_index]
  //       = beat[(port_tile * TILE_OUTPUT_LANES + tile_output_index) * B_WIDTH]
  //   request.output_axis_tile_base = span * B_PORT_TILES
  //
  // MACRO_WRITE_INPUT_LANES is currently constrained to one, so every request
  // writes one input-axis row across B_PORT_TILES output tiles
  void write_weight_beat(Set write_set, int input_index, int span,
                         const ac_int<WEIGHT_WRITE_WIDTH, false>& beat) {
    WriteRequest request;
    request.write_set = write_set;
    request.input_tile_index = input_index / Array::TILE_INPUT_LANES;
    request.output_axis_tile_base = span * B_PORT_TILES;
    request.write_input_index = input_index % Array::TILE_INPUT_LANES;
    request.replicate = 0;
    pack::clear_pack(request.data);

#pragma hls_unroll yes
    for (int port_tile_idx = 0; port_tile_idx < B_PORT_TILES; port_tile_idx++) {
#pragma hls_unroll yes
      for (int tile_output_index = 0;
           tile_output_index < Array::TILE_OUTPUT_LANES; tile_output_index++) {
        const int lane =
            port_tile_idx * Array::TILE_OUTPUT_LANES + tile_output_index;
        request.data[port_tile_idx][0][tile_output_index] =
            beat.template slc<B_WIDTH>(lane * B_WIDTH);
      }
    }
    write_request_channel.Push(request);
  }

  // Advance one physical set index around the resident-set ring
  static Set next_resident_set(Set current) {
    return current == Set(WEIGHT_SETS - 1) ? Set(0) : Set(current + 1);
  }

  // Select one tile-relative physical set around the resident-set ring
  static Set resident_set_at_offset(Set first, ac_int<16, false> offset) {
    ac_int<17, false> index = first;
    index += offset;
    if (index >= WEIGHT_SETS) {
      index -= WEIGHT_SETS;
    }
    return Set(index);
  }

  // Return whether the selected physical set currently has the requested state
  bool resident_set_has_state(Set selected, ResidentSetState expected) const {
    bool matches = false;
#pragma hls_unroll yes
    for (int set = 0; set < WEIGHT_SETS; set++) {
      if (selected == Set(set)) {
        matches = resident_set_state[set].read() == expected;
      }
    }
    return matches;
  }

  // Return whether the selected set is ready for this descriptor
  bool resident_set_is_ready_for(Set selected,
                                 WeightDescriptorTag descriptor_tag) const {
    bool matches = false;
#pragma hls_unroll yes
    for (int set = 0; set < WEIGHT_SETS; set++) {
      if (selected == Set(set)) {
        matches = resident_set_state[set].read() == SET_READY &&
                  resident_set_owner_tag[set].read() == descriptor_tag;
      }
    }
    return matches;
  }

  // Track FREE -> READY after fill and READY -> FREE after the final replay
  void set_scoreboard() {
    set_ready_channel.ResetRead();
    set_release_channel.ResetRead();
#pragma hls_unroll yes
    for (int set = 0; set < WEIGHT_SETS; set++) {
      resident_set_state[set].write(ResidentSetStateBits(SET_FREE));
      resident_set_owner_tag[set].write(0);
    }

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      ResidentSetEvent ready_event;
      ResidentSetEvent release_event;
      const bool became_ready = set_ready_channel.PopNB(ready_event);
      const bool released = set_release_channel.PopNB(release_event);
#ifndef __SYNTHESIS__
      if (became_ready && !resident_set_has_state(ready_event.set, SET_FREE)) {
        SC_REPORT_FATAL("CIMProcessor", "completed resident set is not free");
      }
      if (released && !resident_set_is_ready_for(
                          release_event.set, release_event.descriptor_tag)) {
        SC_REPORT_FATAL("CIMProcessor",
                        "released resident set ownership does not match");
      }
#endif

#pragma hls_unroll yes
      for (int set = 0; set < WEIGHT_SETS; set++) {
        ResidentSetStateBits next = resident_set_state[set].read();
        WeightDescriptorTag next_descriptor_tag =
            resident_set_owner_tag[set].read();
        if (released && release_event.set == Set(set) &&
            release_event.descriptor_tag == next_descriptor_tag) {
          next = SET_FREE;
        }
        if (became_ready && ready_event.set == Set(set)) {
          next = SET_READY;
          next_descriptor_tag = ready_event.descriptor_tag;
        }
        resident_set_state[set].write(next);
        resident_set_owner_tag[set].write(next_descriptor_tag);
      }
      wait();
    }
  }

  // Assign descriptor sequences to the resident ring ahead of their payloads
  void schedule_weight_descriptors() {
    weight_descriptor_channel.Reset();
    scheduled_weight_descriptor_enq.ResetWrite();
    loader_weight_descriptor_enq.ResetWrite();

    wait();

    Set next_set = 0;
    WeightDescriptorTag next_descriptor_tag = 0;
    while (true) {
      const cim::WeightDescriptor descriptor = weight_descriptor_channel.Pop();

#ifndef __SYNTHESIS__
      if (descriptor.set_count == 0 || descriptor.set_count > WEIGHT_SETS ||
          descriptor.replay_count == 0) {
        SC_REPORT_FATAL("CIMProcessor", "invalid resident weight sequence");
      }
#endif

      ScheduledWeightDescriptor scheduled;
      scheduled.logical = descriptor;
      scheduled.first_set = next_set;
      scheduled.descriptor_tag = next_descriptor_tag;
      scheduled_weight_descriptor_enq.Push(scheduled);
      loader_weight_descriptor_enq.Push(scheduled);
      next_set = resident_set_at_offset(next_set, descriptor.set_count);
      next_descriptor_tag++;
    }
  }

  // Fill resident sets without draining the payload pipeline at descriptors
  void load_weights() {
    weight_channel.Reset();
    loader_weight_descriptor_deq.ResetRead();
    write_request_channel.ResetWrite();
    set_ready_channel.ResetWrite();

    wait();

    ScheduledWeightDescriptor current;
    ScheduledWeightDescriptor pending;
    bool current_valid = false;
    bool pending_valid = false;
    ac_int<16, false> sets_remaining = 0;
    int input_index = 0;
    int span = 0;
    Set write_set = 0;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      ScheduledWeightDescriptor next;
      if (!pending_valid && loader_weight_descriptor_deq.PopNB(next)) {
        pending = next;
        pending_valid = true;
      }
      if (!current_valid) {
        if (!pending_valid) {
          wait();
          continue;
        }
        current = pending;
        current_valid = true;
        pending_valid = false;
        sets_remaining = current.logical.set_count;
        write_set = current.first_set;
      }

      // Only the first payload beat needs to wait for ownership of this set
      if (input_index == 0 && span == 0) {
        while (!resident_set_has_state(write_set, SET_FREE)) {
          wait();
        }
      }

      const ac_int<WEIGHT_WRITE_WIDTH, false> beat = weight_channel.Pop();
      write_weight_beat(write_set, input_index, span, beat);

      const bool row_complete = span == WEIGHT_BEATS_PER_ROW - 1;
      const bool set_complete = row_complete && input_index == INPUT_LANES - 1;
      if (set_complete) {
        ResidentSetEvent ready_event;
        ready_event.set = write_set;
        ready_event.descriptor_tag = current.descriptor_tag;
        set_ready_channel.Push(ready_event);
        if (sets_remaining == 1) {
          current_valid = false;
        } else {
          sets_remaining--;
          write_set = next_resident_set(write_set);
        }
        input_index = 0;
        span = 0;
      } else if (row_complete) {
        input_index++;
        span = 0;
      } else {
        span++;
      }
    }
  }

  // Pack one complete A beat from an input-buffer word
  static ABeat pack_a_beat(const ac_int<INPUT_BUFFER_WIDTH, false>& inputs) {
    ABeat beat;
    pack::clear_pack(beat);
#pragma hls_unroll yes
    for (int input_index = 0; input_index < INPUT_LANES; input_index++) {
      const int input_tile_index = input_index / Array::TILE_INPUT_LANES;
      const int tile_input_index = input_index % Array::TILE_INPUT_LANES;
      beat[input_tile_index][tile_input_index] =
          inputs.template slc<A_WIDTH>(input_index * A_WIDTH);
    }
    return beat;
  }

  // Consume resident-set descriptors while issuing the full compute schedule
  //
  // A descriptor is the only weight-lifetime policy seen here: it names a
  // consecutive set sequence and how many times that sequence is replayed
  // before its sets can be released
  void issue_operations() {
    params_in.Reset();
    input_channel.Reset();
    process_accumulation_params_enq.ResetWrite();
    collect_results_params_enq.ResetWrite();
    write_back_params_enq.ResetWrite();
    mac_request_channel.ResetWrite();
    scheduled_weight_descriptor_deq.ResetRead();
    set_release_channel.ResetWrite();
    start.Reset();
#if ENABLE_PERF_COUNTERS
    MatrixPerformance::Counter mac_wait_weight_set_load_cycles = 0;
    perf_mac_wait_weight_set_load_cycles.write(0);
#endif

    wait();

    while (true) {
      const MatrixParams params = params_in.Pop();
      process_accumulation_params_enq.Push(params);
      collect_results_params_enq.Push(params);
      write_back_params_enq.Push(params);
      start.SyncPush();

#ifndef __SYNTHESIS__
      if (params.use_input_codebook || params.use_weight_codebook) {
        SC_REPORT_FATAL("CIMProcessor",
                        "currently does not support codebook operands");
      }
      if (params.is_mx_op || params.is_resnet_replication ||
          params.is_generic_replication) {
        SC_REPORT_FATAL("CIMProcessor",
                        "currently does not support MX or replicated operands");
      }
      const auto l2_ic = params.loops[0][params.reduction_loop_idx[0]];
      const auto l2_fy = params.loops[0][params.fy_loop_idx[0]];
      const bool outer_oc_partial_context =
          params.loops[0][params.weight_loop_idx[0]] > 1 &&
          ((l2_ic > 1 &&
            params.reduction_loop_idx[0] < params.weight_loop_idx[0]) ||
           (l2_fy > 1 && params.fy_loop_idx[0] < params.weight_loop_idx[0]));
      if (outer_oc_partial_context) {
        SC_REPORT_FATAL("CIMProcessor",
                        "outer output-column partial contexts are unsupported");
      }
#if DOUBLE_BUFFERED_ACCUM_BUFFER
      const bool l2_output_context =
          (l2_ic > 1 &&
           (params.reduction_loop_idx[0] < params.x_loop_idx[0] ||
            params.reduction_loop_idx[0] < params.y_loop_idx[0])) ||
          (l2_fy > 1 && (params.fy_loop_idx[0] < params.x_loop_idx[0] ||
                         params.fy_loop_idx[0] < params.y_loop_idx[0]));
      if (l2_output_context && params.write_output_to_accum_buffer) {
        SC_REPORT_FATAL(
            "CIMProcessor",
            "outer partial contexts cannot use banked accumulation output");
      }
#endif
      if (accumulation_footprint(params) > buffer_size) {
        SC_REPORT_FATAL("CIMProcessor",
                        "live partial outputs exceed accumulation capacity");
      }
#endif

      ac_int<LOOP_WIDTH, false> loop_counters[2][6];
      reset_loop_counters(loop_counters);

      const ac_int<32, false> total_ops = total_operations(params);
      const auto l1_oc = params.loops[1][params.weight_loop_idx[1]];
      const auto l1_ic = params.loops[1][params.reduction_loop_idx[1]];
      const auto l1_fy = params.loops[1][params.fy_loop_idx[1]];
      const auto l1_fx = params.loops[1][params.fx_loop_idx];
      const bool l1_ox_reuses_weights =
          (l1_oc == 1 || params.weight_loop_idx[1] < params.x_loop_idx[1]) &&
          (l1_ic == 1 || params.reduction_loop_idx[1] < params.x_loop_idx[1]) &&
          (l1_fy == 1 || params.fy_loop_idx[1] < params.x_loop_idx[1]) &&
          (l1_fx == 1 || params.fx_loop_idx < params.x_loop_idx[1]);
      const bool l1_oy_reuses_weights =
          (l1_oc == 1 || params.weight_loop_idx[1] < params.y_loop_idx[1]) &&
          (l1_ic == 1 || params.reduction_loop_idx[1] < params.y_loop_idx[1]) &&
          (l1_fy == 1 || params.fy_loop_idx[1] < params.y_loop_idx[1]) &&
          (l1_fx == 1 || params.fx_loop_idx < params.y_loop_idx[1]);
      // Walk the current descriptor in [replay][set] order
      ScheduledWeightDescriptor descriptor;
      Set selected_weight_set = 0;
      ac_int<16, false> set_index = 0;
      ac_int<16, false> replay_index = 0;
      bool sequence_complete = true;
      bool selected_set_is_on_final_replay = false;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (ac_int<32, false> step = 0; step < total_ops; step++) {
        if (weight_set_boundary(params, loop_counters, l1_ox_reuses_weights,
                                l1_oy_reuses_weights)) {
          if (sequence_complete) {
            descriptor = scheduled_weight_descriptor_deq.Pop();

#ifndef __SYNTHESIS__
            if (descriptor.logical.set_count == 0 ||
                descriptor.logical.set_count > WEIGHT_SETS ||
                descriptor.first_set >= WEIGHT_SETS ||
                descriptor.logical.replay_count == 0) {
              SC_REPORT_FATAL("CIMProcessor",
                              "invalid scheduled weight descriptor");
            }
#endif

            set_index = 0;
            replay_index = 0;
            sequence_complete = false;
          }

          selected_weight_set =
              resident_set_at_offset(descriptor.first_set, set_index);
          selected_set_is_on_final_replay =
              replay_index == descriptor.logical.replay_count - 1;
          if (replay_index == 0) {
            while (!resident_set_is_ready_for(selected_weight_set,
                                              descriptor.descriptor_tag)) {
#if ENABLE_PERF_COUNTERS
              ++mac_wait_weight_set_load_cycles;
#endif
              wait();
            }
          }

          set_index++;
          if (set_index == descriptor.logical.set_count) {
            set_index = 0;
            replay_index++;
            if (replay_index == descriptor.logical.replay_count) {
              replay_index = 0;
              sequence_complete = true;
            }
          }
        }

        MACRequest request;
        request.a = pack_a_beat(input_channel.Pop());
        request.compute_set = selected_weight_set;
        // Multicast each A beat to every tile along the output axis.
        // output_tile_index is ignored for multicast requests; zero is its
        // canonical unused value
        request.output_tile_index = 0;
        request.multicast = 1;
        // Reduction leaves one result per output-axis tile, and the C-port
        // assertion packs all OUTPUT_LANES values in one beat
        request.reduce = 1;
        // The synthesized channel can buffer this request before physical
        // issue, so the array protects both waiting and active requests.
        mac_request_channel.Push(request);

        advance_loop_counters(loop_counters, params);
        const bool finishes_set_run =
            step + 1 == total_ops ||
            weight_set_boundary(params, loop_counters, l1_ox_reuses_weights,
                                l1_oy_reuses_weights);
        if (selected_set_is_on_final_replay && finishes_set_run) {
          // The array protects this set until the pending MAC and its physical
          // issue window have both finished.
          ResidentSetEvent release_event;
          release_event.set = selected_weight_set;
          release_event.descriptor_tag = descriptor.descriptor_tag;
          set_release_channel.Push(release_event);
        }
      }

#ifndef __SYNTHESIS__
      if (!sequence_complete) {
        SC_REPORT_FATAL(
            "CIMProcessor",
            "matrix job ended before completing its weight sequence");
      }
#endif
#if ENABLE_PERF_COUNTERS
      // Publish once per issued command. Per-cycle signal writes inside the
      // weight-readiness loop would constrain the HLS datapath schedule.
      perf_mac_wait_weight_set_load_cycles.write(
          mac_wait_weight_set_load_cycles);
#endif
    }
  }

  // Flatten SRAM output addresses as [OC@L1][combined OY][combined OX]
  //
  // OX is innermost, OY is next, and OC@L1 is outermost, so the index is
  // ((oc * combined_oy_extent) + oy) * combined_ox_extent + ox
  //
  // Outer OX/OY participates only when nested inside an unfinished reduction;
  // otherwise that outer loop completes one output before starting the next and
  // may reuse the same accumulation-buffer address
  static ac_int<16, false> output_context_index(
      const MatrixParams& params,
      const ac_int<LOOP_WIDTH, false> loop_counters[2][6]) {
    // FX@L2 is fixed to one, so only IC/FY can keep an outer output live
    const bool includes_outer_ox =
        (params.loops[0][params.reduction_loop_idx[0]] > 1 &&
         params.reduction_loop_idx[0] < params.x_loop_idx[0]) ||
        (params.loops[0][params.fy_loop_idx[0]] > 1 &&
         params.fy_loop_idx[0] < params.x_loop_idx[0]);
    const bool includes_outer_oy =
        (params.loops[0][params.reduction_loop_idx[0]] > 1 &&
         params.reduction_loop_idx[0] < params.y_loop_idx[0]) ||
        (params.loops[0][params.fy_loop_idx[0]] > 1 &&
         params.fy_loop_idx[0] < params.y_loop_idx[0]);
    const ac_int<16, false> x0_extent = params.loops[1][params.x_loop_idx[1]];
    const ac_int<16, false> y0_extent = params.loops[1][params.y_loop_idx[1]];
    ac_int<16, false> x_extent = x0_extent;
    ac_int<16, false> y_extent = y0_extent;
    ac_int<16, false> x = loop_counters[1][params.x_loop_idx[1]];
    ac_int<16, false> y = loop_counters[1][params.y_loop_idx[1]];
    if (includes_outer_ox) {
      x_extent *= params.loops[0][params.x_loop_idx[0]];
      x += loop_counters[0][params.x_loop_idx[0]] * x0_extent;
    }
    if (includes_outer_oy) {
      y_extent *= params.loops[0][params.y_loop_idx[0]];
      y += loop_counters[0][params.y_loop_idx[0]] * y0_extent;
    }
    return loop_counters[1][params.weight_loop_idx[1]] * y_extent * x_extent +
           y * x_extent + x;
  }

  // Keep only output coordinates enclosed by a nonunit reduction
  static bool output_loop_is_live(const MatrixParams& params, int level,
                                  int position) {
    const bool outer =
        (params.loops[0][params.reduction_loop_idx[0]] > 1 &&
         (level == 1 || params.reduction_loop_idx[0] < position)) ||
        (params.loops[0][params.fy_loop_idx[0]] > 1 &&
         (level == 1 || params.fy_loop_idx[0] < position));
    return outer ||
           (level == 1 && ((params.loops[1][params.reduction_loop_idx[1]] > 1 &&
                            params.reduction_loop_idx[1] < position) ||
                           (params.loops[1][params.fx_loop_idx] > 1 &&
                            params.fx_loop_idx < position) ||
                           (params.loops[1][params.fy_loop_idx[1]] > 1 &&
                            params.fy_loop_idx[1] < position)));
  }

  // Bind registers to an inner reduction that fits, then spill at its end.
  // A later outer reduction reloads the partial sum from its full SRAM address.
  struct LocalAccumLayout {
    ac_int<LOOP_WIDTH, false> extents[2][3];
    // Physical slots run outermost first; 12 means no fitting reduction.
    ac_int<4, false> reduction_boundary;
  };

  // Prefer the largest inner reduction whose live outputs fit in registers.
  // If even the innermost reduction is wider, retain the existing first-context
  // optimization; those remaining SRAM dependencies have the wider spacing.
  static LocalAccumLayout local_accum_layout(const MatrixParams& params) {
    LocalAccumLayout layout;
    layout.reduction_boundary = LOOP_LEVEL_COUNT * LOOP_SLOT_COUNT;
    ac_int<32, false> outputs = 1;
#pragma hls_unroll yes
    for (int level = LOOP_LEVEL_COUNT - 1; level >= 0; level--) {
#pragma hls_unroll yes
      for (int slot = LOOP_SLOT_COUNT - 1; slot >= 0; slot--) {
        const bool reduction = slot == params.reduction_loop_idx[level] ||
                               slot == params.fy_loop_idx[level] ||
                               (level == 1 && slot == params.fx_loop_idx);
        const bool output = slot == params.x_loop_idx[level] ||
                            slot == params.y_loop_idx[level] ||
                            slot == params.weight_loop_idx[level];
        if (output && outputs <= LOCAL_ACCUM_CONTEXTS) {
          outputs *= params.loops[level][slot];
        }
        if (reduction && params.loops[level][slot] > 1 &&
            outputs <= LOCAL_ACCUM_CONTEXTS) {
          layout.reduction_boundary = level * LOOP_SLOT_COUNT + slot;
        }
      }
    }
#pragma hls_unroll yes
    for (int level = 0; level < LOOP_LEVEL_COUNT; level++) {
      const ac_int<3, false> positions[3] = {params.weight_loop_idx[level],
                                             params.y_loop_idx[level],
                                             params.x_loop_idx[level]};
#pragma hls_unroll yes
      for (int axis = 0; axis < 3; axis++) {
        const bool live = layout.reduction_boundary <
                                  LOOP_LEVEL_COUNT * LOOP_SLOT_COUNT
                              ? level * LOOP_SLOT_COUNT + positions[axis] >
                                    layout.reduction_boundary
                              : output_loop_is_live(params, level,
                                                    positions[axis]);
        layout.extents[level][axis] =
            (level != 0 || axis != 0) && live
                ? ac_int<LOOP_WIDTH, false>(
                      params.loops[level][positions[axis]])
                : ac_int<LOOP_WIDTH, false>(1);
      }
    }
    return layout;
  }

  // First/last contribution inside the local reduction, not the full command.
  static bool local_reduction_boundary(
      const MatrixParams& params, const LocalAccumLayout& layout,
      const ac_int<LOOP_WIDTH, false> loop_counters[2][6], bool last) {
    bool boundary = true;
#pragma hls_unroll yes
    for (int level = 0; level < LOOP_LEVEL_COUNT; level++) {
      const ac_int<3, false> positions[3] = {
          params.reduction_loop_idx[level], params.fy_loop_idx[level],
          params.fx_loop_idx};
#pragma hls_unroll yes
      for (int reduction = 0; reduction < 3; reduction++) {
        const auto position = positions[reduction];
        if ((reduction != 2 || level == 1) &&
            level * LOOP_SLOT_COUNT + position >=
                layout.reduction_boundary) {
          const ac_int<LOOP_WIDTH, false> final_index =
              params.loops[level][position] - 1;
          boundary = boundary &&
                     (last ? loop_counters[level][position] == final_index
                           : loop_counters[level][position] == 0);
        }
      }
    }
    return boundary;
  }

  // Outer output coordinates select SRAM addresses, but may reuse a register
  // after the preceding local reduction's value has been captured for writeback.
  static ac_int<16, false> local_accum_context_index(
      const MatrixParams& params, const LocalAccumLayout& layout,
      const ac_int<LOOP_WIDTH, false> loop_counters[2][6]) {
    ac_int<16, false> index = 0;
#pragma hls_unroll yes
    for (int axis = 0; axis < 3; axis++) {
#pragma hls_unroll yes
      for (int level = 0; level < LOOP_LEVEL_COUNT; level++) {
        const ac_int<3, false> positions[3] = {params.weight_loop_idx[level],
                                               params.y_loop_idx[level],
                                               params.x_loop_idx[level]};
        const auto extent = layout.extents[level][axis];
        index =
            index * extent + (extent > 1 ? loop_counters[level][positions[axis]]
                                         : ac_int<LOOP_WIDTH, false>(0));
      }
    }
    return index;
  }

  // Return whether this is the first IC/FX/FY contribution to one output
  static bool starts_output_reduction(
      const MatrixParams& params,
      const ac_int<LOOP_WIDTH, false> loop_counters[2][6]) {
    return loop_counters[0][params.reduction_loop_idx[0]] == 0 &&
           loop_counters[1][params.reduction_loop_idx[1]] == 0 &&
           loop_counters[1][params.fx_loop_idx] == 0 &&
           loop_counters[0][params.fy_loop_idx[0]] == 0 &&
           loop_counters[1][params.fy_loop_idx[1]] == 0;
  }

  // Return whether this is the final IC/FX/FY contribution to one output
  static bool finishes_output_reduction(
      const MatrixParams& params,
      const ac_int<LOOP_WIDTH, false> loop_counters[2][6]) {
    return loop_counters[0][params.reduction_loop_idx[0]] ==
               params.loops[0][params.reduction_loop_idx[0]] - 1 &&
           loop_counters[1][params.reduction_loop_idx[1]] ==
               params.loops[1][params.reduction_loop_idx[1]] - 1 &&
           loop_counters[1][params.fx_loop_idx] ==
               params.loops[1][params.fx_loop_idx] - 1 &&
           loop_counters[0][params.fy_loop_idx[0]] ==
               params.loops[0][params.fy_loop_idx[0]] - 1 &&
           loop_counters[1][params.fy_loop_idx[1]] ==
               params.loops[1][params.fy_loop_idx[1]] - 1;
  }

  // Spill only nonlocal partial sums and route final buffered outputs
  // separately
  static bool writes_accumulation_buffer(
      const MatrixParams& params, const LocalAccumLayout& layout,
      const ac_int<LOOP_WIDTH, false> loop_counters[2][6],
      bool uses_local_accum_context) {
    const bool finishes_reduction =
        finishes_output_reduction(params, loop_counters);
    const bool finishes_local =
        layout.reduction_boundary < LOOP_LEVEL_COUNT * LOOP_SLOT_COUNT &&
        local_reduction_boundary(params, layout, loop_counters, true);
    return (!finishes_reduction &&
            (!uses_local_accum_context || finishes_local)) ||
           (finishes_reduction && DOUBLE_BUFFERED_ACCUM_BUFFER &&
            params.write_output_to_accum_buffer);
  }

  // Match MatrixProcessor's double-buffer bank-switch boundary
  static bool finishes_output_tile(
      const MatrixParams& params,
      const ac_int<LOOP_WIDTH, false> loop_counters[2][6]) {
    const bool outer_ox_inside_reduction =
        (params.loops[0][params.reduction_loop_idx[0]] > 1 &&
         params.reduction_loop_idx[0] < params.x_loop_idx[0]) ||
        (params.loops[0][params.fy_loop_idx[0]] > 1 &&
         params.fy_loop_idx[0] < params.x_loop_idx[0]);
    const bool outer_oy_inside_reduction =
        (params.loops[0][params.reduction_loop_idx[0]] > 1 &&
         params.reduction_loop_idx[0] < params.y_loop_idx[0]) ||
        (params.loops[0][params.fy_loop_idx[0]] > 1 &&
         params.fy_loop_idx[0] < params.y_loop_idx[0]);
    const bool finishes_x_context =
        !outer_ox_inside_reduction ||
        loop_counters[0][params.x_loop_idx[0]] ==
            params.loops[0][params.x_loop_idx[0]] - 1;
    const bool finishes_y_context =
        !outer_oy_inside_reduction ||
        loop_counters[0][params.y_loop_idx[0]] ==
            params.loops[0][params.y_loop_idx[0]] - 1;
    return finishes_output_reduction(params, loop_counters) &&
           finishes_x_context && finishes_y_context &&
           loop_counters[1][params.weight_loop_idx[1]] ==
               params.loops[1][params.weight_loop_idx[1]] - 1 &&
           loop_counters[1][params.x_loop_idx[1]] ==
               params.loops[1][params.x_loop_idx[1]] - 1 &&
           loop_counters[1][params.y_loop_idx[1]] ==
               params.loops[1][params.y_loop_idx[1]] - 1;
  }

  // Return whether this operation begins a new OUTPUT_LANES-lane bias vector
  static bool starts_bias_vector(
      const MatrixParams& params,
      const ac_int<LOOP_WIDTH, false> inner_loop_counters[6]) {
    // Bias is indexed by OC only, so every loop nested inside OC@L1 must
    // restart
    const auto inner_oc_position = params.weight_loop_idx[1];
    bool starts = true;
#pragma hls_unroll yes
    for (int slot = 0; slot < LOOP_SLOT_COUNT; slot++) {
      starts = starts &&
               (slot <= inner_oc_position || inner_loop_counters[slot] == 0);
    }
    return starts;
  }

  // Pop the one complete reduced MAC-result beat and unpack its OUTPUT_LANES
  // lanes
  Pack1D<Psum, OUTPUT_LANES> collect_complete_result() {
    const CBeat beat = result_channel.Pop();
    Pack1D<Psum, OUTPUT_LANES> result;
#pragma hls_unroll yes
    for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
         output_tile_index++) {
#pragma hls_unroll yes
      for (int tile_output_index = 0;
           tile_output_index < Array::TILE_OUTPUT_LANES; tile_output_index++) {
        const int output_index =
            output_tile_index * Array::TILE_OUTPUT_LANES + tile_output_index;
        ac_int<Psum::width, SIGNED> value;
        value.set_slc(0, beat[output_tile_index][tile_output_index]);
        result[output_index] = Psum(value);
      }
    }
    return result;
  }

  // Retire raw CIM results before accumulation
  void collect_results() {
    collect_results_params_deq.ResetRead();
    result_channel.ResetRead();
    result_to_accum_channel.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = collect_results_params_deq.Pop();
      const ac_int<32, false> total_ops = total_operations(params);
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (ac_int<32, false> step = 0; step < total_ops; step++) {
        result_to_accum_channel.Push(collect_complete_result());
      }
    }
  }

  // Issue SRAM reads directly and rely on local contexts to separate reductions
  // The schedule and context count must cover the SRAM feedback latency
  void issue_accumulation_reads() {
    process_accumulation_params_deq.ResetRead();
    result_to_accum_channel.ResetRead();
    bias_channel.Reset();
    accumulation_metadata_enq.ResetWrite();
#pragma hls_unroll yes
    for (int bank = 0; bank < ACCUM_BUFFER_BANKS; bank++) {
      accumulation_buffer_read_address[bank].Reset();
    }
    bool accumulation_buffer_bank = false;
#ifndef __SYNTHESIS__
    for (int bank = 0; bank < ACCUM_BUFFER_BANKS; bank++) {
      simulation_partial_sums[bank].assign(
          buffer_size, Pack1D<Buffer, OUTPUT_LANES>::zero());
    }
    simulation_accumulation_reads.clear();
#endif
    wait();

    while (true) {
      const MatrixParams params = process_accumulation_params_deq.Pop();
      const LocalAccumLayout layout = local_accum_layout(params);
      ac_int<LOOP_WIDTH, false> loop_counters[2][6];
      reset_loop_counters(loop_counters);

      Pack1D<Buffer, OUTPUT_LANES> bias = Pack1D<Buffer, OUTPUT_LANES>::zero();
      const ac_int<32, false> total_ops = total_operations(params);
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (ac_int<32, false> step = 0; step < total_ops; step++) {
        AccumulationMetadata metadata;
        metadata.result = result_to_accum_channel.Pop();
        metadata.starts_reduction =
            starts_output_reduction(params, loop_counters);
        metadata.finishes_reduction =
            finishes_output_reduction(params, loop_counters);
        metadata.initial_value = Pack1D<Buffer, OUTPUT_LANES>::zero();
        const ac_int<16, false> address =
            output_context_index(params, loop_counters);
        const ac_int<16, false> context =
            local_accum_context_index(params, layout, loop_counters);
        metadata.uses_local_accum_context = context < LOCAL_ACCUM_CONTEXTS;
        const bool starts_local =
            layout.reduction_boundary < LOOP_LEVEL_COUNT * LOOP_SLOT_COUNT &&
            local_reduction_boundary(params, layout, loop_counters, false);
        metadata.reads_accumulation_buffer =
            !metadata.starts_reduction &&
            (!metadata.uses_local_accum_context || starts_local);
        metadata.local_accum_context = metadata.uses_local_accum_context
                                           ? LocalAccumContext(context)
                                           : LocalAccumContext(0);
        metadata.buffer_bank = accumulation_buffer_bank;
        metadata.begins_matrix_command = step == 0;

        if (metadata.starts_reduction) {
          if (params.has_bias) {
            if (starts_bias_vector(params, loop_counters[1])) {
              bias = bias_channel.Pop();
            }
            metadata.initial_value = bias;
          }
        } else if (metadata.reads_accumulation_buffer) {
          accumulation_buffer_read_address[accumulation_buffer_bank].Push(
              address);
        }

#ifndef __SYNTHESIS__
        {
          if (address >= buffer_size) {
            throw std::runtime_error(
                "CIM accumulation address exceeds buffer capacity");
          }
          auto& expected = simulation_partial_sums[accumulation_buffer_bank]
                                                  [address.to_uint()];
          if (metadata.starts_reduction) {
            expected = metadata.initial_value;
          }
          if (metadata.reads_accumulation_buffer) {
            simulation_accumulation_reads.push_back(
                {address.to_uint(), unsigned(accumulation_buffer_bank),
                 expected});
          }
          for (int output_index = 0; output_index < OUTPUT_LANES;
               output_index++) {
            expected[output_index] +=
                static_cast<Buffer>(metadata.result[output_index]);
          }
        }
#endif

        accumulation_metadata_enq.Push(metadata);

#if DOUBLE_BUFFERED_ACCUM_BUFFER
        if (params.write_output_to_accum_buffer &&
            finishes_output_tile(params, loop_counters)) {
          accumulation_buffer_bank = !accumulation_buffer_bank;
        }
#endif

        advance_loop_counters(loop_counters, params);
      }
    }
  }

  // Accumulate once while local accum contexts retain temporal partial sums
  void complete_accumulation() {
    accumulation_metadata_deq.ResetRead();
    accum_to_wb_enq.ResetWrite();
#pragma hls_unroll yes
    for (int bank = 0; bank < ACCUM_BUFFER_BANKS; bank++) {
      accumulation_buffer_read_data[bank].Reset();
    }

    wait();

    Pack1D<Buffer, OUTPUT_LANES>
        local_accum_context_values[LOCAL_ACCUM_CONTEXTS];
#pragma hls_unroll yes
    for (int context = 0; context < LOCAL_ACCUM_CONTEXTS; context++) {
      local_accum_context_values[context] =
          Pack1D<Buffer, OUTPUT_LANES>::zero();
    }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      const AccumulationMetadata metadata = accumulation_metadata_deq.Pop();
      if (metadata.begins_matrix_command) {
#pragma hls_unroll yes
        for (int context = 0; context < LOCAL_ACCUM_CONTEXTS; context++) {
          local_accum_context_values[context] =
              Pack1D<Buffer, OUTPUT_LANES>::zero();
        }
      }

      Pack1D<Buffer, OUTPUT_LANES> previous = metadata.initial_value;

      if (!metadata.starts_reduction && metadata.uses_local_accum_context &&
          !metadata.reads_accumulation_buffer) {
#pragma hls_unroll yes
        for (int context = 0; context < LOCAL_ACCUM_CONTEXTS; context++) {
          if (metadata.local_accum_context == context) {
            previous = local_accum_context_values[context];
          }
        }
      } else if (metadata.reads_accumulation_buffer) {
#pragma hls_unroll yes
        for (int bank = 0; bank < ACCUM_BUFFER_BANKS; bank++) {
          if (metadata.buffer_bank == bank) {
            previous = accumulation_buffer_read_data[bank].Pop();
#ifndef __SYNTHESIS__
            const auto check = simulation_accumulation_reads.front();
            simulation_accumulation_reads.pop_front();
            if (check.bank != unsigned(bank)) {
              throw std::runtime_error(
                  "CIM RAW diagnostic bookkeeping error: unexpected read bank");
            }
            // Check before adding the current contribution to localize a stale
            // read
            for (int lane = 0; lane < OUTPUT_LANES; lane++) {
              if (!(previous[lane] == check.previous[lane])) {
                std::ostringstream message;
                message
                    << "CIM RAW diagnostic: SRAM feedback partial-sum mismatch"
                    << " at bank " << bank << " address " << check.address
                    << " lane " << lane << ", expected prior partial sum "
                    << check.previous[lane] << ", read " << previous[lane]
                    << "; possible read-after-write hazard: the read may have"
                    << " preceded visibility of its required write"
                    << "; detected before adding the current contribution,"
                    << " independently of final-output validation"
                    << "; verify local-context spacing and SRAM feedback "
                       "timing";
                throw std::runtime_error(message.str());
              }
            }
#endif
          }
        }
      }

      // This is the sole temporal accumulation arithmetic datapath
#pragma hls_unroll yes
      for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) {
        previous[output_index] +=
            static_cast<Buffer>(metadata.result[output_index]);
      }

      if (metadata.uses_local_accum_context) {
#pragma hls_unroll yes
        for (int context = 0; context < LOCAL_ACCUM_CONTEXTS; context++) {
          if (metadata.local_accum_context == context) {
            if (metadata.finishes_reduction) {
              local_accum_context_values[context] =
                  Pack1D<Buffer, OUTPUT_LANES>::zero();
            } else {
              // Feed the unfinished row back through its selected register
              local_accum_context_values[context] = previous;
            }
          }
        }
      }

      // Capture the value independently before later rows reuse its register
      accum_to_wb_enq.Push(previous);
    }
  }

  // Rederive retirement from synchronized loop state and write completed values
  void write_back() {
    write_back_params_deq.ResetRead();
    accum_to_wb_deq.ResetRead();
    accum_output_enq.ResetWrite();
#pragma hls_unroll yes
    for (int bank = 0; bank < ACCUM_BUFFER_BANKS; bank++) {
      accumulation_buffer_write_request[bank].Reset();
#if DOUBLE_BUFFERED_ACCUM_BUFFER
      accumulation_buffer_done[bank].Reset();
#endif
    }

#if ENABLE_PERF_COUNTERS
    perf_completion_toggle.write(false);
#endif
    bool accumulation_buffer_bank = false;
    wait();

    while (true) {
      const MatrixParams params = write_back_params_deq.Pop();
      const LocalAccumLayout layout = local_accum_layout(params);
      ac_int<LOOP_WIDTH, false> loop_counters[2][6];
      reset_loop_counters(loop_counters);

      const ac_int<32, false> total_ops = total_operations(params);
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (ac_int<32, false> step = 0; step < total_ops; step++) {
        const Pack1D<Buffer, OUTPUT_LANES> accumulated = accum_to_wb_deq.Pop();
        const ac_int<16, false> address =
            output_context_index(params, loop_counters);
        const bool uses_local_accum_context =
            local_accum_context_index(params, layout, loop_counters) <
            LOCAL_ACCUM_CONTEXTS;
        const bool writes_buffer = writes_accumulation_buffer(
            params, layout, loop_counters, uses_local_accum_context);
        const bool emits_output =
            finishes_output_reduction(params, loop_counters) && !writes_buffer;

        if (emits_output) {
          accum_output_enq.Push(accumulated);
        } else if (writes_buffer) {
          AccumulationWriteRequest request;
          request.address = address;
          request.data = accumulated;
          request.last = false;
          accumulation_buffer_write_request[accumulation_buffer_bank].Push(
              request);
        }

#if DOUBLE_BUFFERED_ACCUM_BUFFER
        if (params.write_output_to_accum_buffer &&
            finishes_output_tile(params, loop_counters)) {
          accumulation_buffer_done[accumulation_buffer_bank].SyncPush();
          accumulation_buffer_bank = !accumulation_buffer_bank;
        }
#endif

        advance_loop_counters(loop_counters, params);
      }
#if ENABLE_PERF_COUNTERS
      perf_completion_toggle.write(!perf_completion_toggle.read());
#endif
    }
  }

#if ENABLE_PERF_COUNTERS
  // Keep totals independent of command overlap and reporting speed.
  void monitor_performance() {
    using namespace MatrixPerformance;
    ProcessorCounters counters;
    counters.reset();
    Counter weight_beats = 0;
#pragma hls_unroll yes
    for (int i = 0; i < PROCESSOR_COUNTER_COUNT; i++) perf_counters[i].write(0);
    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
    while (true) {
      counters.observe(params_in.vld.read() && params_in.rdy.read(),
                       perf_completion_toggle.read(),
                       mac_request_channel.vld.read(),
                       mac_request_channel.rdy.read(),
                       result_channel.vld.read(), result_channel.rdy.read());
#pragma hls_unroll yes
      for (int i = 0; i < COMMON_PROCESSOR_COUNTER_COUNT; i++)
        perf_counters[i].write(counters.values[i]);
      perf_counters[storage_index(MAC_WAIT_WEIGHT_SET_LOAD_CYCLES)].write(
          perf_mac_wait_weight_set_load_cycles.read());
      if (weight_channel.vld.read() && weight_channel.rdy.read())
        ++weight_beats;
      perf_counters[storage_index(CIM_WEIGHT_LOAD_CYCLES)].write(weight_beats);
      wait();
    }
  }
#endif
};
