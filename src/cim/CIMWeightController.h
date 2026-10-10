#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include <algorithm>
#include <tuple>

#include "AccelTypes.h"
#include "CIMConfig.h"
#include "CIMTypes.h"
#include "TypeToBits.h"
#include "Utils.h"

#ifndef __SYNTHESIS__
#include <stdexcept>
#include <string>
#endif

template <typename WeightTypeTuple, typename Bias, int INPUT_LANES,
          int OUTPUT_LANES, int MEMORY_PORT_WIDTH, int WEIGHT_ROW_WIDTH,
          int WEIGHT_BEAT_WIDTH = WEIGHT_ROW_WIDTH,
          int WEIGHT_SETS = CIM_WEIGHT_SETS>
struct CIMWeightController;

// Fetch, decode, and stream weights into the CIM resident-set ring.
//
// Each set arrives in input-axis order, with output-axis beats inside each
// input row. Descriptors declare an ordered sequence and its replay count;
// CIMProcessor owns physical set allocation and final-use release.
template <typename... WeightTypes, typename Bias, int INPUT_LANES,
          int OUTPUT_LANES, int MEMORY_PORT_WIDTH, int WEIGHT_ROW_WIDTH,
          int WEIGHT_BEAT_WIDTH, int WEIGHT_SETS>
struct CIMWeightController<std::tuple<WeightTypes...>, Bias, INPUT_LANES,
                           OUTPUT_LANES, MEMORY_PORT_WIDTH, WEIGHT_ROW_WIDTH,
                           WEIGHT_BEAT_WIDTH, WEIGHT_SETS> : public sc_module {
  static constexpr int LOOP_WIDTH = MatrixParams::LOOP_WIDTH;
  static constexpr int LOOP_LEVEL_COUNT = 2;
  static constexpr int LOOP_SLOT_COUNT = 6;
  // Derive one scalar value's width from an output-wide resident weight row
  static constexpr int DATA_WIDTH = WEIGHT_ROW_WIDTH / OUTPUT_LANES;
  // A transposed source row holds one scalar per input lane and becomes
  // one resident column
  static constexpr int SOURCE_ROW_WIDTH = INPUT_LANES * DATA_WIDTH;
  static constexpr int WEIGHT_BEATS_PER_ROW =
      WEIGHT_ROW_WIDTH / WEIGHT_BEAT_WIDTH;
  static_assert(INPUT_LANES > 0 && OUTPUT_LANES > 0,
                "CIM weight geometry must be positive");
  static_assert(WEIGHT_ROW_WIDTH > 0 && WEIGHT_BEAT_WIDTH > 0 &&
                    WEIGHT_ROW_WIDTH % OUTPUT_LANES == 0,
                "One logical weight row must contain whole output lanes");
  static_assert(MEMORY_PORT_WIDTH > 0 && MEMORY_PORT_WIDTH % 8 == 0,
                "CIM weight memory must carry whole bytes");
  static_assert(WEIGHT_SETS > 0,
                "CIMWeightController requires a CIM resident weight set");
  static_assert(WEIGHT_SETS <= 0xFFFF,
                "CIM resident set count exceeds schedule metadata");
  static_assert(WEIGHT_ROW_WIDTH % WEIGHT_BEAT_WIDTH == 0,
                "One logical weight row must split into whole channel beats");
  // packed_bits holds an output-wide weight row or input-wide transposed source
  // row
  static constexpr int MAX_FETCH_WIDTH =
      std::max({dtype_fetch_config<WeightTypes, INPUT_LANES,
                                   MEMORY_PORT_WIDTH>::max_fetch_width...,
                dtype_fetch_config<WeightTypes, OUTPUT_LANES,
                                   MEMORY_PORT_WIDTH>::max_fetch_width...});

  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::Out<MemoryRequest> CCS_INIT_S1(weight_req);
  Connections::In<ac_int<MEMORY_PORT_WIDTH, false>> CCS_INIT_S1(weight_resp);

  // Stream one physical weight-port beat using the weight_channel protocol
  Connections::Out<ac_int<WEIGHT_BEAT_WIDTH, false>> CCS_INIT_S1(
      weight_channel);
  Connections::Out<cim::WeightDescriptor> CCS_INIT_S1(
      weight_descriptor_channel);

  Connections::Out<MemoryRequest> CCS_INIT_S1(bias_req);
  Connections::In<ac_int<MEMORY_PORT_WIDTH, false>> CCS_INIT_S1(bias_resp);
  Connections::Out<Pack1D<Bias, OUTPUT_LANES>> CCS_INIT_S1(bias_data);

  Connections::In<MatrixParams> CCS_INIT_S1(params_in);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(writer_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(reader_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(weight_packer_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(transposer_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(bias_fetcher_params);
  Connections::Combinational<MatrixParams> CCS_INIT_S1(bias_feeder_params);

  // Independent end markers let the packer and transposer retire one fetch
  // stream without coupling their backpressure
  sc_fifo<bool> packer_stream_end;
  sc_fifo<bool> transposer_stream_end;

  // Carry one assembled memory fetch into the datatype unpacker
  Connections::Combinational<ac_int<MAX_FETCH_WIDTH, false>> packed_bits;
  // Carry one complete logical weight row after optional transposition
  Connections::Combinational<ac_int<WEIGHT_ROW_WIDTH, false>> transpose_out;
  // Keep each packed-row index in order while fetch and unpack stages stall.
  Connections::Fifo<ac_int<4, false>, 3> CCS_INIT_S1(packing_indices_fifo);
  Connections::Combinational<ac_int<4, false>> CCS_INIT_S1(packing_indices_enq);
  Connections::Combinational<ac_int<4, false>> CCS_INIT_S1(packing_indices_deq);
  // False announces one resident-set payload; true terminates the matrix job
  sc_fifo<bool> resident_set_stream_end;

  SC_CTOR(CIMWeightController) {
    packing_indices_fifo.clk(clk);
    packing_indices_fifo.rst(rstn);
    packing_indices_fifo.enq(packing_indices_enq);
    packing_indices_fifo.deq(packing_indices_deq);

    SC_THREAD(read_params);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(reader);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(writer);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(transposer);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(weight_packer);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(bias_fetcher);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(bias_feeder);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  // Zero-fill and slice logical weight rows into physical weight-port beats.
  void writer() {
    writer_params.ResetRead();
    transpose_out.ResetRead();
    weight_channel.Reset();

    wait();

    while (true) {
      const MatrixParams params = writer_params.Pop();
      // The innermost IC extent counts source rows containing real data.
      const ac_int<LOOP_WIDTH, false> source_row_count =
          params.weight_addr_loops[1][params.weight_addr_reduction_loop_idx[2]];
      const ac_int<6, false> dtype_width =
          get_type_width<WeightTypes...>(params.weight_dtype);

      // Normal fetches may pack multiple output groups, so recover one row's
      // valid output lanes. The transposer emits complete resident rows.
      ac_int<LOOP_WIDTH, false> valid_columns_per_row = OUTPUT_LANES;
      if (!params.weight_transpose) {
        // Convert burst bytes into the total fetched scalar count
        valid_columns_per_row = params.weight_burst_size * 8 / dtype_width;
        // Divide by the number of output groups packed into the fetch.
        valid_columns_per_row >>= params.weight_pack_factor_lg2;
        if (valid_columns_per_row > OUTPUT_LANES)
          valid_columns_per_row = OUTPUT_LANES;
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (!resident_set_stream_end.read()) {
        for (int row = 0; row < INPUT_LANES; row++) {
          // Every resident set receives a complete input-by-output weight
          // matrix
          ac_int<WEIGHT_ROW_WIDTH, false> data = 0;
          if (params.weight_transpose || row < source_row_count) {
            const ac_int<WEIGHT_ROW_WIDTH, false> fetched = transpose_out.Pop();
#pragma hls_unroll yes
            for (int col = 0; col < OUTPUT_LANES; col++) {
              if (col < valid_columns_per_row) {
                data.set_slc(col * DATA_WIDTH, fetched.template slc<DATA_WIDTH>(
                                                   col * DATA_WIDTH));
              }
            }
          }
          for (int beat = 0; beat < WEIGHT_BEATS_PER_ROW; beat++) {
            weight_channel.Push(
                data.template slc<WEIGHT_BEAT_WIDTH>(beat * WEIGHT_BEAT_WIDTH));
          }
        }
      }
    }
  }

  // Describe source B[FY][FX][IC][OC] extents and flattened memory strides
  struct WeightTensorLayout {
    ac_int<LOOP_WIDTH, false> outer_oc_bound;
    ac_int<LOOP_WIDTH, false> outer_ic_bound;
    ac_int<LOOP_WIDTH, false> inner_ic_bound;
    ac_int<LOOP_WIDTH, false> rows_per_inner_ic;
    ac_int<LOOP_WIDTH, false> fx_bound;
    ac_int<LOOP_WIDTH, false> outer_fy_bound;
    ac_int<LOOP_WIDTH, false> packed_inner_oc_bound;
    ac_int<16, false> oc_tile_stride;
    ac_int<24, false> ic_stride;
    ac_int<24, false> fx_stride;
    ac_int<24, false> fy_stride;
  };

  // Name one B-tensor coordinate decoded from the physical reader loop nest
  struct WeightTensorCoordinate {
    ac_int<LOOP_WIDTH, false> outer_oc;
    ac_int<LOOP_WIDTH, false> outer_ic;
    ac_int<LOOP_WIDTH, false> inner_ic;
    ac_int<LOOP_WIDTH, false> fx;
    ac_int<LOOP_WIDTH, false> inner_fy;
    ac_int<LOOP_WIDTH, false> outer_fy;
    ac_int<LOOP_WIDTH, false> packed_inner_oc;
    ac_int<4, false> packing_index;
  };

  // Decode legacy weight-address metadata into the source B-tensor layout
  static WeightTensorLayout weight_tensor_layout(const MatrixParams& params) {
    WeightTensorLayout layout;
    layout.outer_oc_bound =
        params.weight_addr_loops[0][params.weight_addr_weight_loop_idx[0]];
    layout.outer_ic_bound =
        params.weight_addr_loops[0][params.weight_addr_reduction_loop_idx[0]];
    layout.inner_ic_bound =
        params.weight_addr_loops[1][params.weight_addr_reduction_loop_idx[1]];
    layout.rows_per_inner_ic =
        params.weight_addr_loops[1][params.weight_addr_reduction_loop_idx[2]];
    layout.fx_bound = params.weight_addr_loops[1][params.weight_addr_fx_idx];
    layout.outer_fy_bound =
        params.weight_addr_loops[0][params.weight_addr_fy_idx[0]];
    layout.packed_inner_oc_bound =
        params.weight_addr_loops[1][params.weight_addr_weight_loop_idx[1]] >>
        params.weight_pack_factor_lg2;

    layout.oc_tile_stride = OUTPUT_LANES << params.weight_pack_factor_lg2;
    layout.ic_stride = layout.outer_oc_bound * layout.packed_inner_oc_bound *
                       layout.oc_tile_stride;
    layout.fx_stride = layout.outer_ic_bound * layout.inner_ic_bound *
                       layout.rows_per_inner_ic * layout.ic_stride;
    layout.fy_stride = layout.fx_bound * layout.fx_stride;
    return layout;
  }

  // Read one source B-tensor coordinate from the physical loop counters
  static WeightTensorCoordinate weight_tensor_coordinate(
      const MatrixParams& params,
      const ac_int<LOOP_WIDTH, false> loop_counters[2][6]) {
    WeightTensorCoordinate coordinate;
    coordinate.outer_oc = loop_counters[0][params.weight_loop_idx[0]];
    coordinate.outer_ic = loop_counters[0][params.reduction_loop_idx[0]];
    coordinate.inner_ic = loop_counters[1][params.reduction_loop_idx[1]];
    coordinate.fx = loop_counters[1][params.fx_loop_idx];
    coordinate.inner_fy = loop_counters[1][params.fy_loop_idx[1]];
    coordinate.outer_fy = loop_counters[0][params.fy_loop_idx[0]];
    const ac_int<LOOP_WIDTH, false> inner_oc =
        loop_counters[1][params.weight_loop_idx[1]];
    coordinate.packed_inner_oc = inner_oc >> params.weight_pack_factor_lg2;
    coordinate.packing_index = inner_oc - (coordinate.packed_inner_oc
                                           << params.weight_pack_factor_lg2);
    return coordinate;
  }

  // Flatten one B-tensor coordinate into its packed source-memory address
  static ac_int<32, false> weight_source_address(
      const MatrixParams& params, const WeightTensorLayout& layout,
      const WeightTensorCoordinate& coordinate, int row) {
    const ac_int<16, false> oc_offset =
        (coordinate.outer_oc * layout.packed_inner_oc_bound +
         coordinate.packed_inner_oc) *
        layout.oc_tile_stride;
    if (params.weight_transpose) {
      // Rows select output and columns select reduction in transposed storage
      return ((oc_offset + row) * layout.outer_ic_bound *
                  layout.inner_ic_bound +
              coordinate.outer_ic * layout.inner_ic_bound +
              coordinate.inner_ic) *
             INPUT_LANES;
    }

    const ac_int<16, false> ic_offset =
        (coordinate.outer_ic * layout.inner_ic_bound + coordinate.inner_ic) *
            layout.rows_per_inner_ic +
        row;
    const ac_int<16, false> fy =
        coordinate.inner_fy * layout.outer_fy_bound + coordinate.outer_fy;
    return fy * layout.fy_stride + coordinate.fx * layout.fx_stride +
           ic_offset * layout.ic_stride + oc_offset;
  }

  // Return whether the physical L1 traversal begins its weight sequence
  static bool starts_l1_sequence(
      const ac_int<LOOP_WIDTH, false> inner_loop_counters[6]) {
    bool starts = true;
#pragma hls_unroll yes
    for (int slot = 0; slot < LOOP_SLOT_COUNT; slot++) {
      starts = starts && inner_loop_counters[slot] == 0;
    }
    return starts;
  }

  // Count sets in the L1 weight sequence, capped one past physical capacity
  static ac_int<32, false> l1_set_count(
      const ac_int<LOOP_WIDTH, false> inner_loop_bounds[6]) {
    ac_int<32, false> set_count = 1;
#pragma hls_unroll yes
    for (int slot = 0; slot < LOOP_SLOT_COUNT; slot++) {
      const ac_int<32, false> next_set_count =
          set_count * inner_loop_bounds[slot];
      set_count = next_set_count > WEIGHT_SETS
                      ? ac_int<32, false>(WEIGHT_SETS + 1)
                      : next_set_count;
    }
    return set_count;
  }

  // Identify an L1 spatial loop that repeats the complete inner weight sequence
  static bool l1_spatial_replays_sequence(const MatrixParams& params,
                                          int spatial_loop_idx) {
    const auto l1_oc = params.loops[1][params.weight_loop_idx[1]];
    const auto l1_ic = params.loops[1][params.reduction_loop_idx[1]];
    const auto l1_fy = params.loops[1][params.fy_loop_idx[1]];
    const auto l1_fx = params.loops[1][params.fx_loop_idx];
    return params.loops[1][spatial_loop_idx] > 1 &&
           (l1_oc == 1 || params.weight_loop_idx[1] > spatial_loop_idx) &&
           (l1_ic == 1 || params.reduction_loop_idx[1] > spatial_loop_idx) &&
           (l1_fy == 1 || params.fy_loop_idx[1] > spatial_loop_idx) &&
           (l1_fx == 1 || params.fx_loop_idx > spatial_loop_idx);
  }

  // Fetch each resident sequence once when it fits, and declare its replays.
  // Innermost spatial loops retain one set; enclosing spatial loops replay
  // the complete sequence. Oversized sequences stream and refill single sets.
  void reader() {
    reader_params.ResetRead();

    weight_req.Reset();
    packing_indices_enq.ResetWrite();
    weight_descriptor_channel.Reset();

    wait();

    while (true) {
      const MatrixParams params = reader_params.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][6];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][6];

#pragma hls_unroll yes
      for (int level = 0; level < LOOP_LEVEL_COUNT; level++) {
#pragma hls_unroll yes
        for (int slot = 0; slot < LOOP_SLOT_COUNT; slot++) {
          loop_bounds[level][slot] = params.loops[level][slot];
        }
      }

      const bool l1_ox_reuses_weights = matrix_loop_reuses_weights(
          params, MatrixLoopLevel::L1, MatrixLoopParam::OX);
      const bool l1_oy_reuses_weights = matrix_loop_reuses_weights(
          params, MatrixLoopLevel::L1, MatrixLoopParam::OY);
      if (l1_ox_reuses_weights) {
        loop_bounds[1][params.x_loop_idx[1]] = 1;
      }
      if (l1_oy_reuses_weights) {
        loop_bounds[1][params.y_loop_idx[1]] = 1;
      }

      const bool l1_ox_replays_sequence =
          !l1_ox_reuses_weights &&
          l1_spatial_replays_sequence(params, params.x_loop_idx[1]);
      const bool l1_oy_replays_sequence =
          !l1_oy_reuses_weights &&
          l1_spatial_replays_sequence(params, params.y_loop_idx[1]);
      if (l1_ox_replays_sequence) {
        loop_bounds[1][params.x_loop_idx[1]] = 1;
      }
      if (l1_oy_replays_sequence) {
        loop_bounds[1][params.y_loop_idx[1]] = 1;
      }
      const bool l1_group_replay_requested =
          l1_ox_replays_sequence || l1_oy_replays_sequence;
      const bool l1_group_replay_fits =
          l1_set_count(loop_bounds[1]) <= WEIGHT_SETS;
      const bool l1_group_replay_enabled =
          l1_group_replay_requested && l1_group_replay_fits;
      if (l1_group_replay_requested && !l1_group_replay_fits) {
        if (l1_ox_replays_sequence) {
          loop_bounds[1][params.x_loop_idx[1]] =
              params.loops[1][params.x_loop_idx[1]];
        }
        if (l1_oy_replays_sequence) {
          loop_bounds[1][params.y_loop_idx[1]] =
              params.loops[1][params.y_loop_idx[1]];
        }
      }

      const bool l2_ox_reuses_weights = matrix_loop_reuses_weights(
          params, MatrixLoopLevel::L2, MatrixLoopParam::OX);
      const bool l2_oy_reuses_weights = matrix_loop_reuses_weights(
          params, MatrixLoopLevel::L2, MatrixLoopParam::OY);
      const bool omit_outer_ox_from_weight_reader = l2_ox_reuses_weights;
      const bool omit_outer_oy_from_weight_reader = l2_oy_reuses_weights;
      ac_int<16, false> compute_sequence_replay_count = 1;
      if (l1_group_replay_enabled && l1_ox_replays_sequence) {
        compute_sequence_replay_count *= params.loops[1][params.x_loop_idx[1]];
      }
      if (l1_group_replay_enabled && l1_oy_replays_sequence) {
        compute_sequence_replay_count *= params.loops[1][params.y_loop_idx[1]];
      }
      if (omit_outer_ox_from_weight_reader) {
        loop_bounds[0][params.x_loop_idx[0]] = 1;
        compute_sequence_replay_count *= params.loops[0][params.x_loop_idx[0]];
      }
      if (omit_outer_oy_from_weight_reader) {
        loop_bounds[0][params.y_loop_idx[0]] = 1;
        compute_sequence_replay_count *= params.loops[0][params.y_loop_idx[0]];
      }

      // A fitting sequence is fetched once and retained for every descriptor
      // replay. An oversized sequence streams singleton descriptors and must be
      // fetched again for every replay
      const ac_int<32, false> l1_sequence_set_count =
          l1_set_count(loop_bounds[1]);
      const bool l1_sequence_fits = l1_sequence_set_count <= WEIGHT_SETS;
      const ac_int<16, false> fetch_sequence_replay_count =
          l1_sequence_fits ? ac_int<16, false>(1)
                           : compute_sequence_replay_count;
      const WeightTensorLayout weight_layout = weight_tensor_layout(params);

      // Preserve the established twelve-loop HLS nest and II; the sixth L2
      // slot is fixed-unit FX and therefore has no physical loop here
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0;; loop_counters[0][2]++) {
            for (loop_counters[0][3] = 0;; loop_counters[0][3]++) {
              for (loop_counters[0][4] = 0;; loop_counters[0][4]++) {
                for (ac_int<16, false> fetch_replay_index = 0;;
                     fetch_replay_index++) {
                  for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
                    for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
                      for (loop_counters[1][2] = 0;; loop_counters[1][2]++) {
                        for (loop_counters[1][3] = 0;; loop_counters[1][3]++) {
                          for (loop_counters[1][4] = 0;;
                               loop_counters[1][4]++) {
                            for (loop_counters[1][5] = 0;;
                                 loop_counters[1][5]++) {
                              const bool begins_l1_sequence =
                                  starts_l1_sequence(loop_counters[1]);
                              if (!l1_sequence_fits || begins_l1_sequence) {
                                cim::WeightDescriptor descriptor;
                                if (l1_sequence_fits) {
                                  descriptor.set_count = l1_sequence_set_count;
                                  descriptor.replay_count =
                                      compute_sequence_replay_count;
                                } else {
                                  descriptor.set_count = 1;
                                  descriptor.replay_count = 1;
                                }
                                weight_descriptor_channel.Push(descriptor);
                              }

                              const WeightTensorCoordinate weight_coordinate =
                                  weight_tensor_coordinate(params,
                                                           loop_counters);

                              resident_set_stream_end.write(false);
                              // Fetch one source row per resident weight row
                              // Transposition instead fetches one per column
                              for (int row = 0;
                                   row < (INPUT_LANES > OUTPUT_LANES
                                              ? INPUT_LANES
                                              : OUTPUT_LANES);
                                   row++) {
                                const bool active_row = params.weight_transpose
                                                            ? row < OUTPUT_LANES
                                                            : row < INPUT_LANES;
                                if (active_row &&
                                    row < weight_layout.rows_per_inner_ic) {
                                  const ac_int<32, false> address =
                                      weight_source_address(
                                          params, weight_layout,
                                          weight_coordinate, row);
                                  send_packed_request<WeightTypes...>(
                                      params.weight_dtype, params.weight_offset,
                                      address, params.weight_burst_size,
                                      weight_req);
                                  packing_indices_enq.Push(
                                      weight_coordinate.packing_index);
                                  packer_stream_end.write(false);
                                  transposer_stream_end.write(false);
                                }
                              }

                              if (loop_counters[1][5] == loop_bounds[1][5] - 1)
                                break;
                            }
                            if (loop_counters[1][4] == loop_bounds[1][4] - 1)
                              break;
                          }
                          if (loop_counters[1][3] == loop_bounds[1][3] - 1)
                            break;
                        }
                        if (loop_counters[1][2] == loop_bounds[1][2] - 1) break;
                      }
                      if (loop_counters[1][1] == loop_bounds[1][1] - 1) break;
                    }
                    if (loop_counters[1][0] == loop_bounds[1][0] - 1) break;
                  }
                  if (fetch_replay_index == fetch_sequence_replay_count - 1)
                    break;
                }
                if (loop_counters[0][4] == loop_bounds[0][4] - 1) break;
              }
              if (loop_counters[0][3] == loop_bounds[0][3] - 1) break;
            }
            if (loop_counters[0][2] == loop_bounds[0][2] - 1) break;
          }
          if (loop_counters[0][1] == loop_bounds[0][1] - 1) break;
        }
        if (loop_counters[0][0] == loop_bounds[0][0] - 1) break;
      }
      resident_set_stream_end.write(true);
      packer_stream_end.write(true);
      transposer_stream_end.write(true);
    }
  }

  // Assemble one logical source fetch from its memory-response beats
  void weight_packer() {
    weight_packer_params.ResetRead();
    weight_resp.Reset();
    packed_bits.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = weight_packer_params.Pop();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      while (!packer_stream_end.read()) {
        ac_int<MAX_FETCH_WIDTH, false> bits;

        for (ac_int<4, false> i = 0;; i++) {
          bits.set_slc(i * MEMORY_PORT_WIDTH, weight_resp.Pop());
          if (i == params.weight_num_beats - 1) break;
        }

        packed_bits.Push(bits);
      }
    }
  }

  // Unpack weight rows or transpose [output][input] into [input][output].
  void transposer() {
    transposer_params.ResetRead();
    packed_bits.ResetRead();
    transpose_out.ResetWrite();
    packing_indices_deq.ResetRead();

    wait();

    while (true) {
      const MatrixParams params = transposer_params.Pop();

      // Use the existing weight transposer's size limit to bound the register
      // array in larger hardware configurations.
#ifndef __SYNTHESIS__
      if (params.weight_transpose && !(INPUT_LANES < 64 && OUTPUT_LANES < 64)) {
        // Reject schedules requiring a transpose buffer omitted from hardware.
        SC_REPORT_FATAL("CIMWeightController",
                        "weight transpose is unsupported at this array size");
      }
#endif

      if (params.weight_transpose && INPUT_LANES < 64 && OUTPUT_LANES < 64) {
        // Transposed layouts provide one source row per physical output lane.
#ifndef __SYNTHESIS__
        if (params.weight_addr_loops[1]
                                    [params.weight_addr_reduction_loop_idx[2]] <
            OUTPUT_LANES) {
          SC_REPORT_FATAL("CIMWeightController",
                          "transposed weight set has too few source rows");
        }
#endif
        // Each source row becomes one column of the resident weight set
        ac_int<DATA_WIDTH, false> transpose_buffer[INPUT_LANES][OUTPUT_LANES];

        // Keep the blocking gather and emit phases in one sequential set
        // transfer
        while (!transposer_stream_end.read()) {
          for (int source_col = 0; source_col < OUTPUT_LANES; source_col++) {
            if (source_col != 0) {
              const bool set_done = transposer_stream_end.read();
#ifndef __SYNTHESIS__
              if (set_done) {
                SC_REPORT_FATAL("CIMWeightController",
                                "incomplete CIM transposed weight set");
              }
#endif
            }

            const ac_int<MAX_FETCH_WIDTH, false> bits = packed_bits.Pop();
            const ac_int<4, false> packing_index = packing_indices_deq.Pop();
            // Hold one transposed source row before scattering it by row
            ac_int<SOURCE_ROW_WIDTH, false> source_values = 0;
            const bool handled =
                (unpack_bits<WeightTypes, INPUT_LANES, SOURCE_ROW_WIDTH,
                             MAX_FETCH_WIDTH, WeightTypes...>(
                     params.weight_dtype, bits, source_values, packing_index) ||
                 ...);

#ifndef __SYNTHESIS__
            if (!handled) {
              throw std::runtime_error("Unsupported dtype for matrix weight: " +
                                       std::to_string(params.weight_dtype));
            }
#endif

#pragma hls_unroll yes
            for (int row = 0; row < INPUT_LANES; row++) {
              transpose_buffer[row][source_col] =
                  source_values.template slc<DATA_WIDTH>(row * DATA_WIDTH);
            }
          }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
          for (int row = 0; row < INPUT_LANES; row++) {
            ac_int<WEIGHT_ROW_WIDTH, false> transposed;
#pragma hls_unroll yes
            for (int col = 0; col < OUTPUT_LANES; col++) {
              transposed.set_slc(col * DATA_WIDTH, transpose_buffer[row][col]);
            }
            transpose_out.Push(transposed);
          }
        }
      } else {
#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
        while (!transposer_stream_end.read()) {
          const ac_int<MAX_FETCH_WIDTH, false> bits = packed_bits.Pop();
          const ac_int<4, false> packing_index = packing_indices_deq.Pop();
          ac_int<WEIGHT_ROW_WIDTH, false> outputs = 0;
          const bool handled =
              (unpack_bits<WeightTypes, OUTPUT_LANES, WEIGHT_ROW_WIDTH,
                           MAX_FETCH_WIDTH, WeightTypes...>(
                   params.weight_dtype, bits, outputs, packing_index) ||
               ...);

#ifndef __SYNTHESIS__
          if (!handled) {
            throw std::runtime_error("Unsupported dtype for matrix weight: " +
                                     std::to_string(params.weight_dtype));
          }
#endif

          transpose_out.Push(outputs);
        }
      }
    }
  }

  // Restrict bias traversal to the points where the processor loads its cache
  static void set_bias_loop_bounds(
      const MatrixParams& params,
      ac_int<LOOP_WIDTH, false> loop_bounds[2][LOOP_SLOT_COUNT]) {
    loop_bounds[0][params.reduction_loop_idx[0]] = 0;
    loop_bounds[0][params.fy_loop_idx[0]] = 0;
    loop_bounds[1][params.fx_loop_idx] = 0;
    loop_bounds[1][params.fy_loop_idx[1]] = 0;
    loop_bounds[1][params.reduction_loop_idx[1]] = 0;

    // The processor retains bias while loops nested inside L1 OC advance
#pragma hls_unroll yes
    for (int slot = 0; slot < LOOP_SLOT_COUNT; slot++) {
      if (slot > params.weight_loop_idx[1]) {
        loop_bounds[1][slot] = 0;
      }
    }
  }

  void bias_fetcher() {
    bias_fetcher_params.ResetRead();
    bias_req.Reset();

    wait();

    while (true) {
      const MatrixParams params = bias_fetcher_params.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][6];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j] - 1;
        }
      }

      set_bias_loop_bounds(params, loop_bounds);

      ac_int<LOOP_WIDTH, false> inner_oc_bound =
          params.loops[1][params.weight_loop_idx[1]];

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0;; loop_counters[0][2]++) {
            for (loop_counters[0][3] = 0;; loop_counters[0][3]++) {
              for (loop_counters[0][4] = 0;; loop_counters[0][4]++) {
                for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
                  for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
                    for (loop_counters[1][2] = 0;; loop_counters[1][2]++) {
                      for (loop_counters[1][3] = 0;; loop_counters[1][3]++) {
                        for (loop_counters[1][4] = 0;; loop_counters[1][4]++) {
                          for (loop_counters[1][5] = 0;;
                               loop_counters[1][5]++) {
                            ac_int<LOOP_WIDTH, false> outer_oc =
                                loop_counters[0][params.weight_loop_idx[0]];
                            ac_int<LOOP_WIDTH, false> inner_oc =
                                loop_counters[1][params.weight_loop_idx[1]];

                            ac_int<16, false> address =
                                outer_oc * inner_oc_bound * OUTPUT_LANES +
                                inner_oc * OUTPUT_LANES;

                            MemoryRequest request = {
                                params.bias_offset + address * Bias::width / 8,
                                OUTPUT_LANES * Bias::width / 8};

                            bias_req.Push(request);

                            if (loop_counters[1][5] == loop_bounds[1][5]) break;
                          }
                          if (loop_counters[1][4] == loop_bounds[1][4]) break;
                        }
                        if (loop_counters[1][3] == loop_bounds[1][3]) break;
                      }
                      if (loop_counters[1][2] == loop_bounds[1][2]) break;
                    }
                    if (loop_counters[1][1] == loop_bounds[1][1]) break;
                  }
                  if (loop_counters[1][0] == loop_bounds[1][0]) break;
                }
                if (loop_counters[0][4] == loop_bounds[0][4]) break;
              }
              if (loop_counters[0][3] == loop_bounds[0][3]) break;
            }
            if (loop_counters[0][2] == loop_bounds[0][2]) break;
          }
          if (loop_counters[0][1] == loop_bounds[0][1]) break;
        }
        if (loop_counters[0][0] == loop_bounds[0][0]) break;
      }
    }
  }

  void bias_feeder() {
    bias_feeder_params.ResetRead();
    bias_resp.Reset();
    bias_data.Reset();

    wait();

    while (true) {
      const MatrixParams params = bias_feeder_params.Pop();

      ac_int<LOOP_WIDTH, false> loop_counters[2][6];
      ac_int<LOOP_WIDTH, false> loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
#pragma hls_unroll yes
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j] - 1;
        }
      }

      set_bias_loop_bounds(params, loop_bounds);

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0;; loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0;; loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0;; loop_counters[0][2]++) {
            for (loop_counters[0][3] = 0;; loop_counters[0][3]++) {
              for (loop_counters[0][4] = 0;; loop_counters[0][4]++) {
                for (loop_counters[1][0] = 0;; loop_counters[1][0]++) {
                  for (loop_counters[1][1] = 0;; loop_counters[1][1]++) {
                    for (loop_counters[1][2] = 0;; loop_counters[1][2]++) {
                      for (loop_counters[1][3] = 0;; loop_counters[1][3]++) {
                        for (loop_counters[1][4] = 0;; loop_counters[1][4]++) {
                          for (loop_counters[1][5] = 0;;
                               loop_counters[1][5]++) {
                            ac_int<Bias::width * OUTPUT_LANES, false> bits;

                            process_matrix_input<Bias, OUTPUT_LANES,
                                                 MEMORY_PORT_WIDTH,
                                                 Bias::width * OUTPUT_LANES>(
                                bias_resp, bits);

                            Pack1D<Bias, OUTPUT_LANES> biases =
                                BitsToType<Pack1D<Bias, OUTPUT_LANES>>(
                                    TypeToBits(bits));

                            bias_data.Push(biases);
                            if (loop_counters[1][5] == loop_bounds[1][5]) break;
                          }
                          if (loop_counters[1][4] == loop_bounds[1][4]) break;
                        }
                        if (loop_counters[1][3] == loop_bounds[1][3]) break;
                      }
                      if (loop_counters[1][2] == loop_bounds[1][2]) break;
                    }
                    if (loop_counters[1][1] == loop_bounds[1][1]) break;
                  }
                  if (loop_counters[1][0] == loop_bounds[1][0]) break;
                }
                if (loop_counters[0][4] == loop_bounds[0][4]) break;
              }
              if (loop_counters[0][3] == loop_bounds[0][3]) break;
            }
            if (loop_counters[0][2] == loop_bounds[0][2]) break;
          }
          if (loop_counters[0][1] == loop_bounds[0][1]) break;
        }
        if (loop_counters[0][0] == loop_bounds[0][0]) break;
      }
    }
  }

  void read_params() {
    params_in.Reset();
    writer_params.ResetWrite();
    reader_params.ResetWrite();
    transposer_params.ResetWrite();
    weight_packer_params.ResetWrite();
    bias_fetcher_params.ResetWrite();
    bias_feeder_params.ResetWrite();

    wait();

    while (true) {
      const MatrixParams params = params_in.Pop();

      writer_params.Push(params);
      reader_params.Push(params);
      transposer_params.Push(params);
      weight_packer_params.Push(params);

      if (params.has_bias) {
        bias_fetcher_params.Push(params);
        bias_feeder_params.Push(params);
      }
    }
  }
};
