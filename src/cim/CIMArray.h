// CIMArray is a CIM-tile grid organized by input and output axes

#pragma once

#include <ac_int.h>
#include <mc_connections.h>
#include <systemc.h>

#include <sstream>

#include "AccelTypes.h"
#include "PackUtils.h"
#include "CIMTile.h"
#include "ConnectionsSignal.h"

static constexpr int CIM_C_BEAT_INPUT_MAJOR = 0;
static constexpr int CIM_C_BEAT_OUTPUT_MAJOR = 1;

// CIMArray routes tile-shaped A/B/C data across input and output axes
template <int MACRO_INPUT_LANES, int MACRO_OUTPUT_LANES, int WEIGHT_SETS,
          int BASE_A_WIDTH, int BASE_B_WIDTH,
          int BASE_C_WIDTH, int MACRO_WRITE_INPUT_LANES, int MAC_LATENCY, int MODE,
          // A_WIDTH, B_WIDTH, and C_WIDTH come from the input, weight, and
          // accumulation datatypes. C_WIDTH sizes the reduced array result;
          // BASE_C_WIDTH sizes each macro's accumulator.
          int A_WIDTH, int B_WIDTH, int C_WIDTH, bool SIGNED,
          // Tile-internal layout; the input axis reduces into C while the
          // output axis retains distinct B/C channels
          int TILE_INPUT_AXIS_ELEMENTS, int TILE_OUTPUT_AXIS_ELEMENTS,
          // Input-axis tiles hold C partials that MACRequest::reduce optionally
          // combines
          int INPUT_AXIS_TILES,
          // Output-axis tiles hold distinct B/C channels that
          // MACRequest::multicast optionally shares A among
          int OUTPUT_AXIS_TILES,
          // Beat and port geometry: one ready/valid transfer moves a beat
          // containing an integer number of complete tile payloads.
          // A_PORT_TILES is the number of A tiles per beat. Each carries one
          // Tile::AData payload of Tile::INPUT_LANES scalars. The A beat spans
          // the full input axis; narrower beats would need assembly before
          // MAC issue.
          int A_PORT_TILES,
          // B_PORT_TILES is the number of B tiles per beat
          // A direct request addresses one aligned, non-wrapping span along the
          // output axis, so B_PORT_TILES must be no wider than that axis and
          // must evenly divide it. data[input_index] maps to
          // output_axis_tile_base + input_index. A replicate request copies
          // data[0] across the output axis and leaves the remaining payload
          // tiles unused. Routing a B beat wider than the output axis is not
          // implemented.
          int B_PORT_TILES,
          // C_PORT_TILES is the number of C tiles per beat
          int C_PORT_TILES,
          // C_BEAT_LAYOUT selects which tile axis advances first when one
          // operation requires multiple C beats. Input-major advances input
          // indices first; output-major advances output indices first
          int C_BEAT_LAYOUT,
          // Each output tile owns this many tile-vector result slots
          int RESULT_SLOTS_PER_OUTPUT_TILE = INPUT_AXIS_TILES>
SC_MODULE(CIMArray) {
 private:
  // Return the ceil log2 used for static port widths
  static constexpr int log2_ceil(int value) {
    return (value <= 1) ? 0 : 1 + log2_ceil((value + 1) / 2);
  }

  // Return the ceiling division for static beat counts
  static constexpr int ceil_div(int dividend, int divisor) {
    return (dividend + divisor - 1) / divisor;
  }

  using Tile =
      CIMTile<MACRO_INPUT_LANES, MACRO_OUTPUT_LANES, WEIGHT_SETS, BASE_A_WIDTH, BASE_B_WIDTH, BASE_C_WIDTH,
              MACRO_WRITE_INPUT_LANES, MAC_LATENCY, MODE, A_WIDTH, B_WIDTH, SIGNED,
              TILE_INPUT_AXIS_ELEMENTS, TILE_OUTPUT_AXIS_ELEMENTS>;

 public:
  static_assert(TILE_INPUT_AXIS_ELEMENTS > 0,
                "TILE_INPUT_AXIS_ELEMENTS must be positive");
  static_assert(TILE_OUTPUT_AXIS_ELEMENTS > 0,
                "TILE_OUTPUT_AXIS_ELEMENTS must be positive");
  static_assert(INPUT_AXIS_TILES > 0, "INPUT_AXIS_TILES must be positive");
  static_assert(OUTPUT_AXIS_TILES > 0, "OUTPUT_AXIS_TILES must be positive");
  static_assert(A_PORT_TILES > 0, "A_PORT_TILES must be positive");
  static_assert(B_PORT_TILES > 0, "B_PORT_TILES must be positive");
  static_assert(C_PORT_TILES > 0, "C_PORT_TILES must be positive");
  static_assert(RESULT_SLOTS_PER_OUTPUT_TILE >= INPUT_AXIS_TILES,
                "each output tile must hold one unreduced result");
  static_assert(
      A_PORT_TILES == INPUT_AXIS_TILES,
      "CIMArray currently requires one A beat to span the input axis");
  static_assert(
      B_PORT_TILES <= OUTPUT_AXIS_TILES,
      "CIMArray currently does not route one B beat beyond the output axis");
  static_assert(B_PORT_TILES == 0 || OUTPUT_AXIS_TILES % B_PORT_TILES == 0,
                "B_PORT_TILES must evenly divide OUTPUT_AXIS_TILES");
  static_assert(C_BEAT_LAYOUT == CIM_C_BEAT_INPUT_MAJOR ||
                    C_BEAT_LAYOUT == CIM_C_BEAT_OUTPUT_MAJOR,
                "C_BEAT_LAYOUT must be input-major or output-major");

  static constexpr int TILE_INPUT_LANES = Tile::INPUT_LANES;
  static constexpr int TILE_WRITE_INPUT_LANES = Tile::WRITE_INPUT_LANES;
  static constexpr int TILE_OUTPUT_LANES = Tile::OUTPUT_LANES;
  static constexpr int INPUT_LANES = TILE_INPUT_LANES * INPUT_AXIS_TILES;
  static constexpr int OUTPUT_LANES = TILE_OUTPUT_LANES * OUTPUT_AXIS_TILES;
  static constexpr int TILE_C_WIDTH = Tile::C_WIDTH;
  static constexpr int REDUCTION_GUARD_WIDTH =
      (INPUT_AXIS_TILES <= 1) ? 0 : log2_ceil(INPUT_AXIS_TILES);
  // C_WIDTH is the accumulation datatype width supplied by the caller; it must
  // hold one fully reduced result (all input tiles summed)
  static_assert(C_WIDTH >= TILE_C_WIDTH + REDUCTION_GUARD_WIDTH,
                "C_WIDTH must hold one reduced CIM array result");

  using AValue = typename Tile::AValue;
  using BValue = typename Tile::BValue;
  using TileCValue = typename Tile::CValue;
  using CValue = ac_int<C_WIDTH, false>;
  using Set = typename Tile::WeightSet;
  using TileInputIndex = typename Tile::InputIndex;
  using TileAData = typename Tile::AData;
  using TileBData = typename Tile::BData;
  using TileCData = typename Tile::CData;
  using CData = Pack1D<CValue, TILE_OUTPUT_LANES>;

  // A beat, B beat, and C beat each match the corresponding port payload
  using ABeat = Pack1D<TileAData, A_PORT_TILES>;
  using BBeat = Pack1D<TileBData, B_PORT_TILES>;
  using CBeat = Pack1D<CData, C_PORT_TILES>;

  // A complete MAC result may exceed the C port; this index selects one
  // C_PORT_TILES-wide result transfer
  static constexpr int MAX_C_RESULT_BEATS =
      ceil_div(INPUT_AXIS_TILES * OUTPUT_AXIS_TILES, C_PORT_TILES);
  static constexpr int C_RESULT_BEAT_INDEX_WIDTH =
      (MAX_C_RESULT_BEATS <= 1) ? 1 : log2_ceil(MAX_C_RESULT_BEATS);
  using CResultBeatIndex = ac_int<C_RESULT_BEAT_INDEX_WIDTH, false>;

  // The completion descriptor queue can never need more entries than the
  // total physical result slots because every accepted request owns at least
  // one slot
  static constexpr int MAC_ISSUE_WINDOW = Tile::issue_window();
  static constexpr int RESULT_SLOT_COUNT =
      OUTPUT_AXIS_TILES * RESULT_SLOTS_PER_OUTPUT_TILE;
  static constexpr int COMPLETION_QUEUE_DEPTH = RESULT_SLOT_COUNT;

  static constexpr int INPUT_TILE_INDEX_WIDTH =
      (INPUT_AXIS_TILES <= 1) ? 1 : log2_ceil(INPUT_AXIS_TILES);
  static constexpr int OUTPUT_TILE_INDEX_WIDTH =
      (OUTPUT_AXIS_TILES <= 1) ? 1 : log2_ceil(OUTPUT_AXIS_TILES);
  using InputTileIndex = ac_int<INPUT_TILE_INDEX_WIDTH, false>;
  using OutputTileIndex = ac_int<OUTPUT_TILE_INDEX_WIDTH, false>;

  // MACRequest carries one A beat plus tile-axis control
  struct MACRequest {
    Set compute_set;
    OutputTileIndex output_tile_index;  // selected tile when multicast is clear
    ac_int<1, false> multicast;
    ac_int<1, false> reduce;
    ABeat a;

    static const unsigned int width =
        Tile::SET_INDEX_WIDTH + OUTPUT_TILE_INDEX_WIDTH + 1 + 1 + ABeat::width;

    template <unsigned int Size>
    void Marshall(Marshaller<Size>& m) {
      m & compute_set;
      m & output_tile_index;
      m & multicast;
      m & reduce;
      m & a;
    }

    inline friend void sc_trace(sc_trace_file* tf, const MACRequest& request,
                                const std::string& name) {
      sc_trace(tf, request.compute_set, name + ".compute_set");
      sc_trace(tf, request.output_tile_index, name + ".output_tile_index");
      sc_trace(tf, request.multicast, name + ".multicast");
      sc_trace(tf, request.reduce, name + ".reduce");
      sc_trace(tf, request.a, name + ".a");
    }

    inline friend std::ostream& operator<<(ostream& os,
                                           const MACRequest& request) {
      os << request.compute_set << " ";
      os << request.output_tile_index << " ";
      os << request.multicast << " ";
      os << request.reduce << " ";
      os << request.a << " ";
      return os;
    }

    inline friend bool operator==(const MACRequest& lhs,
                                  const MACRequest& rhs) {
      return lhs.compute_set == rhs.compute_set &&
             lhs.output_tile_index == rhs.output_tile_index &&
             lhs.multicast == rhs.multicast && lhs.reduce == rhs.reduce &&
             lhs.a == rhs.a;
    }
  };

  // WriteRequest carries one B beat plus tile coordinates, write_set, and tile-local
  // write channel
  struct WriteRequest {
    Set write_set;
    InputTileIndex input_tile_index;
    OutputTileIndex
        output_axis_tile_base;  // first tile in a direct B-port span
    TileInputIndex write_input_index;
    ac_int<1, false> replicate;
    BBeat data;

    static const unsigned int width =
        Tile::SET_INDEX_WIDTH + INPUT_TILE_INDEX_WIDTH + OUTPUT_TILE_INDEX_WIDTH +
        Tile::INPUT_INDEX_WIDTH + 1 + BBeat::width;

    template <unsigned int Size>
    void Marshall(Marshaller<Size>& m) {
      m & write_set;
      m & input_tile_index;
      m & output_axis_tile_base;
      m & write_input_index;
      m & replicate;
      m & data;
    }

    inline friend void sc_trace(sc_trace_file* tf, const WriteRequest& request,
                                const std::string& name) {
      sc_trace(tf, request.write_set, name + ".write_set");
      sc_trace(tf, request.input_tile_index, name + ".input_tile_index");
      sc_trace(tf, request.output_axis_tile_base,
               name + ".output_axis_tile_base");
      sc_trace(tf, request.write_input_index, name + ".write_input_index");
      sc_trace(tf, request.replicate, name + ".replicate");
      sc_trace(tf, request.data, name + ".data");
    }

    inline friend std::ostream& operator<<(ostream& os,
                                           const WriteRequest& request) {
      os << request.write_set << " ";
      os << request.input_tile_index << " ";
      os << request.output_axis_tile_base << " ";
      os << request.write_input_index << " ";
      os << request.replicate << " ";
      os << request.data << " ";
      return os;
    }

    inline friend bool operator==(const WriteRequest& lhs,
                                  const WriteRequest& rhs) {
      return lhs.write_set == rhs.write_set && lhs.input_tile_index == rhs.input_tile_index &&
             lhs.output_axis_tile_base == rhs.output_axis_tile_base &&
             lhs.write_input_index == rhs.write_input_index && lhs.replicate == rhs.replicate &&
             lhs.data == rhs.data;
    }
  };

 private:
  Tile* tiles[INPUT_AXIS_TILES][OUTPUT_AXIS_TILES];

  sc_signal<TileAData> bus_a[INPUT_AXIS_TILES];
  sc_signal<Set> bus_compute_set;
  sc_signal<Set> held_compute_set;
  sc_signal<bool> mac_issue[OUTPUT_AXIS_TILES];

  sc_signal<bool> tile_write[INPUT_AXIS_TILES][OUTPUT_AXIS_TILES];
  sc_signal<Set> tile_write_set[INPUT_AXIS_TILES][OUTPUT_AXIS_TILES];
  sc_signal<TileInputIndex> tile_write_input_index[INPUT_AXIS_TILES][OUTPUT_AXIS_TILES];
  sc_signal<TileBData> tile_b[INPUT_AXIS_TILES][OUTPUT_AXIS_TILES];

  sc_signal<TileCData> tile_c[INPUT_AXIS_TILES][OUTPUT_AXIS_TILES];
  sc_signal<bool> tile_c_retire[INPUT_AXIS_TILES][OUTPUT_AXIS_TILES];
  sc_signal<bool> tile_mac_ready[INPUT_AXIS_TILES][OUTPUT_AXIS_TILES];
  sc_signal<bool> tile_mac_busy[INPUT_AXIS_TILES][OUTPUT_AXIS_TILES];

  static constexpr int RESULT_SLOT_SCALAR_WIDTH =
      TILE_C_WIDTH + REDUCTION_GUARD_WIDTH;
  using ResultSlotValue = ac_int<RESULT_SLOT_SCALAR_WIDTH, false>;
  using ResultSlot = Pack1D<ResultSlotValue, TILE_OUTPUT_LANES>;
  static constexpr int RESULT_SLOT_WIDTH = RESULT_SLOT_SCALAR_WIDTH * TILE_OUTPUT_LANES;
  static constexpr int RESULT_OUTPUT_TILE_WIDTH =
      RESULT_SLOTS_PER_OUTPUT_TILE * RESULT_SLOT_WIDTH;
  static constexpr int RESULT_OUTPUT_TILES_WIDTH =
      OUTPUT_AXIS_TILES * RESULT_OUTPUT_TILE_WIDTH;
  static constexpr int OUTPUT_TILE_SLOTS_WIDTH =
      OUTPUT_AXIS_TILES * RESULT_SLOT_WIDTH;
  using PackedResultSlot = ac_int<RESULT_SLOT_WIDTH, false>;
  using PackedResultOutputTile = ac_int<RESULT_OUTPUT_TILE_WIDTH, false>;
  using PackedResultOutputTiles = ac_int<RESULT_OUTPUT_TILES_WIDTH, false>;
  using PackedOutputTileSlots = ac_int<OUTPUT_TILE_SLOTS_WIDTH, false>;
  static constexpr int RESULT_SLOT_INDEX_WIDTH =
      (RESULT_SLOTS_PER_OUTPUT_TILE <= 1)
          ? 1
          : log2_ceil(RESULT_SLOTS_PER_OUTPUT_TILE);
  using ResultSlotIndex = ac_int<RESULT_SLOT_INDEX_WIDTH, false>;
  static constexpr int RESULT_SLOT_BASES_WIDTH =
      OUTPUT_AXIS_TILES * RESULT_SLOT_INDEX_WIDTH;
  using PackedResultSlotBases = ac_int<RESULT_SLOT_BASES_WIDTH, false>;
  // Keep physical slot bases output-tile-local instead of repeating them in
  // every globally ordered completion descriptor
  static constexpr int COMPLETION_TOKEN_WIDTH = OUTPUT_TILE_INDEX_WIDTH + 2;
  using CompletionToken = ac_int<COMPLETION_TOKEN_WIDTH, false>;
  static constexpr int COMPLETION_QUEUE_POINTER_WIDTH =
      log2_ceil(2 * COMPLETION_QUEUE_DEPTH);
  using CompletionQueuePointer = ac_int<COMPLETION_QUEUE_POINTER_WIDTH, false>;
  static constexpr int RESULT_SLOT_POINTER_WIDTH =
      log2_ceil(2 * RESULT_SLOTS_PER_OUTPUT_TILE);
  using ResultSlotPointer = ac_int<RESULT_SLOT_POINTER_WIDTH, false>;

  sc_signal<CompletionToken> completion_tokens[COMPLETION_QUEUE_DEPTH];
  sc_signal<ResultSlot> result_slots[OUTPUT_AXIS_TILES]
                                    [RESULT_SLOTS_PER_OUTPUT_TILE];
  sc_signal<CompletionQueuePointer> completion_allocate_pointer;
  sc_signal<CompletionQueuePointer> completion_capture_pointer;
  // Track the descriptor and beat currently presented at the result interface
  sc_signal<CompletionToken> completion_release_head;
  sc_signal<CResultBeatIndex> completion_release_beat_idx;
  sc_signal<bool> completion_release_head_valid;
  sc_signal<CompletionQueuePointer> completion_release_pointer;
  sc_signal<ResultSlotPointer> result_slot_allocate_pointer[OUTPUT_AXIS_TILES];
  sc_signal<ResultSlotPointer> result_slot_release_pointer[OUTPUT_AXIS_TILES];

#ifndef __SYNTHESIS__
  bool replicate_waste_warning_reported = false;
#endif

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  ConnectionsSignal::In<MACRequest> CCS_INIT_S1(mac_request_channel);
  ConnectionsSignal::In<WriteRequest> CCS_INIT_S1(write_request_channel);
  Connections::Out<CBeat, Connections::SYN_PORT> CCS_INIT_S1(result_channel);

  // Construct CIM tiles and HLS control threads
  SC_CTOR(CIMArray) {
    for (int input_tile_index = 0; input_tile_index < INPUT_AXIS_TILES;
         input_tile_index++) {
      for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
           output_tile_index++) {
        tiles[input_tile_index][output_tile_index] =
            new Tile(sc_gen_unique_name("tile"));

        tiles[input_tile_index][output_tile_index]->wclk(clk);
        tiles[input_tile_index][output_tile_index]->mclk(clk);
        tiles[input_tile_index][output_tile_index]->rstn(rstn);
        tiles[input_tile_index][output_tile_index]->write(
            tile_write[input_tile_index][output_tile_index]);
        tiles[input_tile_index][output_tile_index]->write_set(
            tile_write_set[input_tile_index][output_tile_index]);
        tiles[input_tile_index][output_tile_index]->write_input_index(
            tile_write_input_index[input_tile_index][output_tile_index]);
        tiles[input_tile_index][output_tile_index]->b(
            tile_b[input_tile_index][output_tile_index]);
        tiles[input_tile_index][output_tile_index]->a(bus_a[input_tile_index]);
        tiles[input_tile_index][output_tile_index]->compute_set(bus_compute_set);
        tiles[input_tile_index][output_tile_index]->mac_issue(
            mac_issue[output_tile_index]);
        tiles[input_tile_index][output_tile_index]->mac_ready(
            tile_mac_ready[input_tile_index][output_tile_index]);
        tiles[input_tile_index][output_tile_index]->mac_busy(
            tile_mac_busy[input_tile_index][output_tile_index]);
        tiles[input_tile_index][output_tile_index]->c(
            tile_c[input_tile_index][output_tile_index]);
        tiles[input_tile_index][output_tile_index]->c_retire(
            tile_c_retire[input_tile_index][output_tile_index]);
      }
    }

    SC_METHOD(drive_write_admission);
    sensitive << rstn << write_request_channel.vld << write_request_channel.dat;
    // Write admission depends on which set the MAC holds, so it needs
    // the same issue-path sensitivity as drive_mac_issue
    sensitive << held_compute_set << mac_request_channel.vld
              << mac_request_channel.dat;
    sensitive << completion_allocate_pointer << completion_release_pointer;
    sensitive << completion_release_head << completion_release_beat_idx
              << completion_release_head_valid << result_channel.vld
              << result_channel.rdy;
    for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
         output_tile_index++) {
      sensitive << result_slot_allocate_pointer[output_tile_index]
                << result_slot_release_pointer[output_tile_index];
    }
    for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
         output_tile_index++) {
      for (int input_tile_index = 0; input_tile_index < INPUT_AXIS_TILES;
           input_tile_index++) {
        sensitive << tile_mac_ready[input_tile_index][output_tile_index];
        sensitive << tile_mac_busy[input_tile_index][output_tile_index];
      }
    }

    SC_METHOD(drive_mac_issue);
    sensitive << rstn << mac_request_channel.vld << mac_request_channel.dat;
    sensitive << held_compute_set;
    sensitive << completion_allocate_pointer << completion_release_pointer;
    sensitive << completion_release_head << completion_release_beat_idx
              << completion_release_head_valid << result_channel.vld
              << result_channel.rdy;
    for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
         output_tile_index++) {
      sensitive << result_slot_allocate_pointer[output_tile_index]
                << result_slot_release_pointer[output_tile_index];
    }
    for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
         output_tile_index++) {
      for (int input_tile_index = 0; input_tile_index < INPUT_AXIS_TILES;
           input_tile_index++) {
        sensitive << tile_mac_ready[input_tile_index][output_tile_index];
      }
    }

    SC_THREAD(record_mac_issue);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(record_completion_release);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(capture_mac);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(drain_mac);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

 private:
  // Clear tile-level B signals
  void reset_write_side() {
    TileBData zero_b;
    pack::clear_pack(zero_b);
#pragma hls_unroll yes
    for (int input_tile_index = 0; input_tile_index < INPUT_AXIS_TILES;
         input_tile_index++) {
#pragma hls_unroll yes
      for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
           output_tile_index++) {
        tile_write[input_tile_index][output_tile_index].write(false);
        tile_write_set[input_tile_index][output_tile_index].write(0);
        tile_write_input_index[input_tile_index][output_tile_index].write(0);
        tile_b[input_tile_index][output_tile_index].write(zero_b);
      }
    }
  }

  // Route one direct aligned B-port span or replicate B beat tile zero
  void drive_write_request(const WriteRequest& request, bool request_valid) {
    TileBData zero_b;
    pack::clear_pack(zero_b);
#pragma hls_unroll yes
    for (int input_tile_index = 0; input_tile_index < INPUT_AXIS_TILES;
         input_tile_index++) {
      const bool input_selected =
          request.input_tile_index == InputTileIndex(input_tile_index);
#pragma hls_unroll yes
      for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
           output_tile_index++) {
        const int beat_offset = (output_tile_index / B_PORT_TILES) * B_PORT_TILES;
        const bool span_selected =
            request.output_axis_tile_base == OutputTileIndex(beat_offset);
        const bool selected =
            input_selected && (request.replicate != 0 || span_selected);
        tile_write[input_tile_index][output_tile_index].write(request_valid &&
                                                          selected);
        tile_write_set[input_tile_index][output_tile_index].write(request.write_set);
        tile_write_input_index[input_tile_index][output_tile_index].write(request.write_input_index);
        tile_b[input_tile_index][output_tile_index].write(
            selected ? request.data[request.replicate != 0
                                        ? 0
                                        : output_tile_index % B_PORT_TILES]
                     : zero_b);
      }
    }
  }

  // Admit the B stream combinationally so acceptance is the physical write edge
  void drive_write_admission() {
    WriteRequest request;
    request.write_set = 0;
    request.input_tile_index = 0;
    request.output_axis_tile_base = 0;
    request.write_input_index = 0;
    request.replicate = 0;
    pack::clear_pack(request.data);

    if (!rstn.read()) {
      ConnectionsSignal::set_ready(write_request_channel, false);
      reset_write_side();
      return;
    }

    const bool request_valid = ConnectionsSignal::valid(write_request_channel);
    if (request_valid) {
      request = ConnectionsSignal::peek(write_request_channel);
    }
    // A write to the set the MAC is using waits for the MAC to release it
    const bool blocked = request_valid && write_blocked_by_mac(request);
    ConnectionsSignal::set_ready(write_request_channel, !blocked);

#ifndef __SYNTHESIS__
    if (request_valid && request.input_tile_index.to_int() >= INPUT_AXIS_TILES) {
      SC_REPORT_FATAL("CIMArray",
                      "B request input_tile_index is outside the input axis");
    }
    if (request_valid && request.replicate == 0 &&
        (request.output_axis_tile_base.to_int() % B_PORT_TILES != 0 ||
         request.output_axis_tile_base.to_int() + B_PORT_TILES >
             OUTPUT_AXIS_TILES)) {
      SC_REPORT_FATAL(
          "CIMArray",
          "direct B request must select one aligned, non-wrapping B-port span");
    }
    if (request_valid && request.replicate != 0 &&
        request.output_axis_tile_base != OutputTileIndex(0)) {
      SC_REPORT_FATAL(
          "CIMArray",
          "replicate B request requires output_axis_tile_base zero");
    }
    if constexpr (B_PORT_TILES > 1) {
      if (request_valid && request.replicate != 0 &&
          !replicate_waste_warning_reported) {
        SC_REPORT_WARNING("CIMArray",
                          "replicate B request uses beat tile zero and ignores "
                          "the remaining tiles");
        replicate_waste_warning_reported = true;
      }
    }
#endif

    drive_write_request(request, request_valid && !blocked);
  }

  // Return the storage index selected by one completion queue pointer
  static int completion_queue_index(const CompletionQueuePointer& pointer) {
    const int pointer_value = pointer.to_uint();
    return pointer_value < COMPLETION_QUEUE_DEPTH
               ? pointer_value
               : pointer_value - COMPLETION_QUEUE_DEPTH;
  }

  // Advance one completion queue pointer through its index and phase range
  static CompletionQueuePointer next_completion_queue_pointer(
      const CompletionQueuePointer& pointer) {
    return pointer == CompletionQueuePointer(2 * COMPLETION_QUEUE_DEPTH - 1)
               ? CompletionQueuePointer(0)
               : CompletionQueuePointer(pointer + 1);
  }

  // Return whether every completion descriptor has been reserved
  static bool completion_queue_full(
      const CompletionQueuePointer& allocate_pointer,
      const CompletionQueuePointer& release_pointer) {
    const int release_value = release_pointer.to_uint();
    const CompletionQueuePointer full_pointer =
        release_value < COMPLETION_QUEUE_DEPTH
            ? CompletionQueuePointer(release_value + COMPLETION_QUEUE_DEPTH)
            : CompletionQueuePointer(release_value - COMPLETION_QUEUE_DEPTH);
    return allocate_pointer == full_pointer;
  }

  // Return the physical slot index selected by one extended slot pointer
  static int result_slot_index(const ResultSlotPointer& pointer) {
    const int pointer_value = pointer.to_uint();
    return pointer_value < RESULT_SLOTS_PER_OUTPUT_TILE
               ? pointer_value
               : pointer_value - RESULT_SLOTS_PER_OUTPUT_TILE;
  }

  // Advance one exact-depth slot pointer by a variable request allocation
  static ResultSlotPointer advance_result_slot_pointer(
      const ResultSlotPointer& pointer, int count) {
    int pointer_value = pointer.to_uint() + count;
    if (pointer_value >= 2 * RESULT_SLOTS_PER_OUTPUT_TILE) {
      pointer_value -= 2 * RESULT_SLOTS_PER_OUTPUT_TILE;
    }
    return ResultSlotPointer(pointer_value);
  }

  // Return the occupied slots between one output tile's pointer pair
  static int result_slot_occupancy(const ResultSlotPointer& allocate_pointer,
                                   const ResultSlotPointer& release_pointer) {
    const int allocate_value = allocate_pointer.to_uint();
    const int release_value = release_pointer.to_uint();
    return allocate_value >= release_value
               ? allocate_value - release_value
               : allocate_value + 2 * RESULT_SLOTS_PER_OUTPUT_TILE -
                     release_value;
  }

#ifndef __SYNTHESIS__
  // Return whether one MAC request selects a valid output-axis tile
  static bool mac_target_valid(const MACRequest& request) {
    return request.multicast != 0 ||
           request.output_tile_index.to_int() < OUTPUT_AXIS_TILES;
  }
#endif

  // Return whether one output tile is selected by a MAC request
  static bool mac_output_tile_selected(const MACRequest& request,
                                       int output_tile_index) {
    return request.multicast != 0 ||
           OutputTileIndex(output_tile_index) == request.output_tile_index;
  }

  // Return the number of output beats owned by one completion descriptor
  static int completion_result_beats(const CompletionToken& token) {
    const bool multicast = token.template slc<1>(OUTPUT_TILE_INDEX_WIDTH) != 0;
    const bool reduce = token.template slc<1>(OUTPUT_TILE_INDEX_WIDTH + 1) != 0;
    const int selected_output_tiles = multicast ? OUTPUT_AXIS_TILES : 1;
    const int logical_results = reduce
                                    ? selected_output_tiles
                                    : selected_output_tiles * INPUT_AXIS_TILES;
    return ceil_div(logical_results, C_PORT_TILES);
  }

  // Return whether the accepted C beat releases its completion descriptor
  bool completion_releases_on_fire() const {
    if (!completion_release_head_valid.read()) {
      return false;
    }
    const CompletionToken token = completion_release_head.read();
    return result_channel.vld.read() && result_channel.rdy.read() &&
           completion_release_beat_idx.read() ==
               CResultBeatIndex(completion_result_beats(token) - 1);
  }

  // Return whether one queued descriptor owns slots in an output tile
  static bool token_output_tile_selected(const CompletionToken& token,
                                         int output_tile_index) {
    const bool multicast = token.template slc<1>(OUTPUT_TILE_INDEX_WIDTH) != 0;
    const OutputTileIndex target =
        token.template slc<OUTPUT_TILE_INDEX_WIDTH>(0);
    return multicast || target == OutputTileIndex(output_tile_index);
  }

  // Return whether one queued descriptor stores early-reduced values
  static bool token_reduced(const CompletionToken& token) {
    return token.template slc<1>(OUTPUT_TILE_INDEX_WIDTH + 1) != 0;
  }

  // Return one output tile's physical base from packed drain state
  static ResultSlotIndex packed_result_slot_base(
      const PackedResultSlotBases& slot_bases, int output_tile_index) {
    return slot_bases.template slc<RESULT_SLOT_INDEX_WIDTH>(
        output_tile_index * RESULT_SLOT_INDEX_WIDTH);
  }

  // Return the slots one request owns in each selected output tile
  static int request_slots_per_selected_output_tile(bool reduce) {
    return reduce ? 1 : INPUT_AXIS_TILES;
  }

  // Track each completion-storage resource's admission result
  struct CompletionStorageAvailability {
    bool descriptors;
    bool result_slots;

    // Return whether every completion-storage resource is available
    bool all() const { return descriptors && result_slots; }
  };

  // Return completion-storage admission including same-edge release credit
  CompletionStorageAvailability completion_storage_availability(
      const MACRequest& request) const {
    const bool release_fire = completion_releases_on_fire();
    const CompletionToken release_token = completion_release_head.read();
    const int released_slots = release_fire
                                   ? request_slots_per_selected_output_tile(
                                         token_reduced(release_token))
                                   : 0;
    CompletionStorageAvailability availability;
    availability.descriptors =
        !completion_queue_full(completion_allocate_pointer.read(),
                               completion_release_pointer.read()) ||
        release_fire;
    availability.result_slots = true;
    const int requested_slots =
        request_slots_per_selected_output_tile(request.reduce != 0);

#pragma hls_unroll yes
    for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
         output_tile_index++) {
      if (mac_output_tile_selected(request, output_tile_index)) {
        const int occupied = result_slot_occupancy(
            result_slot_allocate_pointer[output_tile_index].read(),
            result_slot_release_pointer[output_tile_index].read());
        const int available_slots =
            RESULT_SLOTS_PER_OUTPUT_TILE - occupied +
            (release_fire &&
                     token_output_tile_selected(release_token, output_tile_index)
                 ? released_slots
                 : 0);
        availability.result_slots =
            availability.result_slots && available_slots >= requested_slots;
      }
    }
    return availability;
  }

  // Return whether any element still holds the B set of an in-flight MAC
  bool mac_set_busy() const {
    bool busy = false;
#pragma hls_unroll yes
    for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
         output_tile_index++) {
#pragma hls_unroll yes
      for (int input_tile_index = 0; input_tile_index < INPUT_AXIS_TILES;
           input_tile_index++) {
        busy = busy || tile_mac_busy[input_tile_index][output_tile_index].read();
      }
    }
    return busy;
  }

  // Protect the weights of both a pending request and an open issue window.
  // A stalled MAC still needs its selected set even before it can reserve
  // result storage. A new weight descriptor may already be waiting to reuse it.
  bool write_blocked_by_mac(const WriteRequest& request) const {
    const bool blocked_by_open_window =
        mac_set_busy() && request.write_set == held_compute_set.read();

    const bool mac_valid =
        rstn.read() && ConnectionsSignal::valid(mac_request_channel);
    bool blocked_by_pending_mac = false;
    if (mac_valid) {
      const MACRequest mac_request =
          ConnectionsSignal::peek(mac_request_channel);
      blocked_by_pending_mac = request.write_set == mac_request.compute_set;
    }

    return blocked_by_open_window || blocked_by_pending_mac;
  }

  // Return whether the shared tile issue bus can accept an operation
  bool mac_tiles_ready() const {
    bool ready = true;

#pragma hls_unroll yes
    for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
         output_tile_index++) {
#pragma hls_unroll yes
      for (int input_tile_index = 0; input_tile_index < INPUT_AXIS_TILES;
           input_tile_index++) {
        ready = ready && tile_mac_ready[input_tile_index][output_tile_index].read();
      }
    }
    return ready;
  }

  // Drive a newly accepted request and hold its shared set until every tile
  // closes the issue window
  void drive_mac_issue() {
    MACRequest request;
    request.compute_set = 0;
    request.output_tile_index = 0;
    request.multicast = 0;
    request.reduce = 0;
    pack::clear_pack(request.a);

    if (!rstn.read()) {
      ConnectionsSignal::set_ready(mac_request_channel, false);
      bus_compute_set.write(0);
#pragma hls_unroll yes
      for (int input_tile_index = 0; input_tile_index < INPUT_AXIS_TILES;
           input_tile_index++) {
        bus_a[input_tile_index].write(request.a[input_tile_index]);
      }
#pragma hls_unroll yes
      for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
           output_tile_index++) {
        mac_issue[output_tile_index].write(false);
      }
      return;
    }

    const bool valid = ConnectionsSignal::valid(mac_request_channel);
    if (valid) {
      request = ConnectionsSignal::peek(mac_request_channel);
    }
#ifndef __SYNTHESIS__
    if (valid && !mac_target_valid(request)) {
      SC_REPORT_FATAL("CIMArray",
                      "MAC request output_tile_index is outside the output axis");
    }
#endif
    const CompletionStorageAvailability storage =
        completion_storage_availability(request);
    const bool ready = rstn.read() && storage.all() && mac_tiles_ready();
    const bool fire = valid && ready;
    ConnectionsSignal::set_ready(mac_request_channel, ready);

    bus_compute_set.write(fire ? request.compute_set : held_compute_set.read());
#pragma hls_unroll yes
    for (int input_tile_index = 0; input_tile_index < INPUT_AXIS_TILES;
         input_tile_index++) {
      bus_a[input_tile_index].write(request.a[input_tile_index]);
    }
#pragma hls_unroll yes
    for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
         output_tile_index++) {
      mac_issue[output_tile_index].write(
          fire && mac_output_tile_selected(request, output_tile_index));
    }
  }

  // Reserve variable-length output-tile storage and one ordered descriptor
  void record_mac_issue() {
    CompletionQueuePointer allocate_pointer = 0;
    ResultSlotPointer slot_allocate_pointer[OUTPUT_AXIS_TILES];
    completion_allocate_pointer.write(allocate_pointer);
    held_compute_set.write(0);
#pragma hls_unroll yes
    for (int queue_idx = 0; queue_idx < COMPLETION_QUEUE_DEPTH; queue_idx++) {
      completion_tokens[queue_idx].write(CompletionToken(0));
    }
#pragma hls_unroll yes
    for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
         output_tile_index++) {
      slot_allocate_pointer[output_tile_index] = 0;
      result_slot_allocate_pointer[output_tile_index].write(0);
    }

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode bubble
    while (true) {
      if (ConnectionsSignal::fired(mac_request_channel)) {
        const MACRequest request = ConnectionsSignal::peek(mac_request_channel);
        held_compute_set.write(request.compute_set);

#ifndef __SYNTHESIS__
        if (!completion_storage_availability(request).all()) {
          SC_REPORT_FATAL("CIMArray",
                          "MAC issue overflowed completion storage despite "
                          "unavailable output-tile slots");
        }
#endif

        CompletionToken token = 0;
        token.set_slc(0, request.output_tile_index);
        token.set_slc(OUTPUT_TILE_INDEX_WIDTH, request.multicast);
        token.set_slc(OUTPUT_TILE_INDEX_WIDTH + 1, request.reduce);
#pragma hls_unroll yes
        for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
             output_tile_index++) {
          if (mac_output_tile_selected(request, output_tile_index)) {
            slot_allocate_pointer[output_tile_index] =
                advance_result_slot_pointer(
                    slot_allocate_pointer[output_tile_index],
                    request_slots_per_selected_output_tile(request.reduce !=
                                                           0));
            result_slot_allocate_pointer[output_tile_index].write(
                slot_allocate_pointer[output_tile_index]);
          }
        }
        completion_tokens[completion_queue_index(allocate_pointer)].write(
            token);
        allocate_pointer = next_completion_queue_pointer(allocate_pointer);
        completion_allocate_pointer.write(allocate_pointer);
      }
      wait();
    }
  }

  // Publish queue and output-tile release cursors from accepted final C beats
  void record_completion_release() {
    CompletionQueuePointer release_pointer = 0;
    ResultSlotPointer slot_release_pointer[OUTPUT_AXIS_TILES];
    CResultBeatIndex result_beat_idx = 0;
    CompletionToken release_head = 0;
    bool release_head_valid = false;
    completion_release_pointer.write(release_pointer);
    completion_release_head.write(release_head);
    completion_release_beat_idx.write(result_beat_idx);
    completion_release_head_valid.write(release_head_valid);
#pragma hls_unroll yes
    for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
         output_tile_index++) {
      slot_release_pointer[output_tile_index] = 0;
      result_slot_release_pointer[output_tile_index].write(0);
    }

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode bubble
    while (true) {
      const CompletionQueuePointer capture_pointer =
          completion_capture_pointer.read();
      const bool result_fire =
          result_channel.vld.read() && result_channel.rdy.read();
      bool load_head =
          !release_head_valid && release_pointer != capture_pointer;
      CompletionQueuePointer load_pointer = release_pointer;
#ifndef __SYNTHESIS__
      if (completion_release_pointer.read() != release_pointer) {
        SC_REPORT_FATAL("CIMArray", "release public/private pointers diverged");
      }
      if (result_fire && !release_head_valid) {
        SC_REPORT_FATAL("CIMArray",
                        "result transferred without a release descriptor");
      }
#endif
      if (result_fire && release_head_valid) {
        const bool token_complete =
            result_beat_idx ==
            CResultBeatIndex(completion_result_beats(release_head) - 1);
        if (token_complete) {
          const int released_slots = request_slots_per_selected_output_tile(
              token_reduced(release_head));
#pragma hls_unroll yes
          for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
               output_tile_index++) {
            if (token_output_tile_selected(release_head, output_tile_index)) {
              slot_release_pointer[output_tile_index] =
                  advance_result_slot_pointer(
                      slot_release_pointer[output_tile_index], released_slots);
              result_slot_release_pointer[output_tile_index].write(
                  slot_release_pointer[output_tile_index]);
            }
          }
          release_pointer = next_completion_queue_pointer(release_pointer);
          completion_release_pointer.write(release_pointer);
          result_beat_idx = 0;
          release_head_valid = false;
          if (release_pointer != capture_pointer) {
            load_head = true;
            load_pointer = release_pointer;
          }
        } else {
          result_beat_idx += CResultBeatIndex(1);
        }
      }

      // Keep the runtime table read outside the recurrent handshake predicate
      if (load_head) {
        release_head =
            completion_tokens[completion_queue_index(load_pointer)].read();
        release_head_valid = true;
      }
      completion_release_head.write(release_head);
      completion_release_beat_idx.write(result_beat_idx);
      completion_release_head_valid.write(release_head_valid);
      wait();
    }
  }

  // Return whether every tile selected by one operation has retired
  bool request_retired(bool multicast, int target_output_tile_index) const {
    bool retired = true;
#pragma hls_unroll yes
    for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
         output_tile_index++) {
      if (multicast || output_tile_index == target_output_tile_index) {
#pragma hls_unroll yes
        for (int input_tile_index = 0; input_tile_index < INPUT_AXIS_TILES;
             input_tile_index++) {
          retired =
              retired && tile_c_retire[input_tile_index][output_tile_index].read();
        }
      }
    }
    return retired;
  }

  // Sign- or zero-extend one result value to the array C width
  template <int VALUE_WIDTH>
  static CValue widen_result_value(ac_int<VALUE_WIDTH, false> value) {
    ac_int<VALUE_WIDTH, SIGNED> decoded;
    decoded.set_slc(0, value);
    const ac_int<C_WIDTH, SIGNED> widened = decoded;
    CValue result;
    result.set_slc(0, widened);
    return result;
  }

  // Add an input-axis offset to one circular output-tile allocation base
  static int offset_result_slot_index(ResultSlotIndex base, int offset) {
    int slot_idx = base.to_uint() + offset;
    if (slot_idx >= RESULT_SLOTS_PER_OUTPUT_TILE) {
      slot_idx -= RESULT_SLOTS_PER_OUTPUT_TILE;
    }
    return slot_idx;
  }

  // Pack every physical output tile with only static slice writes
  PackedResultOutputTiles read_packed_result_output_tiles() const {
    PackedResultOutputTiles output_tiles = 0;
#pragma hls_unroll yes
    for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
         output_tile_index++) {
#pragma hls_unroll yes
      for (int slot_idx = 0; slot_idx < RESULT_SLOTS_PER_OUTPUT_TILE;
           slot_idx++) {
#pragma hls_unroll yes
        for (int tile_output_index = 0; tile_output_index < TILE_OUTPUT_LANES; tile_output_index++) {
          const int bit_offset =
              ((output_tile_index * RESULT_SLOTS_PER_OUTPUT_TILE + slot_idx) *
                   RESULT_SLOT_WIDTH +
               tile_output_index * RESULT_SLOT_SCALAR_WIDTH);
          output_tiles.set_slc(
              bit_offset,
              result_slots[output_tile_index][slot_idx].read()[tile_output_index]);
        }
      }
    }
    return output_tiles;
  }

  // Select one packed slice through explicit constant-slice mux arms
  template <int SLICE_WIDTH, int SLICE_COUNT>
  static ac_int<SLICE_WIDTH, false> select_packed_slice(
      const ac_int<SLICE_WIDTH * SLICE_COUNT, false>& packed, int slice_idx) {
    ac_int<SLICE_WIDTH, false> selected = 0;
#pragma hls_unroll yes
    for (int candidate_idx = 0; candidate_idx < SLICE_COUNT; candidate_idx++) {
      const ac_int<SLICE_WIDTH, false> candidate =
          packed.template slc<SLICE_WIDTH>(candidate_idx * SLICE_WIDTH);
      if (slice_idx == candidate_idx) {
        selected = candidate;
      }
    }
    return selected;
  }

  // Select within each output tile before the generic output-tile router
  static PackedResultSlot select_generic_multicast_result_slot(
      const PackedResultOutputTiles& output_tiles,
      const PackedResultSlotBases& slot_bases, int output_tile_index,
      int input_tile_index) {
    PackedOutputTileSlots output_tile_slots = 0;
#pragma hls_unroll yes
    for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
         output_tile_index++) {
      const PackedResultOutputTile output_tile =
          output_tiles.template slc<RESULT_OUTPUT_TILE_WIDTH>(
              output_tile_index * RESULT_OUTPUT_TILE_WIDTH);
      const ResultSlotIndex slot_base =
          packed_result_slot_base(slot_bases, output_tile_index);
      const int slot_idx = offset_result_slot_index(slot_base, input_tile_index);
      const PackedResultSlot slot =
          select_packed_slice<RESULT_SLOT_WIDTH, RESULT_SLOTS_PER_OUTPUT_TILE>(
              output_tile, slot_idx);
      output_tile_slots.set_slc(output_tile_index * RESULT_SLOT_WIDTH, slot);
    }
    return select_packed_slice<RESULT_SLOT_WIDTH, OUTPUT_AXIS_TILES>(
        output_tile_slots, output_tile_index);
  }

  // Pack one C beat with shared target and output-tile-local multicast routing
  CBeat pack_queued_output(const PackedResultSlotBases& slot_bases,
                           CResultBeatIndex result_beat_idx, bool multicast,
                           bool reduce, int target_output_tile_index) const {
    const PackedResultOutputTiles output_tiles =
        read_packed_result_output_tiles();

    // A multicast request ignores its target field, which may hold a spare code
    const int routed_target =
        target_output_tile_index < OUTPUT_AXIS_TILES ? target_output_tile_index : 0;
    const PackedResultOutputTile targeted_output_tile =
        select_packed_slice<RESULT_OUTPUT_TILE_WIDTH, OUTPUT_AXIS_TILES>(
            output_tiles, routed_target);
    const ResultSlotIndex targeted_slot_base =
        packed_result_slot_base(slot_bases, routed_target);
    const int targeted_logical_results = reduce ? 1 : INPUT_AXIS_TILES;
    const int multicast_logical_results =
        reduce ? OUTPUT_AXIS_TILES : OUTPUT_AXIS_TILES * INPUT_AXIS_TILES;
    CBeat c_beat;
#pragma hls_unroll yes
    for (int port_idx = 0; port_idx < C_PORT_TILES; port_idx++) {
      const int logical_idx =
          result_beat_idx.to_int() * C_PORT_TILES + port_idx;

      PackedResultSlot targeted_slot = 0;
      if (logical_idx < targeted_logical_results) {
        const int slot_idx =
            offset_result_slot_index(targeted_slot_base, logical_idx);
        targeted_slot = select_packed_slice<RESULT_SLOT_WIDTH,
                                            RESULT_SLOTS_PER_OUTPUT_TILE>(
            targeted_output_tile, slot_idx);
      }

      PackedResultSlot multicast_slot = 0;
      // The processor geometry maps output port p directly to output tile p
      // Both reduce modes remain runtime-controlled
      if constexpr (C_BEAT_LAYOUT == CIM_C_BEAT_OUTPUT_MAJOR &&
                    C_PORT_TILES == OUTPUT_AXIS_TILES) {
        const PackedResultOutputTile multicast_output_tile =
            output_tiles.template slc<RESULT_OUTPUT_TILE_WIDTH>(
                port_idx * RESULT_OUTPUT_TILE_WIDTH);
        const ResultSlotIndex slot_base =
            packed_result_slot_base(slot_bases, port_idx);
        const int input_tile_index = reduce ? 0 : result_beat_idx.to_int();
        const int slot_idx =
            offset_result_slot_index(slot_base, input_tile_index);
        multicast_slot = select_packed_slice<RESULT_SLOT_WIDTH,
                                             RESULT_SLOTS_PER_OUTPUT_TILE>(
            multicast_output_tile, slot_idx);
      } else if (logical_idx < multicast_logical_results) {
        int output_tile_index = 0;
        int input_tile_index = 0;
        if (reduce) {
          output_tile_index = logical_idx;
        } else if constexpr (C_BEAT_LAYOUT == CIM_C_BEAT_INPUT_MAJOR) {
          output_tile_index = logical_idx / INPUT_AXIS_TILES;
          input_tile_index = logical_idx % INPUT_AXIS_TILES;
        } else {
          input_tile_index = logical_idx / OUTPUT_AXIS_TILES;
          output_tile_index = logical_idx % OUTPUT_AXIS_TILES;
        }
        multicast_slot = select_generic_multicast_result_slot(
            output_tiles, slot_bases, output_tile_index, input_tile_index);
      }

      const PackedResultSlot slot = multicast ? multicast_slot : targeted_slot;
#pragma hls_unroll yes
      for (int tile_output_index = 0; tile_output_index < TILE_OUTPUT_LANES; tile_output_index++) {
        const ResultSlotValue value =
            slot.template slc<RESULT_SLOT_SCALAR_WIDTH>(
                tile_output_index * RESULT_SLOT_SCALAR_WIDTH);
        c_beat[port_idx][tile_output_index] =
            reduce ? widen_result_value(value)
                   : widen_result_value(value.template slc<TILE_C_WIDTH>(0));
      }
    }
    return c_beat;
  }

  // Reduce one retiring output tile before its queue register
  //
  // When C_PORT_TILES equals OUTPUT_AXIS_TILES, as CIMProcessor requires, this
  // is effectively the same parallel reduction width needed by the drain path
  // and avoids storing every input-axis partial. A narrower C port could
  // instead reuse fewer drain-side trees across beats, so early reduction then
  // trades queue flops and muxes for more parallel adders. Large input axes may
  // also require a balanced or pipelined tree; HLS must retain capture II=1 and
  // meet the target clock after any geometry change
  ResultSlot reduce_retiring_output_tile(int output_tile_index) const {
    ResultSlot result;
    pack::clear_pack(result);
#pragma hls_unroll yes
    for (int tile_output_index = 0; tile_output_index < TILE_OUTPUT_LANES; tile_output_index++) {
      ac_int<RESULT_SLOT_SCALAR_WIDTH, SIGNED> sum = 0;
#pragma hls_unroll yes
      for (int input_tile_index = 0; input_tile_index < INPUT_AXIS_TILES;
           input_tile_index++) {
        ac_int<TILE_C_WIDTH, SIGNED> value;
        value.set_slc(0,
                      tile_c[input_tile_index][output_tile_index].read()[tile_output_index]);
        sum += value;
      }
      result[tile_output_index].set_slc(0, sum);
    }
    return result;
  }

  // Capture every fixed-latency retirement into its reserved output tiles
  void capture_mac() {
    CompletionQueuePointer capture_pointer = 0;
    // Track each output tile's physical base in global descriptor order
    ResultSlotIndex capture_slot_base[OUTPUT_AXIS_TILES];
    CompletionToken capture_head = 0;
    bool capture_head_valid = false;
    completion_capture_pointer.write(capture_pointer);
#pragma hls_unroll yes
    for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
         output_tile_index++) {
      capture_slot_base[output_tile_index] = 0;
#pragma hls_unroll yes
      for (int slot_idx = 0; slot_idx < RESULT_SLOTS_PER_OUTPUT_TILE;
           slot_idx++) {
        ResultSlot empty_slot;
        pack::clear_pack(empty_slot);
        result_slots[output_tile_index][slot_idx].write(empty_slot);
      }
    }

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode bubble
    while (true) {
      const CompletionQueuePointer allocate_pointer =
          completion_allocate_pointer.read();
      bool load_head = false;
      CompletionQueuePointer load_pointer = capture_pointer;

      // Allocation leads fixed-latency retirement, so cold heads can register
      // before their retire pulse without a bypass through the token table
      if (!capture_head_valid) {
        load_head = capture_pointer != allocate_pointer;
      } else {
        const CompletionToken token = capture_head;
        const bool multicast =
            token.template slc<1>(OUTPUT_TILE_INDEX_WIDTH) != 0;
        const bool reduce = token_reduced(token);
        const OutputTileIndex token_output_tile_index =
            token.template slc<OUTPUT_TILE_INDEX_WIDTH>(0);
        const int target_output_tile_index = token_output_tile_index.to_int();

        if (request_retired(multicast, target_output_tile_index)) {
#pragma hls_unroll yes
          for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
               output_tile_index++) {
            if (multicast || output_tile_index == target_output_tile_index) {
              const ResultSlotIndex slot_base =
                  capture_slot_base[output_tile_index];
              if (reduce) {
                result_slots[output_tile_index][slot_base.to_uint()].write(
                    reduce_retiring_output_tile(output_tile_index));
              } else {
#pragma hls_unroll yes
                for (int input_tile_index = 0; input_tile_index < INPUT_AXIS_TILES;
                     input_tile_index++) {
                  const TileCData tile_result =
                      tile_c[input_tile_index][output_tile_index].read();
                  ResultSlot raw_slot;
                  pack::clear_pack(raw_slot);
#pragma hls_unroll yes
                  for (int tile_output_index = 0; tile_output_index < TILE_OUTPUT_LANES; tile_output_index++) {
                    raw_slot[tile_output_index].set_slc(0, tile_result[tile_output_index]);
                  }
                  const int slot_idx =
                      offset_result_slot_index(slot_base, input_tile_index);
                  result_slots[output_tile_index][slot_idx].write(raw_slot);
                }
              }
              capture_slot_base[output_tile_index] = offset_result_slot_index(
                  slot_base, request_slots_per_selected_output_tile(reduce));
            }
          }

          capture_pointer = next_completion_queue_pointer(capture_pointer);
          completion_capture_pointer.write(capture_pointer);
          capture_head_valid = false;
          if (capture_pointer != allocate_pointer) {
            load_head = true;
            load_pointer = capture_pointer;
          }
        }
      }

      // Keep the runtime table read outside the recurrent retirement predicate
      if (load_head) {
        capture_head =
            completion_tokens[completion_queue_index(load_pointer)].read();
        capture_head_valid = true;
      }
      wait();
    }
  }

  // Serialize captured results while the release tracker observes handshakes
  void drain_mac() {
    result_channel.Reset();

    CompletionQueuePointer release_pointer = 0;
    ResultSlotPointer slot_release_pointer[OUTPUT_AXIS_TILES];
    CResultBeatIndex result_beat_idx = 0;
    CompletionToken drain_head = 0;
    // Keep one active-head base snapshot instead of widening every descriptor
    // or feeding the wide result gather from commit-owned release cursors
    PackedResultSlotBases drain_head_slot_bases = 0;
    bool drain_head_valid = false;
#pragma hls_unroll yes
    for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
         output_tile_index++) {
      slot_release_pointer[output_tile_index] = 0;
    }

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode bubble
    while (true) {
      const CompletionQueuePointer capture_pointer =
          completion_capture_pointer.read();
      const bool result_available = release_pointer != capture_pointer;
      bool pushed = false;
      bool load_head = false;
      CompletionQueuePointer load_pointer = release_pointer;

      // Prefetch issued metadata before capture makes its result visible
      if (!drain_head_valid) {
        load_head = release_pointer != completion_allocate_pointer.read();
      } else if (result_available) {
        const CompletionToken token = drain_head;
        const bool multicast =
            token.template slc<1>(OUTPUT_TILE_INDEX_WIDTH) != 0;
        const bool reduce =
            token.template slc<1>(OUTPUT_TILE_INDEX_WIDTH + 1) != 0;
        const OutputTileIndex token_output_tile_index =
            token.template slc<OUTPUT_TILE_INDEX_WIDTH>(0);
        const int target_output_tile_index = token_output_tile_index.to_int();

        const int result_beats = completion_result_beats(token);
        const bool token_complete =
            result_beat_idx == CResultBeatIndex(result_beats - 1);

        result_channel.Push(pack_queued_output(drain_head_slot_bases,
                                               result_beat_idx, multicast,
                                               reduce, target_output_tile_index));
        pushed = true;

        if (token_complete) {
#pragma hls_unroll yes
          for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
               output_tile_index++) {
            if (token_output_tile_selected(token, output_tile_index)) {
              slot_release_pointer[output_tile_index] =
                  advance_result_slot_pointer(
                      slot_release_pointer[output_tile_index],
                      request_slots_per_selected_output_tile(reduce));
            }
          }
          release_pointer = next_completion_queue_pointer(release_pointer);
          result_beat_idx = 0;
          drain_head_valid = false;

          // Push may stall while later requests arrive, so sample allocation
          // only after it returns before refilling the next head
          if (release_pointer != completion_allocate_pointer.read()) {
            load_head = true;
            load_pointer = release_pointer;
          }
        } else {
          result_beat_idx += CResultBeatIndex(1);
        }
      }

      // Keep the runtime table read outside the recurrent output controller
      if (load_head) {
        drain_head =
            completion_tokens[completion_queue_index(load_pointer)].read();
        PackedResultSlotBases next_slot_bases = 0;
#pragma hls_unroll yes
        for (int output_tile_index = 0; output_tile_index < OUTPUT_AXIS_TILES;
             output_tile_index++) {
          next_slot_bases.set_slc(output_tile_index * RESULT_SLOT_INDEX_WIDTH,
                                  ResultSlotIndex(result_slot_index(
                                      slot_release_pointer[output_tile_index])));
        }
        drain_head_slot_bases = next_slot_bases;
        drain_head_valid = true;
      }

      if (!pushed) {
        wait();
      }
    }
  }
};
