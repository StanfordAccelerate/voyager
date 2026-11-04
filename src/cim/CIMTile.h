// CIMTile owns a two-dimensional grid of CIM elements and exposes tile-shaped
// A/B/C data

#pragma once

#include <ac_int.h>
#include <systemc.h>

#include <sstream>

#include "AccelTypes.h"
#include "PackUtils.h"
#include "CIMElement.h"

// CIMTile maps activation vectors, weight blocks, and result vectors onto
// its CIMElement grid.
//
// Relative to a bare element the tile adds two registered stages: the A station
// captures A/compute_set on an accepted issue and pulses the elements on the following
// cycle, then the C stage registers the reduction across the input-axis
// elements. INPUT_AXIS_ELEMENTS and OUTPUT_AXIS_ELEMENTS describe the grid
// dimensions; no Connections protocol is exposed here. The input axis reduces
// into C while the output axis retains distinct B/C channels
template <int MACRO_INPUT_LANES, int MACRO_OUTPUT_LANES, int WEIGHT_SETS,
          int BASE_A_WIDTH, int BASE_B_WIDTH,
          int BASE_C_WIDTH, int MACRO_WRITE_INPUT_LANES, int MAC_LATENCY, int MODE,
          int A_WIDTH, int B_WIDTH, bool SIGNED, int INPUT_AXIS_ELEMENTS,
          int OUTPUT_AXIS_ELEMENTS>
SC_MODULE(CIMTile) {
 private:
  // Return the ceil log2 used for static port widths
  static constexpr int log2_ceil(int value) {
    return (value <= 1) ? 0 : 1 + log2_ceil((value + 1) / 2);
  }

  using Element = CIMElement<MACRO_INPUT_LANES, MACRO_OUTPUT_LANES, WEIGHT_SETS, BASE_A_WIDTH, BASE_B_WIDTH,
                             BASE_C_WIDTH, MACRO_WRITE_INPUT_LANES, MAC_LATENCY, MODE,
                             A_WIDTH, B_WIDTH, SIGNED>;

 public:
  static_assert(INPUT_AXIS_ELEMENTS > 0,
                "INPUT_AXIS_ELEMENTS must be positive");
  static_assert(OUTPUT_AXIS_ELEMENTS > 0,
                "OUTPUT_AXIS_ELEMENTS must be positive");

  static constexpr int ELEMENT_INPUT_LANES = Element::INPUT_LANES;
  static constexpr int ELEMENT_OUTPUT_LANES = Element::OUTPUT_LANES;
  static constexpr int ELEMENT_WRITE_INPUT_LANES = Element::WRITE_INPUT_LANES;
  static constexpr int ELEMENT_C_WIDTH = Element::C_WIDTH;
  static constexpr int INPUT_LANES = INPUT_AXIS_ELEMENTS * ELEMENT_INPUT_LANES;
  static constexpr int WRITE_INPUT_LANES = ELEMENT_WRITE_INPUT_LANES;
  // Output capacity spans all output-axis elements.
  static constexpr int OUTPUT_LANES = OUTPUT_AXIS_ELEMENTS * ELEMENT_OUTPUT_LANES;
  static constexpr int SET_INDEX_WIDTH = Element::SET_INDEX_WIDTH;
  static constexpr int INPUT_INDEX_WIDTH = (INPUT_LANES <= 1) ? 1 : log2_ceil(INPUT_LANES);

  // One weight write carries WRITE_INPUT_LANES consecutive input positions
  // for every output lane.
  // CIMTile converts write_input_index into the selected input-axis element and its
  // element-local input index

  // Reduction across the input-axis elements widens C by one guard field
  static constexpr int REDUCTION_GUARD_WIDTH =
      (INPUT_AXIS_ELEMENTS <= 1) ? 0 : log2_ceil(INPUT_AXIS_ELEMENTS);
  static constexpr int C_WIDTH = ELEMENT_C_WIDTH + REDUCTION_GUARD_WIDTH;

  using AValue = ac_int<A_WIDTH, false>;
  using BValue = ac_int<B_WIDTH, false>;
  using ElementCValue = ac_int<ELEMENT_C_WIDTH, false>;
  using CValue = ac_int<C_WIDTH, false>;
  using WeightSet = typename Element::WeightSet;
  using InputIndex = ac_int<INPUT_INDEX_WIDTH, false>;
  using ElementInputIndex = ac_int<Element::INPUT_INDEX_WIDTH, false>;
  using ElementAData = typename Element::AData;
  using ElementBData = typename Element::BData;
  using ElementCData = typename Element::CData;
  using AData = Pack1D<AValue, INPUT_LANES>;
  using BData = Pack1D<Pack1D<BValue, OUTPUT_LANES>, WRITE_INPUT_LANES>;
  using CData = Pack1D<CValue, OUTPUT_LANES>;

  // Return the number of mclk cycles an accepted issue keeps the tile not ready
  static constexpr int issue_window() { return Element::issue_window(); }

  // Return the number of mclk cycles from an accepted tile issue to retirement
  static constexpr int operation_latency() {
    return Element::operation_latency() + 2;
  }

  sc_in<bool> CCS_INIT_S1(wclk);
  sc_in<bool> CCS_INIT_S1(mclk);
  sc_in<bool> CCS_INIT_S1(rstn);

  // Tile write interface
  sc_in<bool> CCS_INIT_S1(write);
  sc_in<WeightSet> CCS_INIT_S1(write_set);
  sc_in<InputIndex> CCS_INIT_S1(write_input_index);
  sc_in<BData> CCS_INIT_S1(b);

  // Tile MAC issue interface
  sc_in<AData> CCS_INIT_S1(a);
  sc_in<WeightSet> CCS_INIT_S1(compute_set);
  sc_in<bool> CCS_INIT_S1(mac_issue);
  sc_out<bool> CCS_INIT_S1(mac_ready);
  // High while the elements consume A and the selected B set
  sc_out<bool> CCS_INIT_S1(mac_busy);

  // Tile C interface
  sc_out<CData> CCS_INIT_S1(c);
  // One-cycle pulse when c holds a new tile result
  sc_out<bool> CCS_INIT_S1(c_retire);

 private:
  Element* elements[INPUT_AXIS_ELEMENTS][OUTPUT_AXIS_ELEMENTS];

  sc_signal<bool> element_write[INPUT_AXIS_ELEMENTS];
  sc_signal<ElementInputIndex> element_write_input_index[INPUT_AXIS_ELEMENTS];
  sc_signal<WeightSet> element_write_set[INPUT_AXIS_ELEMENTS];
  sc_signal<ElementBData> element_b[INPUT_AXIS_ELEMENTS][OUTPUT_AXIS_ELEMENTS];

  // Combinationally slice the aggregate A payload before the registered station
  sc_signal<ElementAData> element_a_bus[INPUT_AXIS_ELEMENTS];

  // A station remains stable while the elements consume one issued operation
  sc_signal<ElementAData> station_a[INPUT_AXIS_ELEMENTS];
  sc_signal<WeightSet> station_compute_set;
  sc_signal<bool> element_mac_issue;

  sc_signal<ElementCData> element_c[INPUT_AXIS_ELEMENTS][OUTPUT_AXIS_ELEMENTS];
  sc_signal<bool> element_c_retire[INPUT_AXIS_ELEMENTS][OUTPUT_AXIS_ELEMENTS];
  sc_signal<bool> element_mac_ready[INPUT_AXIS_ELEMENTS][OUTPUT_AXIS_ELEMENTS];
  sc_signal<bool> element_mac_busy[INPUT_AXIS_ELEMENTS][OUTPUT_AXIS_ELEMENTS];

 public:
  // Construct the element grid and bind tile-local signals to its ports
  SC_CTOR(CIMTile) {
    for (int input_element_index = 0; input_element_index < INPUT_AXIS_ELEMENTS;
         input_element_index++) {
      for (int output_element_index = 0; output_element_index < OUTPUT_AXIS_ELEMENTS;
           output_element_index++) {
        elements[input_element_index][output_element_index] =
            new Element(sc_gen_unique_name("element"));

        elements[input_element_index][output_element_index]->wclk(wclk);
        elements[input_element_index][output_element_index]->mclk(mclk);
        elements[input_element_index][output_element_index]->rstn(rstn);
        elements[input_element_index][output_element_index]->wen(
            element_write[input_element_index]);
        elements[input_element_index][output_element_index]->write_input_index(
            element_write_input_index[input_element_index]);
        elements[input_element_index][output_element_index]->write_set(
            element_write_set[input_element_index]);
        elements[input_element_index][output_element_index]->b(
            element_b[input_element_index][output_element_index]);
        elements[input_element_index][output_element_index]->a(station_a[input_element_index]);
        elements[input_element_index][output_element_index]->compute_set(station_compute_set);
        elements[input_element_index][output_element_index]->mac_issue(element_mac_issue);
        elements[input_element_index][output_element_index]->mac_ready(
            element_mac_ready[input_element_index][output_element_index]);
        elements[input_element_index][output_element_index]->mac_busy(
            element_mac_busy[input_element_index][output_element_index]);
        elements[input_element_index][output_element_index]->c(
            element_c[input_element_index][output_element_index]);
        elements[input_element_index][output_element_index]->c_retire(
            element_c_retire[input_element_index][output_element_index]);
      }
    }

    SC_METHOD(drive_write);
    sensitive << rstn << write << write_set << write_input_index << b;

    SC_METHOD(drive_a);
    sensitive << rstn << a;

    SC_THREAD(run_issue);
    sensitive << mclk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(run_collect);
    sensitive << mclk.pos();
    async_reset_signal_is(rstn, false);

    SC_METHOD(drive_mac_ready);
    sensitive << rstn << element_mac_ready[0][0];

    SC_METHOD(drive_mac_busy);
    sensitive << rstn << element_mac_busy[0][0];
  }

 private:
  // Decode tile-local write_input_index to determine the selected input-axis element and its
  // element-local input lane index
  void drive_write() {
    const bool reset_released = rstn.read();
    const bool write_enabled = reset_released && write.read();
    int base_input_index = 0;
    WeightSet selected_write_set = 0;
    BData tile_b;
    pack::clear_pack(tile_b);
    if (reset_released) {
      base_input_index = write_input_index.read().to_int();
      selected_write_set = write_set.read();
      tile_b = b.read();
    }
    ElementBData zero_b;
    pack::clear_pack(zero_b);

#ifndef __SYNTHESIS__
    if (write_enabled && base_input_index >= INPUT_LANES) {
      std::ostringstream message;
      message << "write_input_index " << base_input_index << " is outside INPUT_LANES " << INPUT_LANES;
      SC_REPORT_ERROR("CIMTile write_input_index out of range", message.str().c_str());
    }
#endif

#pragma hls_unroll yes
    for (int input_element_index = 0; input_element_index < INPUT_AXIS_ELEMENTS;
         input_element_index++) {
      const int element_input_base = input_element_index * ELEMENT_INPUT_LANES;
      const bool element_selected =
          base_input_index >= element_input_base && base_input_index < element_input_base + ELEMENT_INPUT_LANES;
      const int element_input_index = base_input_index - element_input_base;
      const bool write_fits =
          element_input_index >= 0 && element_input_index + ELEMENT_WRITE_INPUT_LANES <= ELEMENT_INPUT_LANES;
      element_write[input_element_index].write(write_enabled && element_selected &&
                                          write_fits);
      element_write_input_index[input_element_index].write(write_fits ? element_input_index : 0);
      element_write_set[input_element_index].write(selected_write_set);

#ifndef __SYNTHESIS__
      if (write_enabled && element_selected && !write_fits) {
        std::ostringstream message;
        message << "write at write_input_index " << base_input_index << " crosses an element boundary";
        SC_REPORT_ERROR("CIMTile write_input_index protocol violation",
                        message.str().c_str());
      }
#endif

#pragma hls_unroll yes
      for (int output_element_index = 0; output_element_index < OUTPUT_AXIS_ELEMENTS;
           output_element_index++) {
        ElementBData element_data = zero_b;
#pragma hls_unroll yes
        for (int element_write_input_offset = 0; element_write_input_offset < ELEMENT_WRITE_INPUT_LANES; element_write_input_offset++) {
#pragma hls_unroll yes
          for (int element_output_index = 0; element_output_index < ELEMENT_OUTPUT_LANES; element_output_index++) {
            element_data[element_write_input_offset][element_output_index] =
                tile_b[element_write_input_offset][output_element_index * ELEMENT_OUTPUT_LANES + element_output_index];
          }
        }
        element_b[input_element_index][output_element_index].write(element_data);
      }
    }
  }

  // Slice the aggregate tile A payload into one payload per input-axis element
  void drive_a() {
    AData tile_a;
    pack::clear_pack(tile_a);
    if (rstn.read()) {
      tile_a = a.read();
    }
#pragma hls_unroll yes
    for (int input_element_index = 0; input_element_index < INPUT_AXIS_ELEMENTS;
         input_element_index++) {
      ElementAData element_a;
#pragma hls_unroll yes
      for (int element_input_index = 0; element_input_index < ELEMENT_INPUT_LANES; element_input_index++) {
        element_a[element_input_index] = tile_a[input_element_index * ELEMENT_INPUT_LANES + element_input_index];
      }
      element_a_bus[input_element_index].write(element_a);
    }
  }

  // Capture one tile A payload and issue it to every element on the next cycle
  // Clocked sc_signal writes become visible after the edge, so this station is
  // a pipeline register
  void run_issue() {
    ElementAData zero_a;
    pack::clear_pack(zero_a);

    element_mac_issue.write(false);
    station_compute_set.write(0);
#pragma hls_unroll yes
    for (int input_element_index = 0; input_element_index < INPUT_AXIS_ELEMENTS;
         input_element_index++) {
      station_a[input_element_index].write(zero_a);
    }

    wait();

    while (true) {
      if (mac_issue.read() && element_mac_ready[0][0].read()) {
#pragma hls_unroll yes
        for (int input_element_index = 0; input_element_index < INPUT_AXIS_ELEMENTS;
             input_element_index++) {
          station_a[input_element_index].write(element_a_bus[input_element_index].read());
        }
        station_compute_set.write(compute_set.read());
        element_mac_issue.write(true);
      } else {
        element_mac_issue.write(false);
      }

      wait();
    }
  }

  // Extend one element C value to the tile accumulator width while preserving
  // signedness
  static CValue widen_element_c(ElementCValue value) {
    CValue widened = 0;
    if constexpr (SIGNED) {
      ac_int<ELEMENT_C_WIDTH, true> signed_value;
      signed_value.set_slc(0, value);
      ac_int<C_WIDTH, true> signed_widened = signed_value;
      widened = signed_widened;
    } else {
      widened = value;
    }
    return widened;
  }

#ifndef __SYNTHESIS__
  // Verify that every element stays synchronized with representative element
  // [0][0]
  void check_element_lockstep(bool representative_retire) const {
    const bool reference_ready = element_mac_ready[0][0].read();
    const bool reference_busy = element_mac_busy[0][0].read();
    for (int input_element_index = 0; input_element_index < INPUT_AXIS_ELEMENTS;
         input_element_index++) {
      for (int output_element_index = 0; output_element_index < OUTPUT_AXIS_ELEMENTS;
           output_element_index++) {
        if (element_mac_ready[input_element_index][output_element_index].read() !=
            reference_ready) {
          std::ostringstream message;
          message << "element [" << input_element_index << "][" << output_element_index
                  << "] mac_ready diverged from [0][0]";
          SC_REPORT_ERROR("CIMTile element-ready lockstep violation",
                          message.str().c_str());
        }
        if (element_mac_busy[input_element_index][output_element_index].read() !=
            reference_busy) {
          std::ostringstream message;
          message << "element [" << input_element_index << "][" << output_element_index
                  << "] mac_busy diverged from [0][0]";
          SC_REPORT_ERROR("CIMTile element-busy lockstep violation",
                          message.str().c_str());
        }
        if (element_c_retire[input_element_index][output_element_index].read() !=
            representative_retire) {
          std::ostringstream message;
          message << "element [" << input_element_index << "][" << output_element_index
                  << "] c_retire diverged from [0][0]";
          SC_REPORT_ERROR("CIMTile element-retire lockstep violation",
                          message.str().c_str());
        }
      }
    }
  }
#endif

  // Register the tile C payload when representative element [0][0] retires
  void run_collect() {
    CData reset_c;
    pack::clear_pack(reset_c);
    c.write(reset_c);
    c_retire.write(false);

    wait();

    while (true) {
      // The elements retire in lockstep and pulse for exactly one cycle, so the
      // pulse is the trigger directly -- no retire phase to remember
      const bool element_retired = element_c_retire[0][0].read();
#ifndef __SYNTHESIS__
      check_element_lockstep(element_retired);
#endif

      // Republish the tile pulse every cycle so it is one cycle wide
      c_retire.write(element_retired);

      if (element_retired) {
        CData tile_c;
        // Keep the reduction inline so Catapult can statically enumerate every
        // element_c signal
#pragma hls_unroll yes
        for (int output_element_index = 0; output_element_index < OUTPUT_AXIS_ELEMENTS;
             output_element_index++) {
#pragma hls_unroll yes
          for (int element_output_index = 0; element_output_index < ELEMENT_OUTPUT_LANES; element_output_index++) {
            CValue sum = 0;
#pragma hls_unroll yes
            for (int input_element_index = 0; input_element_index < INPUT_AXIS_ELEMENTS;
                 input_element_index++) {
              sum += widen_element_c(
                  element_c[input_element_index][output_element_index].read()[element_output_index]);
            }
            tile_c[output_element_index * ELEMENT_OUTPUT_LANES + element_output_index] = sum;
          }
        }
        c.write(tile_c);
      }
      wait();
    }
  }

  // Forward representative element status for the lockstep grid
  void drive_mac_busy() {
    mac_busy.write(rstn.read() && element_mac_busy[0][0].read());
  }

  void drive_mac_ready() {
    mac_ready.write(rstn.read() && element_mac_ready[0][0].read());
  }
};
