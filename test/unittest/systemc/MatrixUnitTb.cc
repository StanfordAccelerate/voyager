// Exercise CIM through MatrixUnit's real controllers and input/accumulation
// SRAMs.
#include <cstdint>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "MatrixUnit.h"

static_assert(MATRIX_BACKEND == MATRIX_BACKEND_CIM,
              "This regression must use the CIM backend");
static_assert(INPUT_DTYPE_WIDTH == 8 && WEIGHT_DTYPE_WIDTH == 8,
              "The memory fixture supplies signed INT8 operands");
static_assert(IC_PORT_WIDTH % 8 == 0 && OC_PORT_WIDTH % 8 == 0,
              "The memory fixture uses byte-addressed ports");

using Result = Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>;
using Requests = std::map<unsigned, unsigned>;
static constexpr unsigned JOB_BYTES = 32768;
static constexpr unsigned REGION_BYTES = JOB_BYTES / 4;
static constexpr unsigned RESULT_BYTES = Result::width / 8;
static constexpr unsigned BIAS_BYTES = ACCUM_BUFFER_DATATYPE::width / 8;
static_assert(Result::width % OC_PORT_WIDTH == 0,
              "Output fixtures require complete memory beats");

struct Shape {
  int x = 1, y = 1, outer_x = 1, outer_y = 1;
  int ic = 1, outer_ic = 1, oc = 1, outer_oc = 1;
  int fx = 1, fy = 1, stride = 1;
  bool spatial_first = false, set_major = false;
  bool bias = false, to_memory = false, buffered = false;
};

struct Expected {
  Result value;
  unsigned address;
  int y, x, group;
};

struct Job {
  std::string name;
  Shape shape;
  MatrixParams params;
  std::vector<Expected> expected;
  Requests inputs, weights, biases;
};

SC_MODULE(MatrixUnitTb) {
  sc_clock clk{"clk", 10, SC_NS};
  sc_signal<bool> rstn{"rstn", true};
  MatrixUnit dut{"dut"};
  Connections::Combinational<ac_int<64, false>> params;
  Connections::Combinational<MemoryRequest> input_req, weight_req, bias_req;
  Connections::Combinational<ac_int<IC_PORT_WIDTH, false>> input_resp;
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> weight_resp,
      bias_resp;
  Connections::Combinational<Result> output;
  Connections::Combinational<ac_int<OC_PORT_WIDTH, false>> output_data;
  Connections::Combinational<ac_int<ADDRESS_WIDTH, false>> output_addr;
  Connections::SyncChannel start, done;

  std::vector<Job> jobs;
  std::vector<uint8_t> memory;
  Requests input_requests, weight_requests, bias_requests;
  bool failed = false;
  bool vectors_done = false, memory_done = false, sync_done = false;

  SC_HAS_PROCESS(MatrixUnitTb);
  MatrixUnitTb(sc_module_name name, const std::string& selected)
      : sc_module(name) {
    dut.clk(clk);
    dut.rstn(rstn);
    dut.serial_params_in(params);
    dut.input_req(input_req);
    dut.input_resp(input_resp);
    dut.weight_req(weight_req);
    dut.weight_resp(weight_resp);
    dut.bias_req(bias_req);
    dut.bias_resp(bias_resp);
    dut.output_channel(output);
    dut.output_data(output_data);
    dut.output_addr(output_addr);
    dut.start(start);
    dut.done(done);

    Shape s;
    s.x = 2;
    s.outer_x = 3;
    add("reuse", s, 1);

    s = Shape{};
    s.x = 3;
    s.outer_x = 2;
    s.oc = CIM_WEIGHT_SETS;
    s.spatial_first = true;
    s.bias = true;
    s.buffered = true;
    add("resident_sequence", s, 1);

    // One more set forces the controller to refill on each spatial replay.
    s.oc = CIM_WEIGHT_SETS + 1;
    s.bias = false;
    add("ring_refill", s, s.x * s.outer_x);

    s = Shape{};
    s.x = 2;
    s.ic = 2;
    s.bias = true;
    add("local_reduction", s, 1);

    s.x = 8;  // More live outputs than the four local accumulation contexts.
    s.bias = false;
    s.buffered = true;
    add("sram_reduction", s, 1);

    s.x = 4;
    s.outer_x = 2;
    s.outer_ic = 2;
    s.bias = true;
    add("outer_reduction", s, s.outer_x);

    // Keep two outer output tiles live across a reduction. This schedule uses
    // direct output: banked output requires completing each outer tile first.
    s.x = 2;
    s.ic = 1;
    s.set_major = true;
    s.buffered = false;
    add("set_major_replay", s, 1);

    s = Shape{};
    s.x = 3;
    s.y = 2;
    s.outer_x = 2;
    s.ic = 2;
    s.fx = s.fy = 3;
    s.bias = true;
    s.buffered = true;
    add("convolution", s, s.outer_x);

    s.x = 2;
    s.outer_y = 2;
    s.ic = 1;
    s.stride = 2;
    s.bias = false;
    add("strided_convolution", s, s.outer_x * s.outer_y);

    s = Shape{};
    s.x = 3;
    s.y = 2;
    s.outer_x = 2;
    s.outer_oc = 2;
    s.ic = s.oc = 2;
    s.bias = true;
    s.to_memory = true;
    s.buffered = true;
    add("memory_output", s, s.outer_x);

    s = Shape{};
    s.oc = 2;
    s.x = ACCUM_BUFFER_SIZE / s.oc;
    s.ic = 2;
    s.buffered = true;
    add("full_buffer", s, 1);

    s = Shape{};
    s.x = 3;
    add("direct_after_buffered", s, 1);

    // Filtering retains each job's operand seed and memory addresses.
    for (auto it = jobs.begin(); it != jobs.end();) {
      if (!selected.empty() && it->name != selected)
        it = jobs.erase(it);
      else
        ++it;
    }
    for (const auto& j : jobs) {
      input_requests.insert(j.inputs.begin(), j.inputs.end());
      weight_requests.insert(j.weights.begin(), j.weights.end());
      bias_requests.insert(j.biases.begin(), j.biases.end());
    }

    SC_THREAD(run);
    sensitive << clk.posedge_event();
    SC_THREAD(inputs);
    sensitive << clk.posedge_event();
    SC_THREAD(weights);
    sensitive << clk.posedge_event();
    SC_THREAD(biases);
    sensitive << clk.posedge_event();
    SC_THREAD(vectors);
    sensitive << clk.posedge_event();
    SC_THREAD(memory_outputs);
    sensitive << clk.posedge_event();
    SC_THREAD(synchronization);
    sensitive << clk.posedge_event();
    SC_THREAD(watchdog);
  }

  // Distinct signed tensors make stale weights, partial sums, and banks
  // visible.
  static int input_value(int seed, int y, int x, int channel) {
    return (seed * 11 + y * 7 + x * 3 + channel * 5) % 19 - 9;
  }
  static int weight_value(int seed, int fy, int fx, int channel, int out) {
    return (seed * 3 + fy * 11 + fx * 7 + channel * 5 + out * 2) % 17 - 8;
  }
  static int bias_value(int seed, int out) { return seed * 5 + out * 7 - 19; }

  void add(const char* name, Shape s, unsigned weight_passes) {
    Job j{name, s, {}, {}, {}, {}, {}};
    auto& p = j.params;
    for (int level = 0; level < 2; ++level) {
      for (int slot = 0; slot < 6; ++slot) p.loops[level][slot] = 1;
      for (int slot = 0; slot < 5; ++slot) p.weight_addr_loops[level][slot] = 1;
    }
    // Physical slots are ordered outermost first. Alternate L1 orders exercise
    // retaining one weight set and replaying an entire resident sequence.
    p.y_loop_idx[0] = s.set_major ? 2 : 0;
    p.x_loop_idx[0] = s.set_major ? 3 : 1;
    p.weight_loop_idx[0] = s.set_major ? 0 : 2;
    p.reduction_loop_idx[0] = s.set_major ? 1 : 3;
    p.fy_loop_idx[0] = 4;
    p.reduction_loop_idx[1] = s.spatial_first ? 2 : 0;
    p.weight_loop_idx[1] = s.spatial_first ? 3 : 1;
    p.fy_loop_idx[1] = s.spatial_first ? 4 : 2;
    p.fx_loop_idx = s.spatial_first ? 5 : 3;
    p.y_loop_idx[1] = s.spatial_first ? 0 : 4;
    p.x_loop_idx[1] = s.spatial_first ? 1 : 5;
    p.loops[0][p.y_loop_idx[0]] = s.outer_y;
    p.loops[0][p.x_loop_idx[0]] = s.outer_x;
    p.loops[0][p.weight_loop_idx[0]] = s.outer_oc;
    p.loops[0][p.reduction_loop_idx[0]] = s.outer_ic;
    p.loops[1][p.y_loop_idx[1]] = s.y;
    p.loops[1][p.x_loop_idx[1]] = s.x;
    p.loops[1][p.weight_loop_idx[1]] = s.oc;
    p.loops[1][p.reduction_loop_idx[1]] = s.ic;
    p.loops[1][p.fy_loop_idx[1]] = s.fy;
    p.loops[1][p.fx_loop_idx] = s.fx;

    // Source tensors use NHWC inputs and HWIO (CK for GEMM) weights.
    p.weight_addr_weight_loop_idx[0] = 0;
    p.weight_addr_reduction_loop_idx[0] = 1;
    p.weight_addr_fy_idx[0] = 2;
    p.weight_addr_loops[0][0] = s.outer_oc;
    p.weight_addr_loops[0][1] = s.outer_ic;
    p.weight_addr_fy_idx[1] = 0;
    p.weight_addr_fx_idx = 1;
    p.weight_addr_reduction_loop_idx[1] = 2;
    p.weight_addr_reduction_loop_idx[2] = 3;
    p.weight_addr_weight_loop_idx[1] = 4;
    p.weight_addr_loops[1][0] = s.fy;
    p.weight_addr_loops[1][1] = s.fx;
    p.weight_addr_loops[1][2] = s.ic;
    p.weight_addr_loops[1][3] = IC_DIMENSION;
    p.weight_addr_loops[1][4] = s.oc;
    p.input_pack_factor_lg2 = p.weight_pack_factor_lg2 = 0;
    p.input_burst_size = IC_DIMENSION;
    p.input_num_beats = (IC_DIMENSION * 8 + IC_PORT_WIDTH - 1) / IC_PORT_WIDTH;
    p.weight_burst_size = OC_DIMENSION;
    p.weight_num_beats = (OC_DIMENSION * 8 + OC_PORT_WIDTH - 1) / OC_PORT_WIDTH;
    p.stride = s.stride;
    const int output_x = s.x * s.outer_x, output_y = s.y * s.outer_y;
    const int input_x = (output_x - 1) * s.stride + s.fx;
    const int input_y = (output_y - 1) * s.stride + s.fy;
    const int channels = s.ic * s.outer_ic * IC_DIMENSION;
    const int outputs = s.oc * s.outer_oc * OC_DIMENSION;
    p.input_x = input_x;
    p.input_y = input_y;
    const int seed = jobs.size();
    const unsigned base = JOB_BYTES * (seed + 1);
    memory.resize(base + JOB_BYTES, 0xA5);
    p.input_offset = base;
    p.weight_offset = base + REGION_BYTES;
    p.bias_offset = base + 2 * REGION_BYTES;
    p.output_offset = base + 3 * REGION_BYTES;
    p.has_bias = s.bias;
    p.output_to_memory = s.to_memory;
    p.write_output_to_accum_buffer = DOUBLE_BUFFERED_ACCUM_BUFFER && s.buffered;
    if (input_x * input_y * channels > REGION_BYTES ||
        s.fy * s.fx * channels * outputs > REGION_BYTES ||
        output_x * output_y * outputs * BIAS_BYTES > REGION_BYTES)
      throw std::runtime_error("fixture exceeds its memory region");

    for (int y = 0; y < input_y; ++y)
      for (int x = 0; x < input_x; ++x)
        for (int c = 0; c < channels; ++c)
          memory.at(base + (y * input_x + x) * channels + c) =
              input_value(seed, y, x, c);
    for (int fy = 0; fy < s.fy; ++fy)
      for (int fx = 0; fx < s.fx; ++fx)
        for (int c = 0; c < channels; ++c)
          for (int out = 0; out < outputs; ++out) {
            const unsigned address =
                unsigned(p.weight_offset) +
                ((fy * s.fx + fx) * channels + c) * outputs + out;
            memory.at(address) = weight_value(seed, fy, fx, c, out);
            if (out % OC_DIMENSION == 0) j.weights[address] = weight_passes;
          }
    for (int out = 0; out < outputs; ++out) {
      const uint32_t bits = bias_value(seed, out);
      for (unsigned byte = 0; byte < BIAS_BYTES; ++byte)
        memory.at(unsigned(p.bias_offset) + out * BIAS_BYTES + byte) =
            bits >> (byte * 8);
      if (s.bias && out % OC_DIMENSION == 0)
        j.biases[unsigned(p.bias_offset) + out * BIAS_BYTES] =
            s.outer_x * s.outer_y * (s.spatial_first ? s.x * s.y : 1);
    }

    // Check the complete input window fetched for each tile, including halos.
    // A strided multi-tap window reserves a full stride for its last output;
    // coordinates past the tensor boundary are zero-filled without a request.
    const int window_x = (s.fx == 1 ? s.x : s.x * s.stride) + s.fx - 1;
    const int window_y = (s.fy == 1 ? s.y : s.y * s.stride) + s.fy - 1;
    for (int ty = 0; ty < s.outer_y; ++ty)
      for (int tx = 0; tx < s.outer_x; ++tx)
        for (int wy = 0; wy < window_y; ++wy)
          for (int wx = 0; wx < window_x; ++wx) {
            const int y = ty * s.y * s.stride + wy * (s.fy == 1 ? s.stride : 1);
            const int x = tx * s.x * s.stride + wx * (s.fx == 1 ? s.stride : 1);
            if (x >= input_x || y >= input_y) continue;
            for (int c = 0; c < channels; c += IC_DIMENSION)
              j.inputs[base + (y * input_x + x) * channels + c] += s.outer_oc;
          }

    // Gold is an ordinary dense convolution, independent of descriptors,
    // resident sets, accumulation storage, and the order of partial sums.
    for (int ty = 0; ty < s.outer_y; ++ty)
      for (int tx = 0; tx < s.outer_x; ++tx)
        for (int ko = 0; ko < s.outer_oc; ++ko)
          for (int index = 0; index < s.y * s.x * s.oc; ++index) {
            const int spatial =
                s.spatial_first ? index / s.oc : index % (s.y * s.x);
            const int group =
                ko * s.oc +
                (s.spatial_first ? index % s.oc : index / (s.y * s.x));
            const int y = ty * s.y + spatial / s.x;
            const int x = tx * s.x + spatial % s.x;
            Expected expected{
                {},
                unsigned(p.output_offset) +
                    ((y * output_x + x) * (outputs / OC_DIMENSION) + group) *
                        RESULT_BYTES,
                y,
                x,
                group};
            for (int lane = 0; lane < OC_DIMENSION; ++lane) {
              const int out = group * OC_DIMENSION + lane;
              int sum = s.bias ? bias_value(seed, out) : 0;
              for (int fy = 0; fy < s.fy; ++fy)
                for (int fx = 0; fx < s.fx; ++fx)
                  for (int c = 0; c < channels; ++c)
                    sum += input_value(seed, y * s.stride + fy,
                                       x * s.stride + fx, c) *
                           weight_value(seed, fy, fx, c, out);
              expected.value[lane] = ACCUM_BUFFER_DATATYPE::from_bits(sum);
            }
            j.expected.push_back(expected);
          }
    jobs.push_back(std::move(j));
  }

  bool check(bool ok, const std::string& message) {
    if (!ok) {
      failed = true;
      std::cerr << "FAIL " << message << " @ " << sc_time_stamp() << '\n';
      sc_stop();
    }
    return ok;
  }

  void run() {
    params.ResetWrite();
    rstn = false;
    wait(5);
    rstn = true;
    wait(5);
    if (!check(!jobs.empty(), "unknown case")) return;
    // Queue commands without resets, allowing prefetch and bank handoff to
    // overlap. Bias/no-bias and banked/direct transitions must retain no state.
    for (const auto& j : jobs) {
      ac_int<((MatrixParams::width + 63) / 64) * 64, false> bits =
          BitsToType<ac_int<MatrixParams::width, false>>(TypeToBits(j.params));
      for (int beat = 0; beat < bits.width / 64; ++beat)
        params.Push(bits.slc<64>(beat * 64));
    }
    while (!(vectors_done && memory_done && sync_done)) wait();
    wait(50);  // Keep peers active to catch extra traffic after completion.
    if (!check(input_requests.empty() && weight_requests.empty() &&
                   bias_requests.empty(),
               "missing memory requests"))
      return;
    std::cout << "PASS CIM MatrixUnit mode=" << CIM_MODE
              << " double_buffer=" << DOUBLE_BUFFERED_ACCUM_BUFFER << ": "
              << jobs.size() << " queued jobs\n";
    sc_stop();
  }

  template <int Width>
  void serve(Connections::Combinational<MemoryRequest> & requests,
             Connections::Combinational<ac_int<Width, false>> & responses,
             Requests & expected, unsigned burst, const char* kind) {
    requests.ResetRead();
    responses.ResetWrite();
    wait(10);
    unsigned count = 0;
    while (true) {
      const auto request = requests.Pop();
      const unsigned address = request.address.to_uint();
      auto it = expected.find(address);
      std::ostringstream message;
      message << kind << " address=" << address
              << " burst=" << request.burst_size;
      if (!check(request.address.to_uint64() == address &&
                     request.burst_size == burst && it != expected.end(),
                 message.str()))
        return;
      if (--it->second == 0) expected.erase(it);
      wait(2 + (++count % 3));
      for (unsigned offset = 0; offset < burst; offset += Width / 8) {
        ac_int<Width, false> word = 0;
        for (unsigned byte = 0; byte < Width / 8; ++byte)
          word.set_slc(byte * 8,
                       ac_int<8, false>(memory.at(address + offset + byte)));
        responses.Push(word);
        wait();
      }
    }
  }
  void inputs() {
    serve(input_req, input_resp, input_requests, IC_DIMENSION, "input");
  }
  void weights() {
    serve(weight_req, weight_resp, weight_requests, OC_DIMENSION, "weight");
  }
  void biases() {
    serve(bias_req, bias_resp, bias_requests, RESULT_BYTES, "bias");
  }

  void vectors() {
    output.ResetRead();
    wait(10);
    for (const auto& j : jobs) {
      if (j.params.output_to_memory) continue;
      for (const auto& expected : j.expected) {
        wait(13);  // Backpressure the processor and accumulation-bank reuse.
        const auto actual = output.Pop();
        std::ostringstream message;
        message << j.name << " y=" << expected.y << " x=" << expected.x
                << " group=" << expected.group << " got [" << actual
                << "] expected [" << expected.value << ']';
        if (!check(actual == expected.value, message.str())) return;
      }
      std::cout << "PASS " << j.name << '\n';
    }
    vectors_done = true;
    while (true) {
      Result extra;
      if (output.PopNB(extra)) {
        check(false, "extra vector output");
        return;
      }
      wait();
    }
  }

  void memory_outputs() {
    output_data.ResetRead();
    output_addr.ResetRead();
    wait(10);
    for (const auto& j : jobs) {
      if (!j.params.output_to_memory) continue;
      for (const auto& expected : j.expected) {
        const auto bits = BitsToType<ac_int<Result::width, false>>(
            TypeToBits(expected.value));
        for (unsigned beat = 0; beat < Result::width / OC_PORT_WIDTH; ++beat) {
          wait(17);
          const auto data = output_data.Pop();
          wait(2);
          const auto address = output_addr.Pop();
          if (!check(data == bits.slc<OC_PORT_WIDTH>(beat * OC_PORT_WIDTH) &&
                         address == expected.address + beat * OC_PORT_WIDTH / 8,
                     j.name + " memory output data/address"))
            return;
        }
      }
      std::cout << "PASS " << j.name << '\n';
    }
    memory_done = true;
    while (true) {
      ac_int<OC_PORT_WIDTH, false> data;
      ac_int<ADDRESS_WIDTH, false> address;
      if (output_data.PopNB(data) || output_addr.PopNB(address)) {
        check(false, "extra memory output");
        return;
      }
      wait();
    }
  }

  void synchronization() {
    start.ResetRead();
    done.ResetRead();
    wait(10);
    for (const auto& j : jobs) {
      start.SyncPop();
      done.SyncPop();
    }
    sync_done = true;
    while (true) {
      if (start.SyncPopNB() || done.SyncPopNB()) {
        check(false, "extra start/done");
        return;
      }
      wait();
    }
  }
  void watchdog() {
    wait(2, SC_MS);
    check(false, "watchdog timeout");
  }
};

int sc_main(int argc, char** argv) {
  std::string selected;
  if (argc == 3 && std::string(argv[1]) == "--case")
    selected = argv[2];
  else if (argc != 1) {
    std::cerr << "Usage: " << argv[0] << " [--case NAME]\n";
    return 2;
  }
  MatrixUnitTb tb("tb", selected);
  sc_start();
  return tb.failed ? 1 : 0;
}
