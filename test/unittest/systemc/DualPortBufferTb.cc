// Check write progress and response ownership under read backpressure
#include "AccelTypes.h"
#include "DualPortBuffer.h"

using Word = ac_int<32, false>;
static constexpr int PORTS = DOUBLE_BUFFERED_ACCUM_BUFFER ? 4 : 1;
#if DOUBLE_BUFFERED_ACCUM_BUFFER
static constexpr int CASE_PORTS[] = {0, 1, 0, 2, 3, 2};
#else
static constexpr int CASE_PORTS[] = {0, 0};
#endif
static constexpr int CASES = sizeof(CASE_PORTS) / sizeof(CASE_PORTS[0]);

// Delay responses while issuing writes, then verify every stored word
SC_MODULE(DualPortBufferTb) {
  DualPortBuffer<Word, 16> dut;
  sc_clock clk;
  sc_signal<bool> rstn;
  Connections::Combinational<ac_int<16, false>> address[PORTS];
  Connections::Combinational<Word> response[PORTS];
  Connections::Combinational<BufferWriteRequest<Word>> writes[PORTS];
#if DOUBLE_BUFFERED_ACCUM_BUFFER
  Connections::SyncChannel done[PORTS];
#endif
#if ENABLE_PERF_COUNTERS
  sc_signal<MatrixPerformance::Counter> perf_reads[DOUBLE_BUFFERED_ACCUM_BUFFER ? 2 : 1];
  sc_signal<MatrixPerformance::Counter> perf_writes[DOUBLE_BUFFERED_ACCUM_BUFFER ? 2 : 1];
#endif
  int started = -1;
  int written = -1;
  int finished = -1;
  bool failed = false;

  // Connect both bank owners when double buffering is enabled
  SC_CTOR(DualPortBufferTb) : dut("dut"), clk("clk", 10, SC_NS) {
    dut.clk(clk);
    dut.rstn(rstn);
    for (int p = 0; p < PORTS; ++p) {
      dut.read_address[p](address[p]);
      dut.read_data[p](response[p]);
      dut.write_request[p](writes[p]);
#if DOUBLE_BUFFERED_ACCUM_BUFFER
      dut.done[p](done[p]);
#endif
    }
#if ENABLE_PERF_COUNTERS
    for (int b = 0; b < (DOUBLE_BUFFERED_ACCUM_BUFFER ? 2 : 1); ++b) {
      dut.perf_reads[b](perf_reads[b]);
      dut.perf_writes[b](perf_writes[b]);
    }
#endif
    SC_THREAD(run);
    sensitive << clk.posedge_event();
    SC_THREAD(issue_reads);
    sensitive << clk.posedge_event();
    SC_THREAD(consume);
    sensitive << clk.posedge_event();
    SC_THREAD(watchdog);
  }

  // Stop immediately on a failed progress or data assertion
  bool require(bool condition, const char* message) {
    if (!condition) {
      failed = true;
      std::cerr << "FAIL: " << message << std::endl;
      sc_stop();
    }
    return condition;
  }

  // Submit one independent SRAM write
  void write(int port, int index, int value) {
    BufferWriteRequest<Word> request;
    request.address = index;
    request.data = value;
    request.last = false;
    writes[port].Push(request);
  }

  // Exercise retained responses, subsequent reads, and bank reuse
  void run() {
    rstn = false;
    for (int p = 0; p < PORTS; ++p) {
      writes[p].ResetWrite();
#if DOUBLE_BUFFERED_ACCUM_BUFFER
      done[p].ResetWrite();
#endif
    }
    wait(5);
    rstn = true;
    wait(5);
    for (int c = 0; c < CASES; ++c) {
      const int p = CASE_PORTS[c];
      const int base = 100 * (c + 1);
      write(p, 9, base);
      wait(5);
      started = c;
      wait(20);  // Fill the native Connections stages and block read responses
      for (int i = 0; i < 9; ++i) write(p, i, base + 10 + i);
      written = c;
      while (finished < c) wait();
#if DOUBLE_BUFFERED_ACCUM_BUFFER
      done[p].SyncPush();
#endif
    }
#if ENABLE_PERF_COUNTERS && !DOUBLE_BUFFERED_ACCUM_BUFFER
    if (!require(perf_reads[0].read() == 26, "read counter mismatch")) return;
    if (!require(perf_writes[0].read() == 20, "write counter mismatch")) return;
#endif
    std::cout << "PASS: DualPortBuffer read backpressure, " << CASES
              << " bank-owner cases" << std::endl;
    sc_stop();
  }

  // Keep reads independent so a stalled request cannot stop the write stimulus
  void issue_reads() {
    for (int p = 0; p < PORTS; ++p) address[p].ResetWrite();
    for (int c = 0; c < CASES; ++c) {
      while (started < c) wait();
      const int p = CASE_PORTS[c];
      for (int i = 0; i < 4; ++i) address[p].Push(9);
      while (written < c) wait();
      for (int i = 0; i < 9; ++i) address[p].Push(i);
    }
  }

  // Hold the read port closed until all independent writes must have completed
  void consume() {
    for (int p = 0; p < PORTS; ++p) response[p].ResetRead();
    for (int c = 0; c < CASES; ++c) {
      while (started < c) wait();
      wait(80);
      if (!require(written == c, "blocked read response prevented writes"))
        return;
      const int p = CASE_PORTS[c];
      const int base = 100 * (c + 1);
      for (int i = 0; i < 4; ++i) {
        if (!require(response[p].Pop() == base,
                     "pending read response changed"))
          return;
      }
      for (int i = 0; i < 9; ++i) {
        if (!require(response[p].Pop() == base + 10 + i,
                     "write readback mismatch"))
          return;
      }
      finished = c;
    }
  }

  // Bound a stalled bank handoff or channel operation
  void watchdog() {
    wait(20000, SC_NS);
    require(false, "buffer test timed out");
  }
};

// Run the bounded functional regression without treating native timing as RTL
// timing
int sc_main(int argc, char* argv[]) {
  DualPortBufferTb test("test");
  sc_start();
  return test.failed ? 1 : 0;
}
