#include <mc_scverify.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "DualPortBuffer.h"

static_assert(DOUBLE_BUFFERED_ACCUM_BUFFER,
              "This test requires DOUBLE_BUFFERED_ACCUM_BUFFER=true");

SC_MODULE(DualPortBufferHandoffTb) {
  sc_in<bool> clk;
  sc_in<bool> rstn;
  using Word = ac_int<32, false>;
  DualPortBuffer<Word, 128> buffer;
  Connections::Combinational<ac_int<16, false>> address[4];
  Connections::Combinational<Word> data[4];
  Connections::Combinational<BufferWriteRequest<Word>> writes[4];
  Connections::SyncChannel done[4];
#if ENABLE_PERF_COUNTERS
  sc_signal<MatrixPerformance::Counter> reads_count[2], writes_count[2];
#endif
  int errors = 0;
  bool completed = false;

  SC_CTOR(DualPortBufferHandoffTb) : buffer("buffer") {
    buffer.clk(clk);
    buffer.rstn(rstn);
    for (int p = 0; p < 4; ++p) {
      buffer.read_address[p](address[p]);
      buffer.read_data[p](data[p]);
      buffer.write_request[p](writes[p]);
      buffer.done[p](done[p]);
    }
#if ENABLE_PERF_COUNTERS
    for (int bank = 0; bank < 2; ++bank) {
      buffer.perf_reads[bank](reads_count[bank]);
      buffer.perf_writes[bank](writes_count[bank]);
    }
#endif
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void run() {
    for (int p = 0; p < 4; ++p) {
      address[p].ResetWrite();
      data[p].ResetRead();
      writes[p].ResetWrite();
      done[p].ResetWrite();
    }
    wait();
    // A 7x14 output tile ends at address 97. Vary its phase relative to the
    // buffer's polling loop, and announce done immediately after the last Push.
    for (int delay = 0; delay < 128; ++delay) {
      for (int bank = 0; bank < 2; ++bank) {
        const int producer = 2 * bank;
        const int consumer = producer + 1;
        for (int i = 0; i < delay; ++i) wait();
        for (int i = 0; i < 98; ++i) {
          BufferWriteRequest<Word> request;
          request.address = i;
          request.data = 1000 + delay * 100 + i;
          writes[producer].Push(request);
        }
        done[producer].SyncPush();
        for (int i = 0; i < 98; ++i) {
          address[consumer].Push(i);
          const Word actual = data[consumer].Pop();
          const Word expected = 1000 + delay * 100 + i;
          if (actual != expected) {
            std::cout << "Mismatch bank=" << bank << " delay=" << delay
                      << " address=" << i << " actual=" << actual
                      << " expected=" << expected << std::endl;
            ++errors;
          }
        }
        done[consumer].SyncPush();
      }
    }
    completed = true;
    std::cout << "Bank handoff errors: " << errors << std::endl;
    sc_stop();
  }
};

int sc_main(int, char**) {
  sc_clock clk("clk", 5, SC_NS);
  sc_signal<bool> rstn;
  DualPortBufferHandoffTb tb("tb");
  tb.clk(clk);
  tb.rstn(rstn);
  rstn.write(false);
  sc_start(20, SC_NS);
  rstn.write(true);
  sc_start(5, SC_MS);
  if (!tb.completed) std::cerr << "Bank handoff test timed out" << std::endl;
  return !tb.completed || tb.errors ? 1 : 0;
}
