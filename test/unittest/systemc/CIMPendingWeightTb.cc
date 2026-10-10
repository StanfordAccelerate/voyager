// A MAC stalled by full result storage must keep its weights until issue.
#include <systemc.h>

#include <iostream>
#include <vector>

#include "cim/CIMArray.h"

static int remaining = 2;

template <int Mode>
SC_MODULE(PendingWeightCase) {
  using Dut = CIMArray<4, 2, 2, 8, 8, 20, 1, 3, Mode, 8, 8, 24, true, 1, 1, 1,
                       1, 1, 1, 1, CIM_C_BEAT_OUTPUT_MAJOR, 1>;
  using MAC = typename Dut::MACRequest;
  using Write = typename Dut::WriteRequest;
  using Result = typename Dut::CBeat;
  sc_clock clk{"clk", 1, SC_NS};
  sc_signal<bool> rstn{"rstn"};
  Dut dut{"dut"};
  Connections::Combinational<MAC, Connections::SYN_PORT> mac_channel;
  Connections::Combinational<Write, Connections::SYN_PORT> write_channel;
  Connections::Combinational<Result, Connections::SYN_PORT> result_channel;
  Connections::Out<MAC, Connections::SYN_PORT> mac{"mac"};
  Connections::Out<Write, Connections::SYN_PORT> write{"write"};
  Connections::In<Result, Connections::SYN_PORT> result{"result"};
  unsigned macs = 0, writes = 0;
  std::vector<Result> results;

  SC_CTOR(PendingWeightCase) {
    dut.clk(clk);
    dut.rstn(rstn);
    dut.mac_request_channel(mac_channel);
    dut.write_request_channel(write_channel);
    dut.result_channel(result_channel);
    mac(mac_channel);
    write(write_channel);
    result(result_channel);
    SC_METHOD(observe);
    sensitive << clk.posedge_event();
    dont_initialize();
    SC_THREAD(run);
    sensitive << clk.posedge_event();
    SC_THREAD(watchdog);
  }
  void check(bool ok, const char* message) {
    if (!ok) SC_REPORT_FATAL(name(), message);
  }
  void settle() {
    for (int i = 0; i < 4; ++i) wait(SC_ZERO_TIME);
  }
  void tick() {
    wait(clk.posedge_event());
    settle();
  }
  void observe() {
    if (!rstn.read()) return;
    if (mac.vld.read() && mac.rdy.read()) ++macs;
    if (write.vld.read() && write.rdy.read()) ++writes;
    if (result.vld.read() && result.rdy.read())
      results.push_back(BitsToType<Result>(result.dat.read()));
  }
  Write weight(unsigned set, unsigned row, int first, int second) {
    Write req{};
    req.write_set = set;
    req.write_input_index = row;
    req.data[0][0][0] = first;
    req.data[0][0][1] = second;
    return req;
  }
  void send_write(const Write& req) {
    const unsigned before = writes;
    write.dat.write(TypeToBits(req));
    write.vld.write(true);
    settle();
    do {
      tick();
    } while (writes == before);
    write.vld.write(false);
    settle();
  }
  MAC operation() {
    MAC req{};
    req.compute_set = 0;
    req.multicast = 1;
    req.reduce = 1;
    for (int i = 0; i < 4; ++i) req.a[0][i] = 1;
    return req;
  }
  void send_mac() {
    const unsigned before = macs;
    mac.dat.write(TypeToBits(operation()));
    mac.vld.write(true);
    settle();
    do {
      tick();
    } while (macs == before);
    mac.vld.write(false);
    settle();
  }
  void run() {
    rstn.write(false);
    mac.Reset();
    write.Reset();
    result.Reset();
    settle();
    tick();
    tick();
    rstn.write(true);
    tick();
    for (unsigned row = 0; row < 4; ++row)
      send_write(weight(0, row, row + 1, 2 * (row + 1)));

    // Occupy the only result slot, then leave the next request presented.
    send_mac();
    while (!result.vld.read()) tick();
    mac.dat.write(TypeToBits(operation()));
    mac.vld.write(true);
    settle();
    check(!mac.rdy.read(), "MAC must wait while its result slot is full");

    // An unrelated set can still be programmed during this stall.
    send_write(weight(1, 0, 77, 88));
    const unsigned before_write = writes;
    write.dat.write(TypeToBits(weight(0, 0, 99, 99)));
    write.vld.write(true);
    settle();
    check(!write.rdy.read(),
          "stalled MAC allowed its weight set to be overwritten");
    for (int i = 0; i < 8; ++i) tick();
    check(writes == before_write && macs == 1,
          "a blocked request or same-set write was accepted");

    // Draining C lets the pending MAC consume the old weights, then the write.
    result.rdy.write(true);
    settle();
    while (macs < 2) tick();
    mac.vld.write(false);
    settle();
    while (writes == before_write) tick();
    write.vld.write(false);
    settle();
    while (results.size() < 2) tick();
    for (const auto& value : results)
      check(value[0][0] == 10 && value[0][1] == 20,
            "pending MAC used overwritten weights");
    send_mac();
    while (results.size() < 3) tick();
    check(results.back()[0][0] == 108 && results.back()[0][1] == 117,
          "the deferred weight write did not become visible");
    std::cout << "PASS pending weight protection mode=" << Mode << '\n';
    if (--remaining == 0) sc_stop();
  }
  void watchdog() {
    wait(1000, SC_NS);
    check(false, "test timed out");
  }
};

int sc_main(int, char**) {
  PendingWeightCase<0> parallel("parallel");
  PendingWeightCase<1> serial("serial");
  sc_start();
  return 0;
}
