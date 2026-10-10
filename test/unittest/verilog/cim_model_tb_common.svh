// Common CIMIntMacroWrapper unit-test harness code
// Include this inside a generated test module after defining the localparams below:
// CASE_NAME, INPUT_LANES, OUTPUT_LANES, WEIGHT_SETS, A_WIDTH, B_WIDTH, C_WIDTH, WRITE_INPUT_LANES,
// MAC_LATENCY, INST_MODE, INST_IMPL, A_SIGNED, B_SIGNED_MASK, IS_SERIAL, NUM_ITERS,
// MCLK_PERIOD, WCLK_PERIOD, and TEST_KIND

import CIMExceptionPkg::*;

`include "cim_tb_helpers.svh"

localparam int unsigned TEST_NORMAL = 0;
localparam int unsigned TEST_EXCEPTION = 1;
localparam int unsigned ILLEGAL_SUITE_EXPECTED_EXCEPTION_COUNT = 3;
logic                  wclk;
logic                  mclk;
logic [A_WIDTH-1:0]    a [INPUT_LANES];
logic [B_WIDTH-1:0]    b [WRITE_INPUT_LANES][OUTPUT_LANES];
logic                  wen;
logic                  mac;
logic                  init;
logic                  a_signed;
logic                  b_signed [OUTPUT_LANES];
logic [INPUT_INDEX_WIDTH-1:0] write_input_index;
logic [SET_INDEX_WIDTH-1:0]   write_set;
logic [SET_INDEX_WIDTH-1:0]   compute_set;
logic [C_WIDTH-1:0]    c [OUTPUT_LANES];

// Reference state used by the self-checker. Rows start invalid and become
// usable only after the write scheduler has completed all write groups for that row
logic [B_WIDTH-1:0] model_weights [WEIGHT_SETS][INPUT_LANES][OUTPUT_LANES];
logic [C_WIDTH-1:0] expected [NUM_ITERS][OUTPUT_LANES];
int expected_row [NUM_ITERS];
bit row_valid [WEIGHT_SETS];
int row_mac_pending [WEIGHT_SETS];

// Event scheduler state for independent mclk/wclk periods. The two clocks may
// be equal, divisible, or relatively prime depending on the generated case
time tb_now;
time next_m_edge;
time next_w_edge;
int write_row;
int write_base;
int completed_row_updates;
int overlapped_write_mac_events;
int parallel_init_noise_events;
string waveform_path;

localparam int unsigned BITS_A_COUNT = (A_WIDTH <= 1) ? 1 : $clog2(A_WIDTH + 1);
localparam int unsigned C_VALID_DELAY = (MAC_LATENCY <= 1) ? 0 : MAC_LATENCY - 1;

logic [BITS_A_COUNT-1:0] checker_serial_bits_seen = '0;
logic checker_serial_result_input_valid;
logic checker_result_input_valid;
logic checker_result_valid = 1'b0;

CIMIntMacroWrapper #(
    .INPUT_LANES(INPUT_LANES),
    .OUTPUT_LANES(OUTPUT_LANES),
    .WEIGHT_SETS(WEIGHT_SETS),
    .A_WIDTH(A_WIDTH),
    .B_WIDTH(B_WIDTH),
    .C_WIDTH(C_WIDTH),
    .WRITE_INPUT_LANES(WRITE_INPUT_LANES),
    .MAC_LATENCY(MAC_LATENCY),
    .MODE(INST_MODE),
    .IMPL(INST_IMPL)
) dut (
    .wclk(wclk),
    .mclk(mclk),
    .a(a),
    .b(b),
    .wen(wen),
    .mac(mac),
    .init(init),
    .a_signed(a_signed),
    .b_signed(b_signed),
    .write_input_index(write_input_index),
    .write_set(write_set),
    .compute_set(compute_set),
    .c(c)
);

// Track the static result timing expected from the public macro wrapper protocol
always_comb begin
  checker_serial_result_input_valid = 1'b0;
  if (mac) begin
    if (init) begin
      checker_serial_result_input_valid = (A_WIDTH == 1);
    end else begin
      checker_serial_result_input_valid = (checker_serial_bits_seen == BITS_A_COUNT'(A_WIDTH - 1));
    end
  end
end

assign checker_result_input_valid = IS_SERIAL ? checker_serial_result_input_valid : mac;

always_ff @(posedge mclk) begin
  if (mac) begin
    if (init || (checker_serial_bits_seen == BITS_A_COUNT'(A_WIDTH))) begin
      checker_serial_bits_seen <= BITS_A_COUNT'(1);
    end else begin
      checker_serial_bits_seen <= checker_serial_bits_seen + BITS_A_COUNT'(1);
    end
  end
end

generate
  if (C_VALID_DELAY == 0) begin : gen_checker_valid_no_delay
    always_ff @(posedge mclk) begin
      checker_result_valid <= checker_result_input_valid;
    end
  end else begin : gen_checker_valid_pipe
    logic checker_result_valid_pipe [C_VALID_DELAY] = '{default: 1'b0};

    always_ff @(posedge mclk) begin
      checker_result_valid_pipe[0] <= checker_result_input_valid;
      for (int stage = 1; stage < C_VALID_DELAY; stage++) begin
        checker_result_valid_pipe[stage] <= checker_result_valid_pipe[stage-1];
      end
      checker_result_valid <= checker_result_valid_pipe[C_VALID_DELAY-1];
    end
  end
endgenerate

// Start VCD dumping when the runner supplies a waveform path
task automatic start_waveform_dump;
  begin
    if ($value$plusargs("waveform=%s", waveform_path)) begin
      $dumpfile(waveform_path);
      $dumpvars(0);
    end
  end
endtask

// Interpret an activation according to the generated A signedness
function automatic longint signed decode_a(input logic [A_WIDTH-1:0] value);
  logic signed [A_WIDTH-1:0] signed_value;
  begin
    signed_value = value;
    decode_a = A_SIGNED ? longint'(signed_value) : longint'(value);
  end
endfunction

// Interpret a stored weight according to this output channel's signedness
function automatic longint signed decode_b(input logic [B_WIDTH-1:0] value, input int output_index);
  logic signed [B_WIDTH-1:0] signed_value;
  begin
    signed_value = value;
    decode_b = B_SIGNED_MASK[output_index] ? longint'(signed_value) : longint'(value);
  end
endfunction

// Reset all driven signals and scoreboard state before loading weights
task automatic drive_defaults;
  begin
    wclk = 1'b0;
    mclk = 1'b0;
    wen = 1'b0;
    mac = 1'b0;
    init = 1'b0;
    a_signed = A_SIGNED;
    write_input_index = '0;
    write_set = '0;
    compute_set = '0;
    init_rng_from_plusarg();
    tb_now = 0;
    next_m_edge = 0;
    next_w_edge = 0;
    write_row = -1;
    write_base = 0;
    completed_row_updates = 0;
    overlapped_write_mac_events = 0;
    parallel_init_noise_events = 0;

    for (int row = 0; row < WEIGHT_SETS; row++) begin
      row_valid[row] = 1'b0;
      row_mac_pending[row] = 0;
    end
    for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
      b_signed[output_index] = B_SIGNED_MASK[output_index];
    end
    for (int input_index = 0; input_index < INPUT_LANES; input_index++) begin
      a[input_index] = '0;
    end
    for (int lane = 0; lane < WRITE_INPUT_LANES; lane++) begin
      for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
        b[lane][output_index] = '0;
      end
    end
    #1;
  end
endtask

// Reject generated parameters that would make the harness scheduler ambiguous
task automatic check_test_params;
  begin
    check_common_test_params();
  end
endtask

// Pulse mclk once during blocking setup phases such as init assertion
task automatic tick_mclk_blocking;
  begin
    #1 mclk = 1'b1;
    #1 mclk = 1'b0;
    #1;
  end
endtask

// Pulse wclk once during blocking setup phases such as collision injection
task automatic tick_wclk_blocking;
  begin
    #1 wclk = 1'b1;
    #1 wclk = 1'b0;
    #1;
  end
endtask

// Require one runtime DUT exception type collected by the illegal suite
task automatic require_illegal_suite_exception(input string exception_type, input string reason);
  begin
    if (!exception_type_seen(exception_type)) begin
      $fatal(1, "%s: expected DUT exception did not fire: %s (%s)",
             CASE_NAME, reason, exception_type);
    end
  end
endtask

// Verify exactly the runtime exception types this suite intentionally triggers
task automatic check_illegal_suite_exceptions;
  begin
    require_illegal_suite_exception(EXCEPTION_TYPE_SERIAL_INIT_WITHOUT_MAC,
                                    "serial init asserted without mac");
    require_illegal_suite_exception(EXCEPTION_TYPE_SERIAL_MAC_DROP,
                                    "serial mac deasserted before A_WIDTH bits");
    require_illegal_suite_exception(EXCEPTION_TYPE_ROW_WRITE_MAC_COLLISION,
                                    "write and MAC target the same row");

    if (collected_exception_count() != ILLEGAL_SUITE_EXPECTED_EXCEPTION_COUNT) begin
      $fatal(1, "%s: collected %0d DUT exception(s), expected %0d",
             CASE_NAME, collected_exception_count(), ILLEGAL_SUITE_EXPECTED_EXCEPTION_COUNT);
    end
    if (collected_unknown_exception_type_count() != 0) begin
      $fatal(1, "%s: %0d unknown DUT exception type(s) fired",
             CASE_NAME, collected_unknown_exception_type_count());
    end
  end
endtask

// Drive one WRITE_INPUT_LANES-wide write group of random B values for a row and base channel
task automatic drive_random_weight_group(input int row, input int base);
  int unsigned value;
  begin
    write_set = SET_INDEX_WIDTH'(row);
    write_input_index = INPUT_INDEX_WIDTH'(base);
    for (int lane = 0; lane < WRITE_INPUT_LANES; lane++) begin
      for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
        rng_next(value);
        b[lane][output_index] = B_WIDTH'(value);
      end
    end
    wen = 1'b1;
  end
endtask

// Mirror a completed DUT write group into the reference model
task automatic commit_weight_group(input int row, input int base);
  begin
    for (int lane = 0; lane < WRITE_INPUT_LANES; lane++) begin
      for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
        model_weights[row][base + lane][output_index] = b[lane][output_index];
      end
    end
  end
endtask

// Compute the expected output for the currently driven activation and row
task automatic record_expected(input int slot, input int row);
  longint signed acc;
  begin
    // The scoreboard is sized to NUM_ITERS, so launching beyond that is a harness bug
    if (slot >= NUM_ITERS) begin
      $fatal(1, "%s: expected slot %0d is outside NUM_ITERS=%0d", CASE_NAME, slot, NUM_ITERS);
    end

    expected_row[slot] = row;

    for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
      acc = 0;
      for (int input_index = 0; input_index < INPUT_LANES; input_index++) begin
        acc += decode_a(a[input_index]) * decode_b(model_weights[row][input_index][output_index], output_index);
      end
      expected[slot][output_index] = C_WIDTH'(acc);
    end
  end
endtask

// Choose a row whose weights are fully valid for the next MAC. Pending MACs do
// not block more MACs to the same row; they only block background writes
function automatic int choose_mac_row;
  static int next_mac_row;
  int candidate;
  begin
    choose_mac_row = -1;
    for (int attempt = 0; attempt < WEIGHT_SETS; attempt++) begin
      candidate = (next_mac_row + attempt) % WEIGHT_SETS;
      // A row may be used by MAC only after all write groups of its latest write completed
      if (row_valid[candidate]) begin
        // Advance the round-robin cursor only after this call claims a usable row
        next_mac_row = (candidate + 1) % WEIGHT_SETS;
        return candidate;
      end
    end
  end
endfunction

// Start a background row write when possible. This may initialize a previously
// invalid row or refresh a valid row, but never while a MAC still depends on it
task automatic maybe_start_write(input int avoid_row);
  static int next_write_row;
  int candidate;
  begin
    // A non-negative write_row means a row write is already in progress
    if (write_row >= 0) begin
      return;
    end

    for (int attempt = 0; attempt < WEIGHT_SETS; attempt++) begin
      candidate = (next_write_row + attempt) % WEIGHT_SETS;
      // The writer can take any row that is not the current MAC row and has no
      // in-flight MAC result depending on its old weights
      if ((candidate != avoid_row) && (row_mac_pending[candidate] == 0)) begin
        write_row = candidate;
        write_base = 0;
        row_valid[candidate] = 1'b0;
        next_write_row = (candidate + 1) % WEIGHT_SETS;
        return;
      end
    end
  end
endtask

// Drive the write port for the next wclk edge. This prepares one WRITE_INPUT_LANES
// write group; finish_write_for_event commits it to the reference model after the edge
task automatic setup_write_for_event(input int avoid_row);
  begin
    wen = 1'b0;

    maybe_start_write(avoid_row);
    // If a safe background row is reserved, drive the next write group for this wclk edge
    if ((write_row >= 0) && (write_row != avoid_row)) begin
      drive_random_weight_group(write_row, write_base);
    end
  end
endtask

// Complete the bookkeeping for a prepared wclk edge and mark the row valid once
// all write groups have been written
task automatic finish_write_for_event;
  begin
    // Only commit to the reference model when setup_write_for_event drove wen
    if (wen) begin
      commit_weight_group(write_row, write_base);
      write_base += WRITE_INPUT_LANES;
      // Revalidate the row once the final channel group has reached both models
      if (write_base >= INPUT_LANES) begin
        row_valid[write_row] = 1'b1;
        write_row = -1;
        write_base = 0;
        completed_row_updates++;
      end
    end
    wen = 1'b0;
  end
endtask

// Initialize the event scheduler from the current simulation time
task automatic init_scheduler;
  begin
    tb_now = $time;
    next_m_edge = tb_now + time'(MCLK_PERIOD);
    next_w_edge = tb_now + time'(WCLK_PERIOD);
  end
endtask

// Find the next scheduled clock edge. If both clocks land on the same time, the
// event raises both edges together
task automatic peek_next_event(output bit m_edge, output bit w_edge, output time event_time);
  begin
    event_time = (next_m_edge <= next_w_edge) ? next_m_edge : next_w_edge;
    m_edge = (next_m_edge == event_time);
    w_edge = (next_w_edge == event_time);
  end
endtask

// Advance simulation time, pulse the selected clock edge(s), check any MAC
// result, then schedule the next edge for each clock that fired
task automatic finish_prepared_event(
    input bit m_edge,
    input bit w_edge,
    input time event_time,
    input int launched,
    inout int seen,
    inout int event_count
);
  begin
    // Skip zero-delay waits when the next event is already at the tracked time
    if (event_time > tb_now) begin
      #(event_time - tb_now);
      tb_now = event_time;
    end

    // Raise mclk only for events whose next edge is the current event time
    if (m_edge) begin
      mclk = 1'b1;
    end
    // Raise wclk only for events whose next edge is the current event time
    if (w_edge) begin
      wclk = 1'b1;
    end

    // Leave the edge high for one tick so sequential updates and checker timing settle
    // before the checker samples outputs
    #1;
    tb_now = event_time + 1;

    // Count coverage when a write edge occurs while any launched MAC has not
    // produced its result yet. This includes pipeline latency after mac is low
    if (w_edge && wen && (launched > seen)) begin
      overlapped_write_mac_events++;
    end

    // A prepared write is considered complete after the wclk edge has fired
    if (w_edge) begin
      finish_write_for_event();
    end

    // MAC outputs are sampled only on mclk events, matching the DUT output domain
    if (m_edge) begin
      event_count++;
      // Static result timing marks when the next expected result must match the DUT output bus
      if (checker_result_valid) begin
        // Seeing more valid outputs than launched MACs indicates bogus result timing
        if (seen >= launched) begin
          $fatal(1, "%s: unexpected extra result at event %0d", CASE_NAME, event_count);
        end

        for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
          // Each output channel is checked independently against the scoreboard slot
          if (c[output_index] !== expected[seen][output_index]) begin
            $fatal(1,
                   "%s: op %0d channel %0d event %0d got 0x%0h expected 0x%0h",
                   CASE_NAME, seen, output_index, event_count, c[output_index], expected[seen][output_index]);
          end
        end
        // The result leaving the pipeline releases its row for future writes
        if (row_mac_pending[expected_row[seen]] <= 0) begin
          $fatal(1, "%s: row %0d has no pending MAC result to release",
                 CASE_NAME, expected_row[seen]);
        end
        row_mac_pending[expected_row[seen]]--;
        seen++;
      end
    end

    // Lower mclk and schedule its next rising edge after an mclk event
    if (m_edge) begin
      mclk = 1'b0;
      next_m_edge += time'(MCLK_PERIOD);
    end
    // Lower wclk and schedule its next rising edge after a wclk event
    if (w_edge) begin
      wclk = 1'b0;
      next_w_edge += time'(WCLK_PERIOD);
    end
  end
endtask

// Consume scheduler events without launching a MAC; background writes may still
// progress on wclk edges
task automatic run_idle_event(input int launched, inout int seen, inout int event_count);
  bit m_edge;
  bit w_edge;
  time event_time;
  begin
    peek_next_event(m_edge, w_edge, event_time);
    mac = 1'b0;
    init = 1'b0;
    // Even while MAC is idle, a wclk event can advance a safe background rewrite
    if (w_edge) begin
      setup_write_for_event(-1);
    end else begin
      wen = 1'b0;
    end
    finish_prepared_event(m_edge, w_edge, event_time, launched, seen, event_count);
  end
endtask

// Bit-parallel mode can launch a new MAC on every mclk edge when a valid row is
// available. Background writes continue on safe rows in parallel
task automatic run_parallel;
  int launched;
  int seen;
  int event_count;
  int mac_row;
  int unsigned noise_value;
  bit m_edge;
  bit w_edge;
  time event_time;
  begin
    launched = 0;
    seen = 0;
    event_count = 0;
    // Initialize the next event time
    init_scheduler();

    while ((launched < NUM_ITERS) || (seen < launched)) begin
      // Find what is the next event
      peek_next_event(m_edge, w_edge, event_time);
      mac = 1'b0;
      wen = 1'b0;
      mac_row = -1;
      rng_next(noise_value);
      init = noise_value[0] || (parallel_init_noise_events == 0);
      if (init) begin
        parallel_init_noise_events++;
      end

      // A parallel MAC can be launched only on mclk and only while work remains
      if (m_edge && (launched < NUM_ITERS)) begin
        mac_row = choose_mac_row();
        // If all rows are being rewritten, skip this mclk edge rather than using stale data
        if (mac_row >= 0) begin
          randomize_activation();
          compute_set = SET_INDEX_WIDTH'(mac_row);
          mac = 1'b1;
          record_expected(launched, mac_row);
          row_mac_pending[mac_row]++;
          launched++;
        end
      end

      // A wclk event may rewrite a row, but setup_write_for_event avoids mac_row
      if (w_edge) begin
        setup_write_for_event(mac_row);
      end

      finish_prepared_event(m_edge, w_edge, event_time, launched, seen, event_count);
    end
  end
endtask

// Drive illegal structural and protocol coverage in one collected failure run
task automatic run_illegal_suite;
  begin
    check_test_params();
    #2;

    mac = 1'b0;
    init = 1'b1;
    tick_mclk_blocking();
    init = 1'b0;

    randomize_activation();
    compute_set = '0;
    mac = 1'b1;
    init = 1'b1;
    tick_mclk_blocking();
    init = 1'b0;
    mac = 1'b0;
    tick_mclk_blocking();

    randomize_activation();
    drive_random_weight_group(0, 0);
    compute_set = '0;
    mac = 1'b1;
    init = 1'b0;
    tick_wclk_blocking();
    wen = 1'b0;
    mac = 1'b0;

    check_illegal_suite_exceptions();
    $display("[pass] %s illegal suite captured expected DUT exception types", CASE_NAME);
    $finish;
  end
endtask

// Bit-serial mode holds one activation and row for A_WIDTH mclk edges, while
// still allowing safe background writes between and during those cycles
task automatic run_serial;
  int launched;
  int seen;
  int event_count;
  int mac_row;
  int m_edges_seen;
  bit m_edge;
  bit w_edge;
  time event_time;
  begin
    launched = 0;
    seen = 0;
    event_count = 0;
    init_scheduler();

    for (int op = 0; op < NUM_ITERS; op++) begin
      mac_row = choose_mac_row();
      while (mac_row < 0) begin
        run_idle_event(launched, seen, event_count);
        mac_row = choose_mac_row();
      end

      randomize_activation();
      compute_set = SET_INDEX_WIDTH'(mac_row);
      record_expected(launched, mac_row);
      row_mac_pending[mac_row]++;

      launched++;
      m_edges_seen = 0;
      while (m_edges_seen < A_WIDTH) begin
        peek_next_event(m_edge, w_edge, event_time);
        mac = 1'b1;
        init = (m_edges_seen == 0);
        // While serial MAC is active, background writes must avoid the active MAC row
        if (w_edge) begin
          setup_write_for_event(mac_row);
        end else begin
          wen = 1'b0;
        end
        finish_prepared_event(m_edge, w_edge, event_time, launched, seen, event_count);
        // Count only mclk edges toward the A_WIDTH bit-serial MAC duration
        if (m_edge) begin
          m_edges_seen++;
        end
      end

      mac = 1'b0;
      while (seen < launched) begin
        peek_next_event(m_edge, w_edge, event_time);
        mac = 1'b0;
        init = 1'b0;
        // While draining the MAC pipeline, safe background writes can continue
        if (w_edge) begin
          setup_write_for_event(mac_row);
        end else begin
          wen = 1'b0;
        end
        finish_prepared_event(m_edge, w_edge, event_time, launched, seen, event_count);
      end
    end
  end
endtask

// Multi-row cases are expected to exercise the intended stress path: at least
// one complete row rewrite and at least one write edge while MAC is active
task automatic check_stress_coverage;
  begin
    // Parallel cases intentionally toggle init as ignored noise
    if (!IS_SERIAL && (parallel_init_noise_events == 0)) begin
      $fatal(1, "%s: no parallel init noise was injected", CASE_NAME);
    end
    // Single-row cases allow writes only when no MAC result is pending, so they
    // do not need the multi-row overlap coverage requirement
    if (WEIGHT_SETS > 1) begin
      // A passing multi-row test should have completed at least one full row update
      if (completed_row_updates == 0) begin
        $fatal(1, "%s: no background row updates completed", CASE_NAME);
      end
      // A passing multi-row test should include true write/MAC overlap
      if (overlapped_write_mac_events == 0) begin
        $fatal(1, "%s: no weight write occurred while mac was active", CASE_NAME);
      end
    end
  end
endtask

initial begin
  start_waveform_dump();
  drive_defaults();
  if (TEST_KIND == TEST_EXCEPTION) begin
    run_illegal_suite();
  end else begin
    check_test_params();
    // Dispatch to the mode-specific driver because serial and parallel launch rules differ
    if (IS_SERIAL) begin
      run_serial();
    end else begin
      run_parallel();
    end

    check_stress_coverage();
    $display("[pass] %s", CASE_NAME);
    $finish;
  end
end
