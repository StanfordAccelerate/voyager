// Common CIMIntElement unit-test harness code
// Include this inside a generated test module after defining the localparams below:
// CASE_NAME, INPUT_LANES, OUTPUT_LANES, WRITE_INPUT_LANES, WEIGHT_SETS, BASE_A_WIDTH, BASE_B_WIDTH, BASE_C_WIDTH,
// MAC_LATENCY, INST_MODE, INST_IMPL, SIGNED, A_WIDTH, B_WIDTH, NUM_ITERS,
// MCLK_PERIOD, WCLK_PERIOD, EXPECT_DROPPED_ISSUE, and TEST_KIND

localparam int unsigned WEIGHT_SLICES = B_WIDTH / BASE_B_WIDTH;
localparam int unsigned MACRO_INPUT_LANES = INPUT_LANES;
localparam int unsigned MACRO_OUTPUT_LANES = OUTPUT_LANES * WEIGHT_SLICES;
localparam int unsigned MACRO_WRITE_INPUT_LANES = WRITE_INPUT_LANES;
localparam int unsigned SUM_GUARD_WIDTH = (INPUT_LANES <= 1) ? 1 : $clog2(INPUT_LANES);
localparam int unsigned C_WIDTH = A_WIDTH + B_WIDTH + SUM_GUARD_WIDTH;
localparam int unsigned INPUT_INDEX_WIDTH = (INPUT_LANES <= 1) ? 1 : $clog2(INPUT_LANES);
localparam int unsigned SET_INDEX_WIDTH = (WEIGHT_SETS <= 1) ? 1 : $clog2(WEIGHT_SETS);
localparam int unsigned DEFAULT_RNG_SEED = 32'h1;

localparam int unsigned TEST_NORMAL = 0;
localparam int unsigned TEST_RESET_MID_OP = 1;
localparam int unsigned MAX_WAIT_CYCLES = 4096;

logic                       wclk;
logic                       mclk;
logic                       rstn;
logic [A_WIDTH-1:0]         a [INPUT_LANES];
logic [B_WIDTH-1:0]         b [WRITE_INPUT_LANES][OUTPUT_LANES];
logic                       wen;
logic [INPUT_INDEX_WIDTH-1:0]          write_input_index;
logic [SET_INDEX_WIDTH-1:0]        write_set;
logic                       mac_issue;
logic [SET_INDEX_WIDTH-1:0]        compute_set;
logic [C_WIDTH-1:0]         c [OUTPUT_LANES];
logic                       c_retire;
logic                       mac_ready;
logic                       mac_busy;

logic [B_WIDTH-1:0] model_b [WEIGHT_SETS][INPUT_LANES][OUTPUT_LANES];
logic [C_WIDTH-1:0] expected [NUM_ITERS][OUTPUT_LANES];
int unsigned dropped_issue_attempts;
int unsigned rng_state;
string waveform_path;

CIMIntElement #(
    .MACRO_INPUT_LANES(MACRO_INPUT_LANES),
    .MACRO_OUTPUT_LANES(MACRO_OUTPUT_LANES),
    .WEIGHT_SETS(WEIGHT_SETS),
    .BASE_A_WIDTH(BASE_A_WIDTH),
    .BASE_B_WIDTH(BASE_B_WIDTH),
    .BASE_C_WIDTH(BASE_C_WIDTH),
    .MACRO_WRITE_INPUT_LANES(MACRO_WRITE_INPUT_LANES),
    .MAC_LATENCY(MAC_LATENCY),
    .MODE(INST_MODE),
    .MACRO_IMPL(INST_IMPL),
    .A_WIDTH(A_WIDTH),
    .B_WIDTH(B_WIDTH),
    .SIGNED(SIGNED)
) dut (
    .wclk(wclk),
    .mclk(mclk),
    .rstn(rstn),
    .a(a),
    .b(b),
    .wen(wen),
    .write_input_index(write_input_index),
    .write_set(write_set),
    .mac_issue(mac_issue),
    .compute_set(compute_set),
    .c(c),
    .c_retire(c_retire),
    .mac_ready(mac_ready),
    .mac_busy(mac_busy)
);

// Start VCD dumping when the runner supplies a waveform path
task automatic start_waveform_dump;
  begin
    if ($value$plusargs("waveform=%s", waveform_path)) begin
      $dumpfile(waveform_path);
      $dumpvars(0);
    end
  end
endtask

// Load the deterministic pseudo-random generator seed from plusargs
task automatic init_rng_from_plusarg;
  int unsigned plusarg_rng_seed;
  begin
    if ($value$plusargs("rng_seed=%h", plusarg_rng_seed)) begin
      rng_state = plusarg_rng_seed;
    end else begin
      rng_state = DEFAULT_RNG_SEED;
    end
  end
endtask

// Advance the deterministic pseudo-random generator used by test stimuli
task automatic rng_next(output int unsigned value);
  begin
    rng_state = (rng_state * 32'd1664525) + 32'd1013904223;
    value = rng_state;
  end
endtask

// Reject common generated parameters that would make scheduling ambiguous
task automatic check_common_test_params;
  begin
    if ((MCLK_PERIOD <= 1) || (WCLK_PERIOD <= 1)) begin
      $fatal(1, "%s: clock periods must be greater than one tick", CASE_NAME);
    end
    if (NUM_ITERS == 0) begin
      $fatal(1, "%s: NUM_ITERS must be positive", CASE_NAME);
    end
  end
endtask

// Generate one random A vector on the a interface
task automatic randomize_a;
  int unsigned value;
  begin
    for (int input_index = 0; input_index < INPUT_LANES; input_index++) begin
      rng_next(value);
      a[input_index] = A_WIDTH'(value);
    end
  end
endtask

// Interpret an A according to the generated element signedness
function automatic longint signed decode_a(input logic [A_WIDTH-1:0] value);
  logic signed [A_WIDTH-1:0] signed_value;
  begin
    signed_value = value;
    decode_a = SIGNED ? longint'(signed_value) : longint'(value);
  end
endfunction

// Interpret a stored B according to the generated element signedness
function automatic longint signed decode_b(input logic [B_WIDTH-1:0] value);
  logic signed [B_WIDTH-1:0] signed_value;
  begin
    signed_value = value;
    decode_b = SIGNED ? longint'(signed_value) : longint'(value);
  end
endfunction

// Pulse the element mclk once
task automatic tick_mclk;
  begin
    #(MCLK_PERIOD) mclk = 1'b1;
    #1 mclk = 1'b0;
    #1;
  end
endtask

// Pulse the element wclk once
task automatic tick_wclk;
  begin
    #(WCLK_PERIOD) wclk = 1'b1;
    #1 wclk = 1'b0;
    #1;
  end
endtask

// Reset all driven signals and scoreboard state before applying DUT reset
task automatic drive_defaults;
  begin
    wclk = 1'b0;
    mclk = 1'b0;
    rstn = 1'b0;
    wen = 1'b0;
    write_input_index = '0;
    write_set = '0;
    mac_issue = 1'b0;
    compute_set = '0;
    dropped_issue_attempts = 0;

    init_rng_from_plusarg();

    for (int input_index = 0; input_index < INPUT_LANES; input_index++) begin
      a[input_index] = '0;
    end
    for (int write_input_offset = 0; write_input_offset < WRITE_INPUT_LANES; write_input_offset++) begin
      for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
        b[write_input_offset][output_index] = '0;
      end
    end
    for (int set_idx = 0; set_idx < WEIGHT_SETS; set_idx++) begin
      for (int input_index = 0; input_index < INPUT_LANES; input_index++) begin
        for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
          model_b[set_idx][input_index][output_index] = '0;
        end
      end
    end
    for (int slot = 0; slot < NUM_ITERS; slot++) begin
      for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
        expected[slot][output_index] = '0;
      end
    end
    #1;
  end
endtask

// Reject generated parameters that would make the element harness ambiguous
task automatic check_test_params;
  begin
    check_common_test_params();
    if ((INPUT_LANES % WRITE_INPUT_LANES) != 0) begin
      $fatal(1, "%s: INPUT_LANES must be divisible by WRITE_INPUT_LANES", CASE_NAME);
    end
    if ((B_WIDTH % BASE_B_WIDTH) != 0) begin
      $fatal(1, "%s: B_WIDTH must be divisible by BASE_B_WIDTH", CASE_NAME);
    end
    if ((MACRO_OUTPUT_LANES % WEIGHT_SLICES) != 0) begin
      $fatal(1, "%s: MACRO_OUTPUT_LANES must be divisible by WEIGHT_SLICES", CASE_NAME);
    end
    if (TEST_KIND > TEST_RESET_MID_OP) begin
      $fatal(1, "%s: unknown TEST_KIND=%0d", CASE_NAME, TEST_KIND);
    end
  end
endtask

// Apply element reset and check the reset-state protocol
task automatic apply_reset;
  begin
    rstn = 1'b0;
    mac_issue = 1'b0;
    wen = 1'b0;
    tick_mclk();
    tick_mclk();
    if (mac_ready !== 1'b0) begin
      $fatal(1, "%s: mac_ready must be low during reset", CASE_NAME);
    end
    if (mac_busy !== 1'b0) begin
      $fatal(1, "%s: mac_busy must be low during reset", CASE_NAME);
    end
    if (c_retire !== 1'b0) begin
      $fatal(1, "%s: c_retire must be low during reset", CASE_NAME);
    end

    rstn = 1'b1;
    #1;
    if (mac_ready !== 1'b1) begin
      $fatal(1, "%s: mac_ready must be high after reset release", CASE_NAME);
    end
    if (mac_busy !== 1'b0) begin
      $fatal(1, "%s: mac_busy must be low after reset release", CASE_NAME);
    end
    if (c_retire !== 1'b0) begin
      $fatal(1, "%s: c_retire must stay low after reset release", CASE_NAME);
    end
  end
endtask

// Drive one WRITE_INPUT_LANES-wide B block for a set and base INPUT_LANES index
task automatic drive_random_b_group(input int set_idx, input int base_input_index);
  int unsigned value;
  begin
    write_set = SET_INDEX_WIDTH'(set_idx);
    write_input_index = INPUT_INDEX_WIDTH'(base_input_index);
    for (int write_input_offset = 0; write_input_offset < WRITE_INPUT_LANES; write_input_offset++) begin
      for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
        rng_next(value);
        b[write_input_offset][output_index] = B_WIDTH'(value);
      end
    end
    wen = 1'b1;
  end
endtask

// Mirror a completed DUT write into the logical reference model
task automatic commit_b_group(input int set_idx, input int base_input_index);
  begin
    for (int write_input_offset = 0; write_input_offset < WRITE_INPUT_LANES; write_input_offset++) begin
      for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
        model_b[set_idx][base_input_index + write_input_offset][output_index] = b[write_input_offset][output_index];
      end
    end
  end
endtask

// Load every logical B set before MAC checks begin
task automatic load_all_b_sets;
  begin
    for (int set_idx = 0; set_idx < WEIGHT_SETS; set_idx++) begin
      for (int base_input_index = 0; base_input_index < INPUT_LANES; base_input_index += WRITE_INPUT_LANES) begin
        drive_random_b_group(set_idx, base_input_index);
        tick_wclk();
        commit_b_group(set_idx, base_input_index);
        wen = 1'b0;
      end
    end
  end
endtask

// Compute the expected logical output for the currently driven A and set
task automatic record_expected(input int slot, input int set_idx);
  longint signed acc;
  begin
    if (slot >= NUM_ITERS) begin
      $fatal(1, "%s: expected slot %0d is outside NUM_ITERS=%0d", CASE_NAME, slot, NUM_ITERS);
    end

    for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
      acc = 0;
      for (int input_index = 0; input_index < INPUT_LANES; input_index++) begin
        acc += decode_a(a[input_index]) * decode_b(model_b[set_idx][input_index][output_index]);
      end
      expected[slot][output_index] = C_WIDTH'(acc);
    end
  end
endtask

// Issue one logical element MAC operation while the element is ready
task automatic start_element_op(input int slot, input int set_idx);
  begin
    if (mac_ready !== 1'b1) begin
      $fatal(1, "%s: attempted to issue op %0d while mac_ready is low", CASE_NAME, slot);
    end

    randomize_a();
    compute_set = SET_INDEX_WIDTH'(set_idx);
    record_expected(slot, set_idx);

    mac_issue = 1'b1;
    #1;
    if (mac_busy !== 1'b1) begin
      $fatal(1, "%s: mac_busy must cover the issue edge for op %0d", CASE_NAME, slot);
    end
    tick_mclk();
    mac_issue = 1'b0;
    #1;
  end
endtask

// Optionally assert ignored mac_issue noise during current operand use
task automatic drive_dropped_issue_noise;
  begin
    if (EXPECT_DROPPED_ISSUE && (mac_busy === 1'b1)) begin
      mac_issue = 1'b1;
      dropped_issue_attempts++;
    end else begin
      mac_issue = 1'b0;
    end
  end
endtask

// Wait for one retirement and compare every column in OUTPUT_LANES
task automatic wait_for_retire_and_check(input int slot);
  int unsigned wait_cycles;
  begin
    wait_cycles = 0;
    while (c_retire !== 1'b1) begin
      if (wait_cycles >= MAX_WAIT_CYCLES) begin
        $fatal(1, "%s: timed out waiting for c_retire on op %0d", CASE_NAME, slot);
      end
      drive_dropped_issue_noise();
      tick_mclk();
      mac_issue = 1'b0;
      #1;
      wait_cycles++;
    end
    for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
      if (c[output_index] !== expected[slot][output_index]) begin
        $fatal(1,
               "%s: op %0d column output_index=%0d got 0x%0h expected 0x%0h",
               CASE_NAME, slot, output_index, c[output_index], expected[slot][output_index]);
      end
    end
    if (mac_ready !== 1'b1) begin
      $fatal(1, "%s: mac_ready must be high once op %0d has retired", CASE_NAME, slot);
    end
    if (mac_busy !== 1'b0) begin
      $fatal(1, "%s: mac_busy must be low once op %0d has retired", CASE_NAME, slot);
    end
  end
endtask

// Check that retired outputs stay stable until the next retirement
task automatic check_result_hold(input int slot);
  logic [C_WIDTH-1:0] held_c [OUTPUT_LANES];
  begin
    for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
      held_c[output_index] = c[output_index];
    end
    mac_issue = 1'b0;
    tick_mclk();
    if (c_retire !== 1'b0) begin
      $fatal(1, "%s: c_retire pulsed without a new op after slot %0d", CASE_NAME, slot);
    end
    for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
      if (c[output_index] !== held_c[output_index]) begin
        $fatal(1, "%s: c changed without a new retirement after slot %0d column output_index=%0d",
               CASE_NAME, slot, output_index);
      end
    end
  end
endtask

// Run one complete logical operation and stable-output check
task automatic run_one_op(input int slot, input int set_idx);
  begin
    start_element_op(slot, set_idx);
    wait_for_retire_and_check(slot);
    check_result_hold(slot);
  end
endtask

// Run the transactional legal-operation sequence for this case
task automatic run_normal_ops;
  begin
    dropped_issue_attempts = 0;
    for (int op = 0; op < NUM_ITERS; op++) begin
      run_one_op(op, op % WEIGHT_SETS);
    end
    if (EXPECT_DROPPED_ISSUE && (dropped_issue_attempts == 0)) begin
      $fatal(1, "%s: no dropped mac_issue noise was injected", CASE_NAME);
    end
  end
endtask

// Issue ops back to back at the ready cadence and check retirements in order
task automatic run_pipelined_ops;
  int unsigned issued;
  int unsigned retired;
  int unsigned wait_cycles;
  bit issue_pending;
  bit capture_next;
  begin
    issued = 0;
    retired = 0;
    wait_cycles = 0;
    issue_pending = 1'b0;
    while (retired < NUM_ITERS) begin
      mac_issue = issue_pending;
      #1;
      capture_next = (issued < NUM_ITERS) && (mac_ready === 1'b1);
      tick_mclk();
      mac_issue = 1'b0;

      if (c_retire === 1'b1) begin
        for (int output_index = 0; output_index < OUTPUT_LANES; output_index++) begin
          if (c[output_index] !== expected[retired][output_index]) begin
            $fatal(1,
                   "%s: pipelined op %0d column output_index=%0d got 0x%0h expected 0x%0h",
                   CASE_NAME, retired, output_index, c[output_index], expected[retired][output_index]);
          end
        end
        retired++;
        wait_cycles = 0;
      end

      if (capture_next) begin
        randomize_a();
        compute_set = SET_INDEX_WIDTH'(issued % WEIGHT_SETS);
        record_expected(issued, issued % WEIGHT_SETS);
        issued++;
        issue_pending = 1'b1;
      end else begin
        issue_pending = 1'b0;
      end

      wait_cycles++;
      if (wait_cycles >= MAX_WAIT_CYCLES) begin
        $fatal(1, "%s: pipelined run stalled with issued=%0d retired=%0d", CASE_NAME, issued, retired);
      end
    end
  end
endtask

// Issue an operation and reset before it can retire
task automatic reset_during_active_op;
  begin
    if (mac_ready !== 1'b1) begin
      $fatal(1, "%s: reset-mid-op precondition failed because mac_ready is low", CASE_NAME);
    end

    randomize_a();
    compute_set = '0;
    mac_issue = 1'b1;
    tick_mclk();
    mac_issue = 1'b0;

    if (INST_MODE == CIM_MODE_BIT_SERIAL) begin
      // Let the resetless serial macro wrapper consume one full window before parent reset
      for (int cycle = 1; cycle < BASE_A_WIDTH; cycle++) begin
        drive_dropped_issue_noise();
        tick_mclk();
        mac_issue = 1'b0;
      end
    end else begin
      drive_dropped_issue_noise();
      tick_mclk();
      mac_issue = 1'b0;
    end

    rstn = 1'b0;
    tick_mclk();
    tick_mclk();
    if (mac_ready !== 1'b0) begin
      $fatal(1, "%s: mac_ready must be low while reset interrupts an op", CASE_NAME);
    end
    if (mac_busy !== 1'b0) begin
      $fatal(1, "%s: mac_busy must be low while reset interrupts an op", CASE_NAME);
    end
    if (c_retire !== 1'b0) begin
      $fatal(1, "%s: c_retire must clear when reset interrupts an op", CASE_NAME);
    end

    rstn = 1'b1;
    #1;
    if (mac_ready !== 1'b1) begin
      $fatal(1, "%s: mac_ready did not recover after reset-mid-op", CASE_NAME);
    end
    if (mac_busy !== 1'b0) begin
      $fatal(1, "%s: mac_busy did not clear after reset-mid-op", CASE_NAME);
    end
    if (c_retire !== 1'b0) begin
      $fatal(1, "%s: stale c_retire appeared after reset-mid-op", CASE_NAME);
    end

    tick_mclk();
    if (c_retire !== 1'b0) begin
      $fatal(1, "%s: stale retirement appeared after reset recovery idle cycle", CASE_NAME);
    end
  end
endtask

// Run reset interruption followed by a clean legal sequence
task automatic run_reset_mid_op;
  begin
    apply_reset();
    load_all_b_sets();
    reset_during_active_op();
    run_normal_ops();
  end
endtask

initial begin
  start_waveform_dump();
  drive_defaults();
  check_test_params();
  if (TEST_KIND == TEST_RESET_MID_OP) begin
    run_reset_mid_op();
  end else begin
    apply_reset();
    load_all_b_sets();
    run_normal_ops();
  end

  run_pipelined_ops();

  $display("[pass] %s", CASE_NAME);
  $finish;
end
