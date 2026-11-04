// Shared CIM testbench helpers for generated Verilator harnesses
// Include this inside a generated test module after defining CASE_NAME, INPUT_LANES,
// WEIGHT_SETS, A_WIDTH, NUM_ITERS, MCLK_PERIOD, and WCLK_PERIOD

localparam int unsigned INPUT_INDEX_WIDTH = (INPUT_LANES <= 1) ? 1 : $clog2(INPUT_LANES);
localparam int unsigned SET_INDEX_WIDTH = (WEIGHT_SETS <= 1) ? 1 : $clog2(WEIGHT_SETS);
localparam int unsigned DEFAULT_RNG_SEED = 32'h1;

int unsigned rng_state;

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

// Generate one random activation vector on the a interface
task automatic randomize_activation;
  int unsigned value;
  begin
    for (int input_index = 0; input_index < INPUT_LANES; input_index++) begin
      rng_next(value);
      a[input_index] = A_WIDTH'(value);
    end
  end
endtask
