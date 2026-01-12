## How to Add a New Function to the Vector Unit using the Approximation Unit:

1. **Piecewise Fit:**
    * Linear piecewise: for linear functions, you can manually define ranges and coefficients.
    * Nonlinear piecewise: generate ranges and polynomial coefficients using the python scripts `CoeffGen.py` or `FreeKnotSplineCoeffGen.py`.
      * `CoeffGen.py` requires user-defined number of ranges and range bounds, according to the complexity of the nonlinear part of the function. It generates polynomial coefficients (`NUM_COEFFS` = 4 per range) and plots the fitted nonlinear part compared to the original one. After combining with the linear part of the function (e.g. clamp to 0 at x > 0 or f(x) = x at x > 1), you should have `NUM_RANGES` = 7.
      * `FreeKnotSplineCoeffGen.py` requires user-defined number of ranges (`num_knots` + 1) and automatically defines the bounds of each range. It generates range bounds and polynomial coefficients (`NUM_COEFFS` = degree + 1 = 3 per range) and plots the fitted function compared to the original one. After combining with the linear part of the function, you should have `NUM_RANGES` = 7.
    * Functions with parameters: define what ranges and coefficients can be determined before runtime.
    * Further check the accuracy of this piecewise fit using `AccuracyTest.py`. Add the ranges and coefficients to it and choose a model using the corresponding activation function. Check the model accuracy before and after replacing the activation function.

2. **Software Simulation (Gold):**
    * In `ApproximationConstants.h`:
      * Add the ranges and coefficients in the correct format.
      * If the function takes in extra parameters at runtime (e.g. leaky_relu), add the function opcode to `unary_ops_with_kwargs` in `ApproximationConstants.h`.
    * In `test/common/operations/VectorOps.h`:
      * Add the function opcode to `unary_ops`.
      * Add `else if` case as done for the other activation functions to call `poly_approx()` with appropriate ranges and constants in `ApproximationConstants.h`.
      * If the function takes in extra parameters at runtime (e.g. leaky_relu), some math and coefficient loading will be done within this `else if` block.

3. **HLS Integration:**
    * In `test/toolchain/Common.h`: 
      * Add the new function's opcode to the second stage in `vector_unit_stages`.
      * Map the new function's opcode to `VectorInstructions::vpoly`.
      * Note: see unit test support to determine the function's opcode if you are unsure. 
    * In `test/toolchain/VectorOps.h`:
      * Add `else if` case as done for the other activation functions to feed appropriate ranges and constants from `ApproximationConstants.h` into `vector_instruction_config`.
      * If the function takes in extra parameters at runtime (e.g. leaky_relu), some math and coefficient loading will be done within this `else if` block.
    * In `test/toolchain/MatrixOps.h`:
      * Add `else if` case as done for the other activation functions to feed appropriate ranges and constants from `ApproximationConstants.h` into `vector_instruction_config` for fused operations.
      * If the function takes in extra parameters at runtime (e.g. leaky_relu), some math and coefficient loading will be done within this `else if` block.

4. **Unit Test Support:**
    * In `voyager_compiler/test_codegen.py` add the new function to the `polyapprox` model portion.
    * Gold vs. PyTorch test (e.g. calling gelu, silu, and tanhshrink): `DATATYPE=MXINT8 IC_DIMENSION=32 OC_DIMENSION=32 python run_regression.py --models polyapprox --tests gelu,silu,tanh_1 --sims gold_model --num_processes 16`
      * We expect differences in output because (1) we are approximating and (2) datatype differences (the Vector Unit's 16-bit BFloat16 versus Pytorch's 32-bit type).
    * Gold vs. Fast-SystemC (e.g. calling gelu, silu, and tanhshrink): `DATATYPE=MXINT8 IC_DIMENSION=32 OC_DIMENSION=32 python run_regression.py --models polyapprox --tests gelu,silu,tanh_1 --sims fast-systemc --num_processes 16`
      * We expect no difference in outputs. 
    * Opcode: Running the Gold vs. PyTorch test also serves as a way to verify the new function's opcode. After running, check `test/compiler/networks/polyapprox/model.txt` and the opcode should be in here. 

--- 
## Approximation Unit Code Overview

### `test/toolchain/ApproximationConstants.h`

* Defines constants for each non-linear function that is approximated using the Approximation Unit.
* For each function:
  * `maxes`: VectorType array of size `NUM_MAXES`
  * `ranges` VectorType 2D array of size `NUM_RANGES` by `NUM_COEFFS`
  * `clamp_min` and `clamp_max` ac_ints
* `NUM_MAXES` = 6, `NUM_RANGES` = 7, and `NUM_COEFFS` = 4 is specified in `ArchitectureParams.h`, which is widely included on both software and hardware sides for simplicity. These are hard-coded and should not be changed because the Approximation Unit relies on them with its quadratic approximation.

### `src/ApproximationUnit.h`

* Implements a piecewise polynomial approximation unit.
* Supports `NUM_RANGES` ranges, each with a polynomial of degree 3 (because `NUM_COEFFS` = 4).
* `clamp_min` applies to the lowest max (`<= max[0]`)
* `clamp_max` applies to the highest max (`<= max[NUM_MAXES - 1]`) (so you lose the 7th range from `max[NUM_MAXES - 1]` to infinity)

### `src/Params.h`
* `VectorInstructionConfig` struct extended in with `ApproxUnitConfig` that holds:
  * `maxes`, `ranges`, `clamp_min`, `clamp_max`

### `src/VectorPipelines.h`
* For each of 8 instructions in `VectorInstructionConfig` sends the `ApproxUnitConfig` in the instruction config to `VectorPipelineMain.h`

### `src/VectorPipelineMain.h`
* Pops `approx_unit_config` only for the first instruction (matching push frequency).
* And calls `vpoly()` in `VectorOps.h` with the fields of `approx_unit_config`.

### `src/VectorOps.h`
* Function `vpoly()` handles all approximated functions and forwards inputs to `vepoly()` in `ApproximationUnit.h`.

### Gold Simulation
* The Gold model of `vepoly` is called `poly_approx` and lives in `test/common/operations/VectorOps.h`.
* Because `VectorInstructionConfig` is not used in Gold, coefficients are passed from `ApproximationConstants.h` to `poly_approx` directly in `test/common/operations/Softmax.h` and `test/common/operations/VectorOps.h`.


