# MINOTAUR DNN Accelerator
[![Tests](https://code.stanford.edu/tsmc40r/brainpower/accelerator/badges/master/pipeline.svg?key_text=Tests)](https://code.stanford.edu/tsmc40r/brainpower/accelerator/-/commits/master)

This repository contains the MINOTAUR DNN accelerator HLS code. 

It is written in SystemC/C++ and can be used for simulation (which allows for fast verification) and for generating Verilog (RTL code). The RTL code is then used for RTL-level (cycle-accurate) simulations and further synthesis of real HW. 

## Setup
1. Clone the repository `git@code.stanford.edu:tsmc40r/brainpower/accelerator.git`
2. Initialize the submodules `git submodule update --init`
3. A couple of things to check:
    - You need to have [`git lfs`](https://git-lfs.github.com/) installed
    - You need a `g++` with at least C++17 support
    - Make sure you have set `LD_LIBRARY_PATH=/cad/mentor/2021.1/Mgc_home/shared/lib/`
4. If you want to run more than just the `simple` model, you will need to generate the data files TODO(fpedd)

## Test 
1. After setup, use the `run_tests.py` script to run the tests
2. If you want to "manually" run the tests, you can invoke `make` directly TODO(fpedd)

## HLS
TODO(fpedd)

## Structure
- `/data` test data for verification
- `/lib` external libraries we are using
- `/models` DNN models that can be run for verification
- `/scripts` various `.tcl` scripts for RTL generation
- `/src` actual SystemC accelerator implementation
- `/test` contains the testing infrastructure, mostly SystemC/C++ files that invoke the accelerator 
- `/tools` various files and utilities that generate or transform data used for testing
- `Makefile` builds the source code
- `run_test.py` main script to invoke tests from

## Integration
The accelerator by itself is not very useful. It has to be integrated into an SoC. We are using the [Chipyard Framework](https://github.com/ucb-bar/chipyard) for that. Our fork [`soc`](https://code.stanford.edu/tsmc40r/brainpower/soc) of Chipyard includes the [`minotaur-blocks`](https://code.stanford.edu/tsmc40r/brainpower/minotaur-blocks) repo as a submodule under `soc/generators/minotaur-blocks`, which in turn includes this [`accelerator`](https://code.stanford.edu/tsmc40r/brainpower/accelerator) repo under `minotaur-blocks/src/main/resources/sysc/accelerator`.
