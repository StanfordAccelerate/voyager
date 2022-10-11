# MINOTAUR DNN Accelerator
This repository contains the MINOTAUR DNN accelerator HLS code. 

It is written in SystemC/C++ and can be used for simulation (which allows for fast verification) and for generating Verilog (RTL code). The RTL code is then used for RTL-level (cycle-accurate) simulations and further synthesis of real HW. 

## CI 
[![Tests](https://code.stanford.edu/tsmc40r/brainpower/accelerator/badges/master/pipeline.svg?key_text=Tests)](https://code.stanford.edu/tsmc40r/brainpower/accelerator/-/commits/master)

We are using the build-in [GitLab CI](https://docs.gitlab.com/ee/ci/). It triggers whenever someone pushes to a branch. 

### Naming
The CI naming convention for a job is `<type_of_test>_<model>_<what_is_being_compared>`. 
- `<type_of_test>` 
    - `LayerTest` tests on a layer-by-layer basis 
    - `End2EndTest` running from start to finish through all models
    - `UnitTest` testing individual components
- `<model>`
    - `Mobilebert`
    - `ResNet`
- `<what_is_being_compared>`
    - `Accelerator` the SystemC HLS model
    - `Customposit` gold model using our Posit implementation
    - `FP32` gold model using 32bit floating-point
    - `Universal` gold model using universal's Posit implementation
    - `File` reference data collected through PyTorch

### Runners
- We are using three runners on `rsgvm9`
- They can be configured using `/etc/gitlab-runner/config.toml`
- The working-dir of the runners is `/pool0/minotaur-ci`

## Setup
1. Clone the repository `git@code.stanford.edu:tsmc40r/brainpower/accelerator.git`
2. Initialize the submodules `git submodule update --init`
3. A couple of things to check:
    - You need to have [`git lfs`](https://git-lfs.github.com/) installed
    - You need a `g++` with at least C++17 support
    - Make sure you have set `LD_LIBRARY_PATH=/cad/mentor/2021.1/Mgc_home/shared/lib/`
4. If you want to run more than just the `simple` model, you will need to generate data files. Please see `models/` for Python scripts that help you with that.

## Test 
1. After setup, use the `run_tests.py` script to run the tests
2. If you want to "manually" run the tests, you can invoke `make TestRunner` directly and then use, for example, `MODEL=resnet TESTS=conv1 SIMS=fp32,file ./build/TestRunner` to run layer `conv1` of resnet, comparing `fp32` (Floating-point gold model) vs `file` (tensor data recorded from PyTorch)

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


## Roadmap / TODOs

### Refactor CI / Verification concept
- Unify most of the MobileBERT / ResNet code
- Related to that, we need to prepare the verification for more than just these two types of networks
- Refactor testing / tolerance criteria
  - Currently we are simply looking at the greater than 1 realtive error bin and comparing it to a fixed threshold
  - Instead, we should probably differentiate:
    - For layer-by-layer tests we should iterate over all output values, calculate the relative difference and maybe even apply exp(). We should then sum all the errors up and compare to a threshold (called TOLERANCE in out code). This should help with detecting outliers, while ignoring small differences. However, this comes with the caveat that it is a lot harder to debug (because of the non-linearity). So we should still keep the diff tables and print them.
    - For end-to-end tests we only compare the output layer and we only care about whether the top (or top5/10) value(s) match(es). So we need to implement a special case here. Furthermore, this really only makes sense when we run multiple workloads (10s to 100s), as top categories might not always match (both models (fp and posit) make errors, but they don't necessarly make the same errors on the same inputs). However, this quickly becomes infeasable (one end-to-end run of ResNet18 takes ~10h on the SystemC model). So we need to come up with smth here. Maybe we can choose some test(s) where the top category is very obivous (large diff, high confidence) and thus unlikely to be different between models.
- Prepare for extended end-to-end tests going from model training all the way to SystemC simulation