# MINOTAUR DNN Accelerator
This repository contains the MINOTAUR DNN accelerator HLS code. 

It is written in SystemC/C++ and can be used for simulation (which allows for fast verification) and for generating Verilog (RTL code). The RTL code is then used for RTL-level (cycle-accurate) simulations and further synthesis of real HW. 

## CI 
[![Tests](https://code.stanford.edu/tsmc40r/brainpower/accelerator/badges/master/pipeline.svg?key_text=Tests)](https://code.stanford.edu/tsmc40r/brainpower/accelerator/-/commits/master)

We are using the built-in [GitLab CI](https://docs.gitlab.com/ee/ci/). It triggers whenever someone pushes to a branch. 

### Runners
- We are using a single gitlab runner on `rsgvm9` (two are inactive)
- They can be configured using `/etc/gitlab-runner/config.toml`
- The working-dir of the runners is `/home/gitlab-runner/minotaur-ci`

## Setup
1. Clone the repository `git@code.stanford.edu:tsmc40r/brainpower/accelerator.git`
2. Initialize the submodules `git submodule update --init --recursive`
3. Environment requirements:
  - You need to have [`git lfs`](https://git-lfs.github.com/) installed
  - You need a `g++` with at least C++17 support
  - You need a python3 environment with required packages installed (try running the tests and you will be notified of the missing packages).
  - You need to include the Catapult lib in your `LD_LIBRARY_PATH`. For example, `export LD_LIBRARY_PATH=/cad/mentor/2024.1/Mgc_home/lib:/cad/mentor/2024.1/Mgc_home/shared/lib:$LD_LIBRARY_PATH`.
  - Note: you can change the `.envrc` file according to your system environment and use [`direnv`](https://direnv.net) to automatically load it.

## Test 
1. The easiest way to run tests is to check the CI files, `.gitlab/ci/sysc.yml` and `.gitlab/ci/rtl.yml`, for SystemC and RTL simulation, respectively. The command to run regression can be found in the `script` field. The result will be written to the `regression_results` directory.
2. You can also manually build and run individual configuration and layer. The instruction will be added later.

## Structure
- `/data` test data for simulation
- `/lib` external libraries we are using
- `/models` DNN models that can be run for verification
- `/scripts` various `.tcl` scripts for RTL generation
- `/src` actual SystemC accelerator implementation
- `/test` contains the testing infrastructure, mostly SystemC/C++ files that invoke the accelerator 
- `Makefile` builds the source code
- `run_regression.py` main script to invoke regression tests

## Integration
The accelerator by itself is not very useful. It has to be integrated into an SoC. We are using the [Chipyard Framework](https://github.com/ucb-bar/chipyard) for that. Our fork [`soc`](https://code.stanford.edu/tsmc40r/brainpower/soc) of Chipyard includes the [`minotaur-blocks`](https://code.stanford.edu/tsmc40r/brainpower/minotaur-blocks) repo as a submodule under `soc/generators/minotaur-blocks`, which in turn includes this [`accelerator`](https://code.stanford.edu/tsmc40r/brainpower/accelerator) repo under `minotaur-blocks/src/main/resources/sysc/accelerator`.

## TODO
- [ ] Remove dependence on parameters NETWORK and TESTS in CI (they could change)
- [ ] Introduce global CI variables for things like data_path, common flags, etc.
- [ ] Get end2end Python script working again, so that once can either run a single end2end sample, or a batch of samples in order to determine end2end accuracy
- [ ] Add other ResNets and AlexNet to CI. Use git patches to increase #banks in VerificationTypes.h
