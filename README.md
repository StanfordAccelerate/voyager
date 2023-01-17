# MINOTAUR DNN Accelerator
This repository contains the MINOTAUR DNN accelerator HLS code. 

It is written in SystemC/C++ and can be used for simulation (which allows for fast verification) and for generating Verilog (RTL code). The RTL code is then used for RTL-level (cycle-accurate) simulations and further synthesis of real HW. 

## CI 
[![Tests](https://code.stanford.edu/tsmc40r/brainpower/accelerator/badges/master/pipeline.svg?key_text=Tests)](https://code.stanford.edu/tsmc40r/brainpower/accelerator/-/commits/master)

We are using the build-in [GitLab CI](https://docs.gitlab.com/ee/ci/). It triggers whenever someone pushes to a branch. 

### Runners
- We are using a single gitlab runner on `rsgvm9` (two are inactive)
- They can be configured using `/etc/gitlab-runner/config.toml`
- The working-dir of the runners is `/home/gitlab-runner/minotaur-ci`

## Setup
1. Clone the repository `git@code.stanford.edu:tsmc40r/brainpower/accelerator.git`
2. Initialize the submodules `git submodule update --init --recursive`
3. A couple of things to check:
    - You need to have [`git lfs`](https://git-lfs.github.com/) installed
    - You need a `g++` with at least C++17 support
    - Make sure you have set `LD_LIBRARY_PATH=/cad/mentor/2021.1/Mgc_home/shared/lib/`
4. Create a virtual env (conda or venv) and install the packages from `requirements.txt`
5. If you want to run more than just the `simple` model, you will need to generate data files. Please see `models/` for Python scripts that help you with that.
6. Alternatively, you can use the ZagZig coden. Please take a look at the GitLab CI files for what commnands to run.

## Test 
1. After setup, use the `run_tests.py` script to run the tests
2. If you want to "manually" run the tests, you can invoke `make TestRunner` directly and then use, for example, `NETWORK=resnet TESTS=conv1 SIMS=fp32,file ./build/TestRunner` to run layer `conv1` of resnet, comparing `fp32` (Floating-point gold model) vs `file` (tensor data recorded from PyTorch)

## Structure
- `/data` test data for verification
- `/lib` external libraries we are using
- `/models` DNN models that can be run for verification
- `/scripts` various `.tcl` scripts for RTL generation
- `/src` actual SystemC accelerator implementation
- `/test` contains the testing infrastructure, mostly SystemC/C++ files that invoke the accelerator 
- `/zagzig` codegen to automatically generate code for MINOTAUR
- `Makefile` builds the source code
- `run_test.py` main script to invoke tests from

## Integration
The accelerator by itself is not very useful. It has to be integrated into an SoC. We are using the [Chipyard Framework](https://github.com/ucb-bar/chipyard) for that. Our fork [`soc`](https://code.stanford.edu/tsmc40r/brainpower/soc) of Chipyard includes the [`minotaur-blocks`](https://code.stanford.edu/tsmc40r/brainpower/minotaur-blocks) repo as a submodule under `soc/generators/minotaur-blocks`, which in turn includes this [`accelerator`](https://code.stanford.edu/tsmc40r/brainpower/accelerator) repo under `minotaur-blocks/src/main/resources/sysc/accelerator`.

## TODO
- [ ] Remove dependence on parameters NETWORK and TESTS in CI (they could change)
- [ ] Introduce global CI variables for things like data_path, common flags, etc.
- [ ] Get end2end Python script working again, so that once can either run a single end2end sample, or a batch of samples in order to determine end2end accuracy
- [ ] Add other ResNets and AlexNet to CI. Use git patches to increase #banks in VerificationTypes.h