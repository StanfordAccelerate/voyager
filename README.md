
# Voyager DNN Accelerator Generator

This repository contains the Voyager DNN accelerator generator.

It is written in SystemC/C++ and can be used for simulation (which allows for fast verification) and for generating Verilog (RTL code). The RTL code is then used for RTL-level (cycle-accurate) simulations and further synthesis of real HW.

## CI
[![Tests](https://code.stanford.edu/tsmc40r/brainpower/accelerator/badges/master/pipeline.svg?key_text=Tests)](https://code.stanford.edu/tsmc40r/brainpower/accelerator/-/commits/master)

We are using the built-in [GitLab CI](https://docs.gitlab.com/ee/ci/). It triggers whenever someone pushes to a branch.

### Runners
- We are using a single GitLab runner on `r8cad-tsmc40r`.
- They can be configured using `/etc/gitlab-runner/config.toml`.
- The working directory of the runners is `/sim/gitlab-runner/`.

## Setup

### 0. Preliminary Step
Install conda with [miniforge](https://github.com/conda-forge/miniforge/#download).

### 1. Clone the repository
```bash
git clone git@code.stanford.edu:tsmc40r/brainpower/accelerator.git
```

### 2. Environment Requirements
Install [direnv](https://direnv.net/), and then create your own `.envrc` file in the project root directory that sets up the conda environment and sources the `env.sh` file. A sample configuration might look like this:

```bash
layout anaconda ./.conda-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
source env.sh
```

After creating the `.envrc` file, ensure the conda environment is activated.

- **Additional Requirements**:
  - You need to have [`git lfs`](https://git-lfs.github.com/) installed.
  - You need `g++` with at least C++17 support.
  - You need a python3 environment with required packages installed (try running the tests and you will be notified of the missing packages).

### 3. Initialize the submodules
```bash
git submodule update --init --recursive
```
- After initialization, install required packages:
  - In the submodule `quantized_training`, install with `pip install -r requirements.txt`
  - Further, run `pip install -e .`
  - From the root directory, execute `pip install quantized_training`

### Running Examples
Once the setup is complete, you can test the environment with the following command to compile and run ResNet18 through the compiler:
```bash
make test/compiler/networks/resnet18/CFLOAT/params.pb
```
This will generate the folder `test/compiler/networks/resnet18` with files such as a parameter list, a log file, and a `tensor_files` breakdown folder, among others.

*Note:* You may ignore command-line output messages like:
```bash
make: lsb_release: Command not found
```

To run a regression test for ResNet18, you can execute:
```bash
DATATYPE=CFLOAT IC_DIMENSION=16 OC_DIMENSION=16 python run_regression.py --models resnet18 --sims gold_model --num_processes 32
```
Upon completion, check the `regression_results` folder for the `latest` subfolder, where a list of log files for each submodule will be generated. All layers should ideally pass the gold model check. Feel free to adjust parameters and run different simulations as needed.

## Test 
1. The easiest way to run tests is to check the CI files, `.gitlab/ci/sysc.yml` and `.gitlab/ci/rtl.yml`, for SystemC and RTL simulation, respectively. The command to run regression can be found in the `script` field. The result will be written to the `regression_results` directory.
2. You can also manually build and run individual configuration and layer. The instruction will be added later.

## Structure

- `/data`: Test data for simulation.
- `/lib`: External libraries.
- `/models`: DNN models used for verification.
- `/scripts`: Various `.tcl` scripts for RTL generation.
- `/src`: SystemC accelerator implementation.
- `/test`: Testing infrastructure, primarily SystemC/C++ files that invoke the accelerator.
- `Makefile`: Builds the source code.
- `run_regression.py`: Main script to invoke regression tests.

## Adding New Models
To try a new model, you’ll need to modify this script:
- [run_compiler.py](https://code.stanford.edu/tsmc40r/brainpower/accelerator/-/blob/compiler/test/compiler/run_compiler.py?ref_type=heads): Reference existing models to understand the required compilation steps.

Add the network in the following file:
- [codegen.mk](https://code.stanford.edu/tsmc40r/brainpower/accelerator/-/blob/compiler/codegen.mk?ref_type=heads): Specify the model and set up any required quantization.

