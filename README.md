
# Voyager DNN Accelerator Generator

This repository contains the Voyager DNN accelerator generator, a tool for generating and evaluating deep neural network accelerators. For more details about the methodology and architecture, please refer to the paper: https://arxiv.org/abs/2509.15205

## Setup

### 0. Preliminary Step
Install conda using [miniforge](https://github.com/conda-forge/miniforge/#download). Miniforge is recommended as it uses conda-forge as the default channel, which provides better compatibility with the required packages.

### 1. Clone the repository and submodules
```bash
git clone https://github.com/StanfordAccelerate/voyager.git
cd voyager
git submodule update --init --recursive
```

### 2. Environment Requirements

Create a Conda environment from the environment.yml file:
```bash
conda env create -f environment.yml
```

**Note:** You must activate this environment every time you want to use this project:
```bash
conda activate accelerator-env
```

After activating the environment, install the required submodule packages:
```bash
pip install ./interstellar
pip install ./voyager-compiler
```

Voyager also requires the following EDA tools for RTL generation and simulation:
- **Catapult HLS** (tested with version 2024.2_1) - Used for high-level synthesis
- **VCS** (tested with T-2022.06-SP2) - Verilog simulator for RTL verification
- **Verdi** (tested with T-2022.06-SP2) - Debug tool for waveform analysis
- **VCS_GNU_PACKAGE** (tested with S-2021.09) - Required compiler toolchain for VCS

**Note**: You must load these tools every time you want to use this project. The method for activating these tools depends on your EDA environment setup (e.g., module system, environment scripts, etc.).

Additionally, set the following environment variables every time you want to use this project:
```bash
export PROJECT_ROOT=`pwd`
export CODEGEN_DIR=test/compiler
```

These variables specify the project root directory and the code generation output directory, respectively.

## Repository Structure

- `/data`: Test datasets and input data used for simulation and accuracy evaluation
- `/lib`: External libraries and dependencies required by the project
- `/models`: Pre-trained DNN model definitions and weights used for verification and evaluation
- `/scripts`: Technology-specific `.tcl` scripts for RTL generation and synthesis configuration
- `/src`: SystemC implementation of the accelerator architecture
- `/test`: Testing infrastructure, including SystemC/C++ testbenches that invoke the accelerator
- `Makefile`: Build configuration for compiling the SystemC source code
- `run_regression.py`: Main regression testing script that orchestrates accuracy evaluation, functional simulations, and RTL generation


## Usage

### Configuration

Accelerator configurations are specified through environment variables. The following variables control the accelerator architecture:

- **`DATATYPE`**: The datatype/quantization scheme used for computations. Some of the supported values include: `BF16`, `E4M3`, `INT8`, `MXINT8`
- **`IC_DIMENSION`**: The systolic array dimension size for the input channel (reduction) dimension
- **`OC_DIMENSION`**: The systolic array dimension size for the output channel dimension
- **`INPUT_BUFFER_SIZE`**: The depth (number of entries) of the input activation buffer
- **`WEIGHT_BUFFER_SIZE`**: The depth (number of entries) of the weight buffer
- **`ACCUM_BUFFER_SIZE`**: The depth (number of entries) of the accumulation buffer

Additional variables for RTL generation:
- **`CLOCK_PERIOD`**: Target clock period in nanoseconds for RTL synthesis
- **`TECHNOLOGY`**: Technology name (must match a `.tcl` file in `scripts/tech/`)

The `run_regression.py` script is the main entry point for running various types of evaluations and simulations, including:
- Accuracy evaluation on full models and datasets
- Functional SystemC simulations for verification
- Cycle-accurate RTL simulations after HLS synthesis

For detailed information about available options and underlying steps, refer to the script's help message: `python3 run_regression.py --help`. Upon completion, results and log files are generated in the `regression_results/latest/` directory.

### Running Accuracy Evaluation

Accuracy evaluation runs inference on full models and datasets to measure the impact of quantization and accelerator configuration on model accuracy. This is useful for validating that the accelerator maintains acceptable accuracy compared to floating-point baselines.

**Supported models and datasets:**
- BERT on SST-2 (sentiment classification)
- MobileBERT on SST-2 (sentiment classification)
- MobileBERT on SQuAD (question answering)
- ResNet18 on ImageNet (image classification)
- ResNet50 on ImageNet (image classification)
- ViT (Vision Transformer) on ImageNet (image classification)
- MobileNetV2 on ImageNet (image classification)

**Example:** To evaluate the accuracy of an INT8 16×16 systolic array accelerator on the ResNet-18 model using the ImageNet dataset with 32 parallel processes:
```bash
DATATYPE=INT8 IC_DIMENSION=16 OC_DIMENSION=16 python3 run_regression.py --models resnet18 --dataset imagenet --sims accuracy --num_processes 32
```

### Running Functional SystemC Simulations

Functional SystemC simulations verify the correctness of the SystemC accelerator implementation by comparing its outputs against a reference (golden) model. These simulations are **not cycle-accurate** but validate that the accelerator produces functionally correct results. The simulations run the SystemC model on every layer of the specified model(s) and compare outputs with the functional gold model.

This is typically the first verification step before proceeding to RTL generation, as it's faster than RTL simulation and catches functional bugs early.

**Example:** To verify an E4M3 32×32 accelerator on ResNet-18 and MobileBERT layers:
```bash
DATATYPE=E4M3 IC_DIMENSION=32 OC_DIMENSION=32 python3 run_regression.py --models resnet18,mobilebert_encoder --sims systemc --num_processes 32
```

### Running HLS and Cycle-Accurate RTL Simulations

This workflow generates synthesizable RTL from the SystemC model using Catapult HLS and then runs cycle-accurate RTL simulations. This is the most detailed verification step and produces actual hardware RTL that can be pushed through downstream EDA tools.

**Prerequisites:** Before running RTL generation, you must create a technology-specific configuration file. In the `scripts/tech/` directory, create a `.tcl` file named after your technology (e.g., `tsmc40.tcl`). This file configures Catapult HLS with your target technology library.

**Technology file template:**
```tcl
# Point to your Catapult-generated characterized library.
# Refer to the Catapult library builder documentation for more details.
solution options set ComponentLibs/SearchPath {/path/to/catapult_library} -append
solution library add catapult_library_name

# Implement the following function that returns the name of the memory instance
# for a given depth and width. Refer to the Catapult memory library generation
# documentation for more details.
proc get_memory_name {is_sp depth width} {
  # Return the appropriate memory name based on parameters
  # is_sp: whether it's a single-port memory
  # depth: memory depth
  # width: memory width in bits
  ...
}
```

**Example:** To run RTL generation on an INT8 32x32 accelerator with a 5.0ns clock period using the `tsmc40` technology and evaluate runtime on MobileBERT:
```bash
DATATYPE=INT8 IC_DIMENSION=32 OC_DIMENSION=32 CLOCK_PERIOD=5.0 TECHNOLOGY=tsmc40 python3 run_regression.py --models mobilebert_encoder --sims rtl --num_processes 32 --uniquify_layers
```

The `--uniquify_layers` flag can be used to reduce the evaluation runtime by only running the layers with unique shapes.

**Output:** The generated RTL files can be found in the `build/` directory, organized by configuration (datatype, dimensions, etc.) in separate subdirectories. Each configuration directory contains the synthesized Verilog RTL and other Catapult-generated collateral.
