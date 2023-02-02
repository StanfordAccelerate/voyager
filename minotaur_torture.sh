set -euo pipefail

### How many times to run the torture test
TORTURE_ROUNDS=1

### Path to the data directory
DATA_DIR=/pool0/fpedd/su/imagenet/val_1k

### ONNX models to test (need to reside in the models/onnx directory)
# MODELS="resnet18 resnet34 resnet50 resnet101 resnet152 mobilebert_single_enc"
MODELS="resnet18"
# MODELS="resnet34"
# MODELS="resnet50"
# MODELS="resnet101"
# MODELS="resnet152"
# MODELS="alexnet"
# MODELS="mobilenetv2"
# MODELS="mobilebert_single_enc"
# MODELS="mobilebert"
# MODELS="testnet"

### Set the frontend to use
ZAGZIG_FRONTEND="zigzag" # or "zigzag_random", "naive"

### Set the number of RRAM banks
# RRAM_BANKS=15
RRAM_BANKS=3

### Number of tests to run (-1 for all)
NUM_TESTS=-1

for tr in $(seq 1 $TORTURE_ROUNDS); do
  for model in $MODELS; do

    # Determine model family
    if [[ $model == *"net"* ]]; then # TODO: Add dirs for alexnet, mobilenetv2, ...
      model_family="resnet"
    elif [[ $model == *"bert"* ]]; then
      model_family="mobilebert"
    else
      echo -e "Unknown model family $model"
      exit 1
    fi

    echo -e "#########################"
    echo -e "##### TORTURE ROUND $tr/$TORTURE_ROUNDS"
    echo -e "##### Model: $model"
    echo -e "##### Family: ${model_family}"
    echo -e "##### Frontend: ${ZAGZIG_FRONTEND}"
    echo -e "#########################"

    # Run ZagZig and create runtime executables
    time (cd zagzig && python3 run_zagzig.py \
      --data_dir /pool0/fpedd/su/imagenet/val_1k \
      --model_path zagzig/models/onnx/${model}.onnx \
      --frontend ${ZAGZIG_FRONTEND} \
      --bw_aware \
      --rram_banks $RRAM_BANKS)
      # --debug \

    # Compile and run tests comparing the goldmodel using floating point and the reference data from the ONNX runtime
    python3 run_tests.py --model ${model}_codegen --data_dir test/${model_family}/gen_data/ --simulators fp32,file --tolerance 0.01 --rram_banks $RRAM_BANKS --num_tests $NUM_TESTS # --make_clean

    # Compile and run tests comparing the goldmodel using custom posit and the systemC HLS code
    python3 run_tests.py --model ${model}_codegen --data_dir test/${model_family}/gen_data/ --simulators accelerator,customposit --tolerance 0.01 --rram_banks $RRAM_BANKS --num_tests $NUM_TESTS

  done
done
