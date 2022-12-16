set -euo pipefail

TORTURE_ROUNDS=100
DATA_DIR=/pool0/fpedd/su/imagenet/val_1k
# DATA_DIR=~/su/imagenet/val_1k # for local testing

for tr in $(seq 1 $TORTURE_ROUNDS); do
  echo -e "##### TORTURE ROUND $tr #####"
  (cd zagzig && python3 run_zagzig.py --data_dir $DATA_DIR --frontend zigzag_random)
  python3 run_tests.py --model resnet18_codegen --data_dir test/resnet/gen_data/ --simulators fp32,file --tolerance 0.01
  python3 run_tests.py --model resnet18_codegen --data_dir test/resnet/gen_data/ --simulators customposit,accelerator --tolerance 0.01
done