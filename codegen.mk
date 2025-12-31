# Makefile targets for different codegen models/datatypes
# Each target has the format: $(CODEGEN_DIR)/networks/$(NETWORK)/$(DATATYPE)/model.txt

E4M3_FLAGS := --activation fp8_e4m3 --weight fp8_e4m3 --bf16
P8_1_FLAGS := --activation posit8_1 --weight posit8_1 --bf16
INT8_FLAGS := --activation int8,qs=per_tensor_symmetric --weight int8,qs=per_tensor_symmetric --bias int24 --bf16 --calibration_steps 10
INT8_32_FLAGS := --activation int8,qs=per_tensor_symmetric --weight int8,qs=per_tensor_symmetric --bias int32 --bf16 --calibration_steps 10
BLOCK_SIZE := $(shell [ $(IC_DIMENSION) -gt $(OC_DIMENSION) ] && echo $(IC_DIMENSION) || echo $(OC_DIMENSION))
MXINT8_FLAGS := --activation int8,qs=microscaling,bs=$(BLOCK_SIZE) --weight int8,qs=microscaling,bs=$(BLOCK_SIZE) --force_scale_power_of_two --bf16
MXNF4_FLAGS := --activation nf4_6,qs=microscaling,bs=$(BLOCK_SIZE),scale=fp8_e5m3 --weight nf4_6,qs=microscaling,bs=$(BLOCK_SIZE),scale=fp8_e5m3 --bf16 --residual fp8_e4m3 --quantize_fc
COMMON_FLAGS := --transpose_weight --hardware_unrolling $(IC_DIMENSION),$(OC_DIMENSION) --dump_tensors
EXTRA_COMPILER_FLAGS ?=

CONTEXT ?= 1024
LLM_FLAGS := --context_length $(CONTEXT) --remove_duplicate --binary_mask

# Set default values if not already defined in the environment
CACHE_SIZE ?= 8388608
NUM_BANKS  ?= 8

ifeq ($(SOC_SIM),1)
COMMON_FLAGS += --cache_size $(CACHE_SIZE) --num_banks $(NUM_BANKS)
endif

################################################################################
$(CODEGEN_DIR)/networks/resnet18/%/model.txt: voyager-compiler/test/test_codegen.py
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py resnet18 $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/resnet50/%/model.txt: voyager-compiler/test/test_codegen.py
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py resnet50 $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/mobilebert/%/model.txt: voyager-compiler/test/test_codegen.py
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py mobilebert $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) --model_name_or_path models/mobilebert/mobilebert-tiny-sst2-bf16 $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/mobilebert_encoder/%/model.txt: voyager-compiler/test/test_codegen.py
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py mobilebert $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) --model_name_or_path models/mobilebert/mobilebert-tiny-sst2-bf16 $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) --remove_duplicate > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/bert/%/model.txt: voyager-compiler/test/test_codegen.py
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py bert $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) &> $(dir $@)codegen.log

$(CODEGEN_DIR)/networks/llama_prefill/%/model.txt: voyager-compiler/test/test_codegen.py
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py llm_prefill $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) $(LLM_FLAGS) &> $(dir $@)codegen.log

$(CODEGEN_DIR)/networks/llama_decode/%/model.txt: voyager-compiler/test/test_codegen.py
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py llm_decode $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) $(LLM_FLAGS) &> $(dir $@)codegen.log

$(CODEGEN_DIR)/networks/llama_decode_kivi/%/model.txt: voyager-compiler/test/test_codegen.py
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py llm_kivi $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) $(LLM_FLAGS) &> $(dir $@)codegen.log

$(CODEGEN_DIR)/networks/llama_prefill_mp/%/model.txt: voyager-compiler/test/test_codegen.py
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py llm_prefill $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) $(LLM_FLAGS) --mixed_precision &> $(dir $@)codegen.log

$(CODEGEN_DIR)/networks/llama_prefill_spmm/%/model.txt: voyager-compiler/test/test_codegen.py
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py llm_prefill $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) $(LLM_FLAGS) --mixed_precision --outlier_pct 0.01 --split_spmm &> $(dir $@)codegen.log

$(CODEGEN_DIR)/networks/llama_decode_mp/%/model.txt: voyager-compiler/test/test_codegen.py
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py llm_decode $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) $(LLM_FLAGS) --mixed_precision &> $(dir $@)codegen.log

$(CODEGEN_DIR)/networks/vit/%/model.txt: voyager-compiler/test/test_codegen.py
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py vit $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) &> $(dir $@)codegen.log

$(CODEGEN_DIR)/networks/segformer/%/model.txt: voyager-compiler/test/test_codegen.py
	mkdir -p $(dir $@)
	python -u voyager-compiler/test/test_codegen.py segformer $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) &> $(dir $@)codegen.log

$(CODEGEN_DIR)/networks/mobilenet_v2/%/model.txt: voyager-compiler/test/test_codegen.py
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py mobilenet_v2 $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) &> $(dir $@)codegen.log

################################################################################
# Gesture
################################################################################
$(CODEGEN_DIR)/networks/gesture/CFLOAT/model.txt: voyager-compiler/test/test_codegen.py
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py gesture --model_name_or_path models/gesture/model.pth --model_output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

################################################################################
# Layer Tests
################################################################################
test/compiler/networks/layertest/CFLOAT/model.txt: voyager-compiler/test/test_codegen.py
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py layertest --model_output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

################################################################################
# Tilings
################################################################################
$(CODEGEN_DIR)/networks/$(NETWORK)/$(DATATYPE)/$(IC_DIMENSION)x$(OC_DIMENSION)_$(INPUT_BUFFER_SIZE)x$(WEIGHT_BUFFER_SIZE)x$(ACCUM_BUFFER_SIZE)_$(DOUBLE_BUFFERED_ACCUM_BUFFER)/tilings.txtpb: $(CODEGEN_DIR)/networks/$(NETWORK)/$(DATATYPE)/model.txt test/compiler/run_tiler.py test/compiler/proto/tiling_pb2.py
	mkdir -p $(dir $@)
	python test/compiler/run_tiler.py --codegen_dir $(dir $<) --IC_dimension $(IC_DIMENSION) --OC_dimension $(OC_DIMENSION) --input_buffer_size $(INPUT_BUFFER_SIZE) --weight_buffer_size $(WEIGHT_BUFFER_SIZE) --accum_buffer_size $(ACCUM_BUFFER_SIZE) $(if $(filter true,$(DOUBLE_BUFFERED_ACCUM_BUFFER)), --double_buffered_accum_buffer) > $(dir $@)/tiler.log 2>&1
