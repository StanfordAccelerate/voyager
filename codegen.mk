# Makefile targets for different codegen models/datatypes
# Each target has the format: $(CODEGEN_DIR)/networks/$(NETWORK)/$(DATATYPE)/model.txt

# Set default values if not already defined in the environment.
# SCRATCHPAD_SIZE is the whole physical L2 SRAM; COMMON_FLAGS always passes
# --double_buffered_l2, so a ping-ponged buffer gets half of it per slot.
SCRATCHPAD_SIZE ?= 2097152
SCRATCHPAD_OFFSET ?= 0
NUM_BANKS ?= 16

# One input beat in bytes: the PE array's column count times the activation
# element size. The store bus writes that word whole, so the memory planner
# aligns allocations to it and reserves the tail a sub-word beat overshoots by.
BYTE_BANK_WIDTH := $(OC_DIMENSION)
NIBBLE_BANK_WIDTH := $(shell expr $(OC_DIMENSION) / 2)

E4M3_FLAGS := --activation fp8_e4m3 --weight fp8_e4m3 --bf16 --bank_width $(BYTE_BANK_WIDTH)
P8_1_FLAGS := --activation posit8_1 --weight posit8_1 --bf16 --bank_width $(BYTE_BANK_WIDTH)
INT8_FLAGS := --activation int8,qs=per_tensor_symmetric --weight int8,qs=per_tensor_symmetric --bias int24 --bf16 --calibration_steps 3 --bank_width $(BYTE_BANK_WIDTH)
INT8_32_FLAGS := --activation int8,qs=per_tensor_symmetric --weight int8,qs=per_tensor_symmetric --bias int32 --bf16 --calibration_steps 3 --bank_width $(BYTE_BANK_WIDTH)
BLOCK_SIZE := $(shell [ $(IC_DIMENSION) -gt $(OC_DIMENSION) ] && echo $(IC_DIMENSION) || echo $(OC_DIMENSION))
MXINT8_FLAGS := --activation int8,qs=microscaling,bs=$(BLOCK_SIZE) --weight int8,qs=microscaling,bs=$(BLOCK_SIZE) --force_scale_power_of_two --bf16 --bank_width $(BYTE_BANK_WIDTH)
MXNF4_FLAGS := --activation nf4_6,qs=microscaling,bs=$(BLOCK_SIZE),scale=fp8_e5m3 --weight nf4_6,qs=microscaling,bs=$(BLOCK_SIZE),scale=fp8_e5m3 --bf16 --residual fp8_e4m3 --quantize_fc --bank_width $(NIBBLE_BANK_WIDTH)
COMMON_FLAGS := --layout_policy systolic --pe_array_size $(IC_DIMENSION),$(OC_DIMENSION) --dump_tensors --double_buffered_l2 --scratchpad_size $(SCRATCHPAD_SIZE) --scratchpad_offset $(SCRATCHPAD_OFFSET) --num_banks $(NUM_BANKS) --input_buffer_size $(INPUT_BUFFER_SIZE) --weight_buffer_size $(WEIGHT_BUFFER_SIZE) --accum_buffer_size $(ACCUM_BUFFER_SIZE)
EXTRA_COMPILER_FLAGS ?=

CONTEXT ?= 1024
LLM_FLAGS := --context_length $(CONTEXT) --compile_single_layer --quantize_attention_mask

ifneq ($(filter true 1,$(DOUBLE_BUFFERED_ACCUM_BUFFER)),)
COMMON_FLAGS += --double_buffered_accum_buffer
endif

ifdef CLOCK_PERIOD
# CLOCK_PERIOD is in ns; the compiler's cost model takes frequency in GHz.
COMMON_FLAGS += --frequency $(shell awk 'BEGIN { print 1.0 / $(CLOCK_PERIOD) }')
endif

ifeq ($(CONV2D_IM2COL),1)
COMMON_FLAGS += --conv2d_im2col
endif

################################################################################
$(CODEGEN_DIR)/networks/resnet18/%/model.txt:
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py resnet18 $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/resnet50/%/model.txt:
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py resnet50 $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/mobilebert/%/model.txt:
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py mobilebert $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) --model_name_or_path models/mobilebert/mobilebert-tiny-sst2-bf16 $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/mobilebert_encoder/%/model.txt:
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py mobilebert $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) --model_name_or_path models/mobilebert/mobilebert-tiny-sst2-bf16 $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) --compile_single_layer > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/bert/%/model.txt:
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py bert $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) &> $(dir $@)codegen.log

$(CODEGEN_DIR)/networks/llama_prefill/%/model.txt:
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py llm_prefill $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) $(LLM_FLAGS) &> $(dir $@)codegen.log

$(CODEGEN_DIR)/networks/llama_decode/%/model.txt:
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py llm_decode $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) $(LLM_FLAGS) &> $(dir $@)codegen.log

$(CODEGEN_DIR)/networks/llama_decode_kivi/%/model.txt:
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py llm_kivi $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) $(LLM_FLAGS) &> $(dir $@)codegen.log

$(CODEGEN_DIR)/networks/llama_prefill_mp/%/model.txt:
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py llm_prefill $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) $(LLM_FLAGS) --enable_mixed_precision &> $(dir $@)codegen.log

$(CODEGEN_DIR)/networks/llama_prefill_spmm/%/model.txt:
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py llm_prefill $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) $(LLM_FLAGS) --enable_mixed_precision --outlier_pct 0.01 &> $(dir $@)codegen.log

$(CODEGEN_DIR)/networks/llama_decode_mp/%/model.txt:
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py llm_decode $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) $(LLM_FLAGS) --enable_mixed_precision &> $(dir $@)codegen.log

$(CODEGEN_DIR)/networks/vit/%/model.txt:
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py vit $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) &> $(dir $@)codegen.log

$(CODEGEN_DIR)/networks/segformer/%/model.txt:
	mkdir -p $(dir $@)
	python -u voyager-compiler/test/test_codegen.py segformer $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) &> $(dir $@)codegen.log

$(CODEGEN_DIR)/networks/mobilenet_v2/%/model.txt:
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py mobilenet_v2 $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) $(EXTRA_COMPILER_FLAGS) --model_output_dir $(dir $@) $(COMMON_FLAGS) &> $(dir $@)codegen.log

################################################################################
# Gesture
################################################################################
$(CODEGEN_DIR)/networks/gesture/CFLOAT/model.txt:
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py gesture --model_name_or_path models/gesture/model.pth --model_output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

################################################################################
# Layer Tests
################################################################################
test/compiler/networks/layertest/CFLOAT/model.txt:
	mkdir -p $(dir $@)
	python voyager-compiler/test/test_codegen.py layertest --model_output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1
