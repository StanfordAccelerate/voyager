# Makefile targets for different codegen models/datatypes
# Each target has the format: $(CODEGEN_DIR)/networks/$(NETWORK)/$(DATATYPE)/model.txt

E4M3_FLAGS := --activation fp8_e4m3 --weight fp8_e4m3 --bf16
P8_1_FLAGS := --activation posit8_1 --weight posit8_1 --bf16
INT8_FLAGS := --activation int8,qs=per_tensor_symmetric --weight int8,qs=per_tensor_symmetric --bias int24 --bf16
INT8_32_FLAGS := --activation int8,qs=per_tensor_symmetric --weight int8,qs=per_tensor_symmetric --bias int32 --bf16
MXINT8_FLAGS := --activation int8,qs=microscaling,bs=32 --weight int8,qs=microscaling,bs=32 --force_scale_power_of_two --bf16

################################################################################
# ResNet18
################################################################################
$(CODEGEN_DIR)/networks/resnet18/INT8/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py resnet18 $(INT8_FLAGS) --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1
 
$(CODEGEN_DIR)/networks/resnet18/INT8_32/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py resnet18 $(INT8_32_FLAGS) --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/resnet18/E4M3/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py resnet18 $(E4M3_FLAGS) --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/resnet18/P8_1/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py resnet18 $(P8_1_FLAGS) --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

test/compiler/networks/resnet18/CFLOAT/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py resnet18 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

test/compiler/networks/resnet18/FP32/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py resnet18 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

test/compiler/networks/resnet18/BF16/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py resnet18 --bf16 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

test/compiler/networks/resnet18/MXINT8/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py resnet18 $(MXINT8_FLAGS) --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

################################################################################
# ResNet50
################################################################################
$(CODEGEN_DIR)/networks/resnet50/INT8/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py resnet50 $(INT8_FLAGS) --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/resnet50/INT8_32/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py resnet50 $(INT8_32_FLAGS) --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/resnet50/E4M3/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py resnet50 $(E4M3_FLAGS) --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/resnet50/P8_1/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py resnet50 $(P8_1_FLAGS) --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

test/compiler/networks/resnet50/CFLOAT/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py resnet50 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

test/compiler/networks/resnet50/FP32/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py resnet50 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

test/compiler/networks/resnet50/BF16/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py resnet50 --bf16 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

test/compiler/networks/resnet50/MXINT8/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py resnet50 $(MXINT8_FLAGS) --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

################################################################################
# MobileBERT (Full Network)
################################################################################
$(CODEGEN_DIR)/networks/mobilebert/INT8/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py mobilebert $(INT8_FLAGS) --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/mobilebert/INT8_32/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py mobilebert $(INT8_32_FLAGS) --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/mobilebert/E4M3/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py mobilebert $(E4M3_FLAGS) --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/mobilebert/P8_1/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py mobilebert $(P8_1_FLAGS) --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/mobilebert/FP32/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py mobilebert --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

test/compiler/networks/mobilebert/CFLOAT/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py mobilebert --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

test/compiler/networks/mobilebert/BF16/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py mobilebert --bf16 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

test/compiler/networks/mobilebert/MXINT8/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py mobilebert $(MXINT8_FLAGS) --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

################################################################################
# MobileBERT (Encoder Only)
################################################################################
$(CODEGEN_DIR)/networks/mobilebert_encoder/INT8/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py mobilebert_encoder $(INT8_FLAGS) --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/mobilebert_encoder/INT8_32/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py mobilebert_encoder $(INT8_32_FLAGS) --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/mobilebert_encoder/E4M3/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py mobilebert_encoder $(E4M3_FLAGS) --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/mobilebert_encoder/P8_1/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py mobilebert_encoder $(P8_1_FLAGS) --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

test/compiler/networks/mobilebert_encoder/CFLOAT/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py mobilebert_encoder --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

test/compiler/networks/mobilebert_encoder/FP32/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py mobilebert_encoder --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

test/compiler/networks/mobilebert_encoder/BF16/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py mobilebert_encoder --bf16 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

test/compiler/networks/mobilebert_encoder/MXINT8/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py mobilebert_encoder $(MXINT8_FLAGS) --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

################################################################################
# Misc.
################################################################################

test/compiler/networks/bert/%/model.txt: quantized-training/test/test_codegen.py
	mkdir -p $(dir $@)
	python quantized-training/test/test_codegen.py bert $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) --output_dir $(dir $@) &> $(dir $@)codegen.log

test/compiler/networks/llama/%/model.txt: quantized-training/test/test_codegen.py
	mkdir -p $(dir $@)
	python quantized-training/test/test_codegen.py llama_decoder $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) --output_dir $(dir $@) &> $(dir $@)codegen.log

test/compiler/networks/vit/%/model.txt: quantized-training/test/test_codegen.py
	mkdir -p $(dir $@)
	python quantized-training/test/test_codegen.py vit $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) --output_dir $(dir $@) &> $(dir $@)codegen.log

test/compiler/networks/segformer/%/model.txt: quantized-training/test/test_codegen.py
	mkdir -p $(dir $@)
	python -u quantized-training/test/test_codegen.py segformer $($(notdir $(patsubst %/,%,$(dir $@)))_FLAGS) --output_dir $(dir $@) &> $(dir $@)codegen.log

################################################################################
# Gesture
################################################################################
test/compiler/networks/gesture/CFLOAT/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py gesture --model_name_or_path models/gesture/model.pth --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

################################################################################
# Layer Tests
################################################################################
test/compiler/networks/layertest/CFLOAT/model.txt: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py layertest --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1
