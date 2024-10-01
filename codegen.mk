# Makefile targets for different codegen models/datatypes
# Each target has the format: $(CODEGEN_DIR)/networks/$(NETWORK)/$(DATATYPE)/params.pb

################################################################################
# ResNet18
################################################################################
$(CODEGEN_DIR)/networks/resnet18/INT8/params.pb: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py resnet18 --activation int8,qs=per_tensor_symmetric --weight int8,qs=per_tensor_symmetric --bf16 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/resnet18/E4M3/params.pb: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py resnet18 --activation fp8_e4m3 --weight fp8_e4m3 --bf16 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/resnet18/P8_1/params.pb: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py resnet18 --activation posit8_1 --weight posit8_1 --bf16 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

test/compiler/networks/resnet18/CFLOAT/params.pb: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py resnet18 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

test/compiler/networks/resnet18/BF16/params.pb: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py resnet18 --bf16 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

################################################################################
# ResNet50
################################################################################
$(CODEGEN_DIR)/networks/resnet50/INT8/params.pb: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py resnet50 --activation int8,qs=per_tensor_symmetric --weight int8,qs=per_tensor_symmetric --bf16 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/resnet50/E4M3/params.pb: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py resnet50 --activation fp8_e4m3 --weight fp8_e4m3 --bf16 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/resnet50/P8_1/params.pb: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py resnet50 --activation posit8_1 --weight posit8_1 --bf16 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

test/compiler/networks/resnet50/CFLOAT/params.pb: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py resnet50 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

test/compiler/networks/resnet50/BF16/params.pb: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py resnet50 --bf16 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

################################################################################
# MobileBERT (Full Network)
################################################################################
$(CODEGEN_DIR)/networks/mobilebert/INT8/params.pb: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py mobilebert --activation int8,qs=per_tensor_symmetric --weight int8,qs=per_tensor_symmetric --bf16 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/mobilebert/E4M3/params.pb: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py mobilebert --activation fp8_e4m3 --weight fp8_e4m3 --bf16 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/mobilebert/P8_1/params.pb: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py mobilebert --activation posit8_1 --weight posit8_1 --bf16 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

test/compiler/networks/mobilebert/CFLOAT/params.pb: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py mobilebert --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

test/compiler/networks/mobilebert/BF16/params.pb: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py mobilebert --bf16 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

################################################################################
# MobileBERT (Encoder Only)
################################################################################
$(CODEGEN_DIR)/networks/mobilebert_encoder/INT8/params.pb: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py mobilebert_encoder --activation int8,qs=per_tensor_symmetric --weight int8,qs=per_tensor_symmetric --bf16 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/mobilebert_encoder/E4M3/params.pb: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py mobilebert_encoder --activation fp8_e4m3 --weight fp8_e4m3 --bf16 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

$(CODEGEN_DIR)/networks/mobilebert_encoder/P8_1/params.pb: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py mobilebert_encoder --activation posit8_1 --weight posit8_1 --bf16 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

test/compiler/networks/mobilebert_encoder/CFLOAT/params.pb: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py mobilebert_encoder --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

test/compiler/networks/mobilebert_encoder/BF16/params.pb: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py mobilebert_encoder --bf16 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

################################################################################
# Gesture
################################################################################
test/compiler/networks/gesture/CFLOAT/params.pb: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py gesture --model_name_or_path models/gesture/model.pth --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

################################################################################
# Layer Tests
################################################################################
test/compiler/networks/layertest/CFLOAT/params.pb: test/compiler/run_compiler.py
	mkdir -p $(dir $@)
	python test/compiler/run_compiler.py layertest --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

