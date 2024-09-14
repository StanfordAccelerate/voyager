# Makefile targets for different codegen models/datatypes
# Each target has the format: test/compiler/networks/$(NETWORK)/$(DATATYPE)/params.pb

################################################################################
# ResNet18
################################################################################
test/compiler/networks/resnet18/INT8/params.pb: quantized-training/test/test_codegen.py
	mkdir -p $(dir $@)
	python quantized-training/test/test_codegen.py --model resnet18 --activation int8,qs=per_tensor_symmetric --weight int8,qs=per_tensor_symmetric --bf16 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

test/compiler/networks/resnet18/E4M3/params.pb: quantized-training/test/test_codegen.py
	mkdir -p $(dir $@)
	python quantized-training/test/test_codegen.py --model resnet18 --activation fp8_e4m3 --weight fp8_e4m3 --bf16 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

################################################################################
# ResNet50
################################################################################
test/compiler/networks/resnet50/INT8/params.pb: quantized-training/test/test_codegen.py
	mkdir -p $(dir $@)
	python quantized-training/test/test_codegen.py --model resnet50 --activation int8,qs=per_tensor_symmetric --weight int8,qs=per_tensor_symmetric --bf16 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

test/compiler/networks/resnet50/E4M3/params.pb: quantized-training/test/test_codegen.py
	mkdir -p $(dir $@)
	python quantized-training/test/test_codegen.py --model resnet50 --activation fp8_e4m3 --weight fp8_e4m3 --bf16 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

################################################################################
# MobileBERT
################################################################################
test/compiler/networks/mobilebert/INT8/params.pb: quantized-training/test/test_codegen.py
	mkdir -p $(dir $@)
	python quantized-training/test/test_codegen.py --model mobilebert --activation int8,qs=per_tensor_symmetric --weight int8,qs=per_tensor_symmetric --bf16 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1

test/compiler/networks/mobilebert/E4M3/params.pb: quantized-training/test/test_codegen.py
	mkdir -p $(dir $@)
	python quantized-training/test/test_codegen.py --model mobilebert --activation fp8_e4m3 --weight fp8_e4m3 --bf16 --output_dir $(dir $@) > $(dir $@)/codegen.log 2>&1
