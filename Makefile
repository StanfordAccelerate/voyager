CC = /opt/rh/devtoolset-10/root/bin/g++
INC = -I/cad/mentor/2021.1/Mgc_home/shared/include/ -Ilib/ -Ilib/universal/include/ -Isrc/ -I. -I/usr/include/python3.6m
BASEFLAGS = $(INC) -DSC_INCLUDE_DYNAMIC_PROCESSES -O3
DEBUGFLAGS = -DDEBUG_LOG -g
FASTFLAGS = -DCONNECTIONS_FAST_SIM

ifeq ($(DEBUG), 1)
    BASEFLAGS += $(DEBUGFLAGS)
endif

ifeq ($(FAST), 1)
    BASEFLAGS += $(FASTFLAGS) 
endif


C11FLAGS = $(BASEFLAGS) -std=c++11 -Wno-deprecated-declarations
C17FLAGS = $(BASEFLAGS) -std=c++17
LDFLAGS = -lsystemc -lpython3.6m
LDLIBS = -L/cad/mentor/2021.1/Mgc_home/shared/lib/

###########################################################
# Catapult Synthesis
###########################################################
rtl: build/Catapult_Accelerator/Accelerator.v1/concat_rtl.v
InputController: build/Catapult_InputController/InputController.v1/concat_rtl.v
WeightController: build/Catapult_WeightController/WeightController.v1/concat_rtl.v
MatrixProcessor: build/Catapult_MatrixProcessor/MatrixProcessor.v1/concat_rtl.v
ProcessingElement: build/Catapult_ProcessingElement/ProcessingElement.v1/concat_rtl.v
VectorUnit: build/Catapult_VectorUnit/VectorUnit.v1/concat_rtl.v
MaxpoolUnit: build/Catapult_MaxpoolUnit/MaxpoolUnit.v1/concat_rtl.v
OutputAddressGenerator: build/Catapult_OutputAddressGenerator/OutputAddressGenerator.v1/concat_rtl.v
VectorFetchUnit: build/Catapult_VectorFetchUnit/VectorFetchUnit.v1/concat_rtl.v
VectorOpUnit: build/Catapult_VectorOpUnit/VectorOpUnit.v1/concat_rtl.v


build/Catapult_InputController/InputController.v1/concat_rtl.v: src/InputController.h
	catapult -shell -file scripts/InputController.tcl
build/Catapult_WeightController/WeightController.v1/concat_rtl.v: src/WeightController.h
	catapult -shell -file scripts/WeightController.tcl
build/Catapult_ProcessingElement/ProcessingElement.v1/concat_rtl.v: src/ProcessingElement.h
	catapult -shell -file scripts/ProcessingElement.tcl
build/Catapult_MatrixProcessor/MatrixProcessor.v1/concat_rtl.v: build/Catapult_ProcessingElement/ProcessingElement.v1/concat_rtl.v
	catapult -shell -file scripts/MatrixProcessor.tcl
build/Catapult_VectorUnit/VectorUnit.v1/concat_rtl.v: build/Catapult_MaxpoolUnit/MaxpoolUnit.v1/concat_rtl.v build/Catapult_OutputAddressGenerator/OutputAddressGenerator.v1/concat_rtl.v build/Catapult_VectorFetchUnit/VectorFetchUnit.v1/concat_rtl.v build/Catapult_VectorOpUnit/VectorOpUnit.v1/concat_rtl.v
	catapult -shell -file scripts/VectorUnit.tcl
build/Catapult_MaxpoolUnit/MaxpoolUnit.v1/concat_rtl.v: src/MaxpoolUnit.h
	catapult -shell -file scripts/MaxpoolUnit.tcl
build/Catapult_OutputAddressGenerator/OutputAddressGenerator.v1/concat_rtl.v: src/OutputAddressGenerator.h
	catapult -shell -file scripts/OutputAddressGenerator.tcl
build/Catapult_VectorFetchUnit/VectorFetchUnit.v1/concat_rtl.v: src/VectorFetch.h
	catapult -shell -file scripts/VectorFetchUnit.tcl
build/Catapult_VectorOpUnit/VectorOpUnit.v1/concat_rtl.v: src/VectorUnit.h
	catapult -shell -file scripts/VectorOpUnit.tcl
build/Catapult_Accelerator/Accelerator.v1/concat_rtl.v: build/Catapult_InputController/InputController.v1/concat_rtl.v build/Catapult_WeightController/WeightController.v1/concat_rtl.v build/Catapult_MatrixProcessor/MatrixProcessor.v1/concat_rtl.v build/Catapult_VectorUnit/VectorUnit.v1/concat_rtl.v
	catapult -shell -file scripts/Accelerator.tcl
	sed '/module CGHpart/,/endmodule/d;/module TSDN/,/endmodule/d;/module TS1N40LPB1024X128M4FWBA /,/endmodule/d;/^`include/d;s/module Accelerator_rtl/module Accelerator/g;s/VectorUnit_rtl/VectorUnit/g' build/Catapult_Accelerator/Accelerator.v1/concat_rtl.v > release/concat_rtl.v

.PHONY: rtl InputController WeightController MatrixProcessor ProcessingElement VectorUnit MaxpoolUnit OutputAddressGenerator VectorFetchUnit VectorOpUnit
# rtl: build/Catapult/Accelerator.v1/concat_rtl.v

# build/Catapult/Accelerator.v1/concat_rtl.v: src/Accelerator.cc $(wildcard src/*.h) scripts/hls.tcl
# 	catapult -shell -file scripts/hls.tcl
# 	sed '/module CGHpart/,/endmodule/d;/module TSDN/,/endmodule/d;/^`include/d' build/Catapult_Accelerator/Accelerator.v1/concat_rtl.v > build/Catapult_Accelerator/Accelerator.v1/concat_rtl_clean.v

# debug_rtl: build/Catapult_debug/Accelerator.v1/concat_rtl.v

# build/Catapult_debug/Accelerator.v1/concat_rtl.v: export DEBUG=1

# build/Catapult_debug/Accelerator.v1/concat_rtl.v: src/Accelerator.cc $(wildcard src/*.h) scripts/hls.tcl
# 	catapult -shell -file scripts/hls.tcl

###########################################################
# SystemC Simulations
###########################################################
.PHONY: sim_sysc

build_folder=build/simv

simv_name = simv
ifeq ($(DEBUG), 1)
	simv_name:=simv-debug
	build_folder:=build/simv-debug
endif

sim_sysc:
	syscan -kdb -cflags "$(C11FLAGS) -g" -Mdir=$(build_folder) src/Accelerator.cc
	syscan -kdb -cflags "$(C11FLAGS) -g" -Mdir=$(build_folder) test/common/Harness.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/GoldModel.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/Utils.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/DataLoader.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/TestRunner.cc
	syscan -kdb -cflags "$(C11FLAGS) -g" -Mdir=$(build_folder) test/toolchain/MapOperation.cc
	vcs -full64 -sysc sc_main -kdb -debug_access+all -Mdir=$(build_folder) -o $(build_folder)/$(simv_name)
	./$(build_folder)/$(simv_name) -ucli -i dump_fsdb.tcl

sim_sysc_gui:
	syscan -kdb -cflags "$(C11FLAGS) -g" -Mdir=$(build_folder) src/Accelerator.cc
	syscan -kdb -cflags "$(C11FLAGS) -g" -Mdir=$(build_folder) test/common/Harness.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/GoldModel.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/Utils.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/DataLoader.cc
	syscan -kdb -cflags "$(C17FLAGS) -g" -Mdir=$(build_folder) test/common/TestRunner.cc
	syscan -kdb -cflags "$(C11FLAGS) -g" -Mdir=$(build_folder) test/toolchain/MapOperation.cc
	vcs -full64 -sysc sc_main -kdb -debug_access+all -Mdir=$(build_folder) -o $(build_folder)/$(simv_name)
	./$(build_folder)/$(simv_name) -verdi

###########################################################
# Untimed C Simulations (Faster than SystemC)
###########################################################
sim: build/TestRunner
	./build/TestRunner

PositTest: build/PositTest
	./build/PositTest

rtl_sim_clean:
	rm -rf build/Catapult_debug/Accelerator.v1/scverify/concat_sim_rtl_v_vcs

rtl_sim: rtl
	cd build/Catapult_Accelerator/Accelerator.v1 && make -f ./scverify/Verify_concat_sim_rtl_v_vcs.mk SIMTOOL=vcs sim

rtl_sim_debug: rtl
	rm -rf build/inter.fsdb*
	cd build/Catapult_Accelerator/Accelerator.v1 && make -f ./scverify/Verify_concat_sim_rtl_v_vcs.mk SIMTOOL=vcs simgui

gui:
	catapult build/Catapult_debug

build/TestRunner: build/Accelerator.o build/Harness.o build/TestRunner.o build/GoldModel.o build/Utils.o build/DataLoader.o build/MapOperation.o
	$(CC) -o $@ $^ $(LDLIBS) $(LDFLAGS)

build/PositTest: build/PositTest.o
	$(CC) -o $@ $^

build/Accelerator.o: src/Accelerator.cc $(wildcard src/*.h)
	$(CC) $(C11FLAGS) -c -o $@ $< 

build/Harness.o: test/common/Harness.cc test/common/Harness.h $(wildcard src/*.h) 
	$(CC) $(C11FLAGS) -c -o $@ $<

build/MapOperation.o: test/toolchain/MapOperation.cc test/toolchain/MapOperation.h
	$(CC) $(C11FLAGS) -DNO_SYSC -c -o $@ $<

build/GoldModel.o: test/common/GoldModel.cc test/common/GoldModel.h src/ArchitectureParams.h
	$(CC) $(C17FLAGS) -c -o $@ $<

build/Utils.o: test/common/Utils.cc test/common/Utils.h src/ArchitectureParams.h
	$(CC) $(C17FLAGS) -c -o $@ $<

build/DataLoader.o: test/common/DataLoader.cc test/common/DataLoader.h src/ArchitectureParams.h
	$(CC) $(C17FLAGS) -c -o $@ $<

build/TestRunner.o: test/common/TestRunner.cc test/simple/params.h test/resnet/params.h test/mobilebert/*.h
	$(CC) $(C17FLAGS) -c -o $@ $<

build/PositTest.o: test/common/PositTest.cc src/PositTypes.h
	$(CC) $(C17FLAGS) -DNO_SYSC -c -o $@ $<

.PHONY: clean rtl sim PositTest clean-catapult
clean:
	rm -rf build/*.o test_outputs/*

clean-catapult:
	rm -rf build/Catapult_*
