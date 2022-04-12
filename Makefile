CC = /opt/rh/devtoolset-10/root/bin/g++
INC = -I/cad/mentor/2021.1/Mgc_home/shared/include/ -Ilib/ -Ilib/universal/include/ -Isrc/ -I. -I/usr/include/python3.6m
BASEFLAGS = $(INC) -DSC_INCLUDE_DYNAMIC_PROCESSES -DCONNECTIONS_NAMING_ORIGINAL
DEBUGFLAGS = -DDEBUG_LOG -g
FASTFLAGS = -O3

ifeq ($(DEBUG), 1)
    BASEFLAGS += $(DEBUGFLAGS)
else
    BASEFLAGS += $(FASTFLAGS)
endif

ifeq ($(FAST), 1)
    BASEFLAGS += $(DEBUGFLAGS) -DCONNECTIONS_FAST_SIM
else
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
VectorUnit: build/Catapult_VectorUnit/VectorUnit.v1/concat_rtl.v

build/Catapult_Accelerator/Accelerator.v1/concat_rtl.v: InputController WeightController MatrixProcessor VectorUnit
	catapult -shell -file scripts/Accelerator.tcl
build/Catapult_InputController/InputController.v1/concat_rtl.v:
	catapult -shell -file scripts/InputController.tcl
build/Catapult_WeightController/WeightController.v1/concat_rtl.v:
	catapult -shell -file scripts/WeightController.tcl
build/Catapult_MatrixProcessor/MatrixProcessor.v1/concat_rtl.v:
	catapult -shell -file scripts/MatrixProcessor.tcl
build/Catapult_VectorUnit/VectorUnit.v1/concat_rtl.v: src/VectorUnit.h
	catapult -shell -file scripts/VectorUnit.tcl


.PHONY: rtl InputController WeightController MatrixProcessor VectorUnit 
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
	vcs -full64 -sysc sc_main -kdb -debug_access+all -Mdir=$(build_folder) -o $(build_folder)/$(simv_name)
	./$(build_folder)/$(simv_name) -verdi

###########################################################
# Untimed C Simulations (Faster than SystemC)
###########################################################
sim: build/TestRunner
	./build/TestRunner 

MobilebertTest: build/MobilebertTest
	./build/MobilebertTest

PositTest: build/PositTest
	./build/PositTest

rtl_sim_clean:
	rm -rf build/Catapult_debug/Accelerator.v1/scverify/concat_sim_rtl_v_vcs

rtl_sim: debug_rtl
	cd build/Catapult_debug/Accelerator.v1 && make -f ./scverify/Verify_concat_sim_rtl_v_vcs.mk SIMTOOL=vcs sim

rtl_sim_debug: debug_rtl
	rm -rf build/inter.fsdb*
	cd build/Catapult_debug/Accelerator.v1 && make -f ./scverify/Verify_concat_sim_rtl_v_vcs.mk SIMTOOL=vcs simgui

gui:
	catapult build/Catapult_debug

build/TestRunner: build/Accelerator.o build/Harness.o build/TestRunner.o build/GoldModel.o build/Utils.o build/DataLoader.o
	$(CC) -o $@ $^ $(LDLIBS) $(LDFLAGS)

build/MobilebertTest: build/Accelerator.o build/Harness.o build/MobilebertTest.o build/GoldModel.o build/Utils.o build/DataLoader.o
	$(CC) -o $@ $^ $(LDLIBS) $(LDFLAGS)

build/PositTest: build/PositTest.o
	$(CC) -o $@ $^

build/Accelerator.o: src/Accelerator.cc $(wildcard src/*.h)
	$(CC) $(C11FLAGS) -c -o $@ $< 

build/Harness.o: test/common/Harness.cc test/common/Harness.h $(wildcard src/*.h) 
	$(CC) $(C11FLAGS) -c -o $@ $<

build/GoldModel.o: test/common/GoldModel.cc test/common/GoldModel.h src/ArchitectureParams.h
	$(CC) $(C17FLAGS) -c -o $@ $<

build/Utils.o: test/common/Utils.cc test/common/Utils.h src/ArchitectureParams.h
	$(CC) $(C17FLAGS) -c -o $@ $<

build/DataLoader.o: test/common/DataLoader.cc test/common/DataLoader.h src/ArchitectureParams.h
	$(CC) $(C17FLAGS) -c -o $@ $<

build/TestRunner.o: test/common/TestRunner.cc test/simple/params.h test/resnet/params.h
	$(CC) $(C17FLAGS) -c -o $@ $<

build/MobilebertTest.o: test/mobilebert/TestRunner.cc test/mobilebert/params.h test/mobilebert/training.h
	$(CC) $(C17FLAGS) -c -o $@ $<

build/PositTest.o: test/common/PositTest.cc src/PositTypes.h
	$(CC) $(C17FLAGS) -c -o $@ $<

.PHONY: clean rtl sim PositTest clean-catapult
clean:
	rm -rf build/*.o

clean-catapult:
	rm -rf build/Catapult_*
