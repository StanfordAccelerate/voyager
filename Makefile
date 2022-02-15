CC = /opt/rh/devtoolset-10/root/bin/g++
BASEFLAGS = -I/cad/mentor/2021.1/Mgc_home/shared/include/ -Ilib/ -Ilib/universal/include/ -Isrc/ -I. -DCONNECTIONS_FAST_SIM -DSC_INCLUDE_DYNAMIC_PROCESSES -DCONNECTIONS_NAMING_ORIGINAL -O3 
C11FLAGS = $(BASEFLAGS) -std=c++11
C17FLAGS = $(BASEFLAGS) -std=c++17
LDFLAGS = -lsystemc
LDLIBS = -L/cad/mentor/2021.1/Mgc_home/shared/lib/
TEST ?= simple

export TEST := $(TEST)

rtl: build/Catapult/Accelerator.v1/concat_rtl.v

build/Catapult/Accelerator.v1/concat_rtl.v: src/Accelerator.cc $(wildcard src/*.h) scripts/hls.tcl
	catapult -shell -file scripts/hls.tcl
	sed '/module CGHpart/,/endmodule/d;/module TSDN/,/endmodule/d;/^`include/d' build/Catapult/Accelerator.v1/concat_rtl.v > build/Catapult/Accelerator.v1/concat_rtl_clean.v

debug_rtl: build/Catapult_debug/Accelerator.v1/concat_rtl.v

build/Catapult_debug/Accelerator.v1/concat_rtl.v: export DEBUG=1

build/Catapult_debug/Accelerator.v1/concat_rtl.v: src/Accelerator.cc $(wildcard src/*.h) scripts/hls.tcl
	catapult -shell -file scripts/hls.tcl

sim: build/TestRunner
	./build/TestRunner 

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

build/TestRunner.o: test/common/TestRunner.cc test/mobilebert/params.h test/simple/params.h test/resnet/params.h
	$(CC) $(C17FLAGS) -c -o $@ $<

build/PositTest.o: test/common/PositTest.cc src/PositTypes.h
	$(CC) $(C17FLAGS) -c -o $@ $<

.PHONY: clean rtl sim PositTest
clean:
	rm -rf build/*.o
