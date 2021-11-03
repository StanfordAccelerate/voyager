CC = /cad/mentor/2021.1/Mgc_home/bin/g++
CPPFLAGS = -I/cad/mentor/2021.1/Mgc_home/shared/include/ -Ilib/ -Isrc/ -I. -std=c++11 -DCONNECTIONS_FAST_SIM -DSC_INCLUDE_DYNAMIC_PROCESSES -DCONNECTIONS_NAMING_ORIGINAL
LDFLAGS = -lsystemc
LDLIBS = -L/cad/mentor/2021.1/Mgc_home/shared/lib/

rtl: build/Catapult/Accelerator.v1/concat_rtl.v

build/Catapult/Accelerator.v1/concat_rtl.v: src/Accelerator.cc $(wildcard src/*.h) scripts/hls.tcl
	rm -rf build/Catapult
	catapult -shell -file scripts/hls.tcl

sim: build/SimpleTest
	./build/SimpleTest

build/SimpleTest: build/Accelerator.o build/Harness.o build/SimpleTest.o build/GoldModel.o build/Utils.o
	$(CC) -o $@ $^ $(LDLIBS) $(LDFLAGS)

build/Accelerator.o: src/Accelerator.cc $(wildcard src/*.h)
	$(CC) $(CPPFLAGS) -c -o $@ $< 

build/Harness.o: test/common/Harness.cc test/common/Harness.h $(wildcard src/*.h) 
	$(CC) $(CPPFLAGS) -c -o $@ $<

build/GoldModel.o: test/common/GoldModel.cc test/common/GoldModel.h src/ArchitectureParams.h
	$(CC) $(CPPFLAGS) -c -o $@ $<

build/Utils.o: test/common/Utils.cc test/common/Utils.h src/ArchitectureParams.h
	$(CC) $(CPPFLAGS) -c -o $@ $<

build/SimpleTest.o: test/simple/SimpleTest.cc
	$(CC) $(CPPFLAGS) -c -o $@ $<


.PHONY: clean rtl sim
clean:
	rm -rf build/*.o

