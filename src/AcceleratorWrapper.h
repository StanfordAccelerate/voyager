#pragma once

#include <mc_connections.h>
#include <mc_scverify.h>

#include "Accelerator.h"

SC_MODULE(AcceleratorWrapper) {
  Accelerator CCS_INIT_S1(accelerator);

  sc_clock CCS_INIT_S1(sysc_clk);

  // RTL interface
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);
  sc_in<bool> CCS_INIT_S1(serialParamsIn_vld);
  sc_out<bool> CCS_INIT_S1(serialParamsIn_rdy);
  sc_in<sc_lv<32> > CCS_INIT_S1(serialParamsIn_dat);
  sc_out<bool> CCS_INIT_S1(inputAddressRequest_vld);
  sc_in<bool> CCS_INIT_S1(inputAddressRequest_rdy);
  sc_out<sc_lv<64> > CCS_INIT_S1(inputAddressRequest_dat);
  sc_in<bool> CCS_INIT_S1(inputDataResponse_vld);
  sc_out<bool> CCS_INIT_S1(inputDataResponse_rdy);
  sc_in<sc_lv<128> > CCS_INIT_S1(inputDataResponse_dat);
  sc_out<bool> CCS_INIT_S1(weightAddressRequest_vld);
  sc_in<bool> CCS_INIT_S1(weightAddressRequest_rdy);
  sc_out<sc_lv<64> > CCS_INIT_S1(weightAddressRequest_dat);
  sc_in<bool> CCS_INIT_S1(weightDataResponse_vld);
  sc_out<bool> CCS_INIT_S1(weightDataResponse_rdy);
  sc_in<sc_lv<128> > CCS_INIT_S1(weightDataResponse_dat);
  sc_out<bool> CCS_INIT_S1(vectorFetch0AddressRequest_vld);
  sc_in<bool> CCS_INIT_S1(vectorFetch0AddressRequest_rdy);
  sc_out<sc_lv<64> > CCS_INIT_S1(vectorFetch0AddressRequest_dat);
  sc_in<bool> CCS_INIT_S1(vectorFetch0DataResponse_vld);
  sc_out<bool> CCS_INIT_S1(vectorFetch0DataResponse_rdy);
  sc_in<sc_lv<128> > CCS_INIT_S1(vectorFetch0DataResponse_dat);
  sc_out<bool> CCS_INIT_S1(vectorFetch1AddressRequest_vld);
  sc_in<bool> CCS_INIT_S1(vectorFetch1AddressRequest_rdy);
  sc_out<sc_lv<64> > CCS_INIT_S1(vectorFetch1AddressRequest_dat);
  sc_in<bool> CCS_INIT_S1(vectorFetch1DataResponse_vld);
  sc_out<bool> CCS_INIT_S1(vectorFetch1DataResponse_rdy);
  sc_in<sc_lv<128> > CCS_INIT_S1(vectorFetch1DataResponse_dat);
  sc_out<bool> CCS_INIT_S1(vectorFetch2AddressRequest_vld);
  sc_in<bool> CCS_INIT_S1(vectorFetch2AddressRequest_rdy);
  sc_out<sc_lv<64> > CCS_INIT_S1(vectorFetch2AddressRequest_dat);
  sc_in<bool> CCS_INIT_S1(vectorFetch2DataResponse_vld);
  sc_out<bool> CCS_INIT_S1(vectorFetch2DataResponse_rdy);
  sc_in<sc_lv<128> > CCS_INIT_S1(vectorFetch2DataResponse_dat);
  sc_out<bool> CCS_INIT_S1(vectorOutput_vld);
  sc_in<bool> CCS_INIT_S1(vectorOutput_rdy);
  sc_out<sc_lv<128> > CCS_INIT_S1(vectorOutput_dat);
  sc_out<bool> CCS_INIT_S1(vectorOutputAddress_vld);
  sc_in<bool> CCS_INIT_S1(vectorOutputAddress_rdy);
  sc_out<sc_lv<32> > CCS_INIT_S1(vectorOutputAddress_dat);
  sc_out<bool> CCS_INIT_S1(scalarUnitOutput_vld);
  sc_in<bool> CCS_INIT_S1(scalarUnitOutput_rdy);
  sc_out<sc_lv<128> > CCS_INIT_S1(scalarUnitOutput_dat);
  sc_out<bool> CCS_INIT_S1(scalarOutputAddress_vld);
  sc_in<bool> CCS_INIT_S1(scalarOutputAddress_rdy);
  sc_out<sc_lv<32> > CCS_INIT_S1(scalarOutputAddress_dat);
  sc_out<bool> CCS_INIT_S1(startSignal_vld);
  sc_in<bool> CCS_INIT_S1(startSignal_rdy);
  sc_out<bool> CCS_INIT_S1(doneSignal_vld);
  sc_in<bool> CCS_INIT_S1(doneSignal_rdy);

  SC_CTOR(AcceleratorWrapper)
      : sysc_clk("sysc_clk", 5, SC_NS, 2.5, 0, SC_NS, false) {
    accelerator.clk(sysc_clk);
    accelerator.rstn(rstn);

    accelerator.serialParamsIn.dat(serialParamsIn_dat);
    accelerator.serialParamsIn.rdy(serialParamsIn_rdy);
    accelerator.serialParamsIn.vld(serialParamsIn_vld);
    accelerator.inputAddressRequest.dat(inputAddressRequest_dat);
    accelerator.inputAddressRequest.rdy(inputAddressRequest_rdy);
    accelerator.inputAddressRequest.vld(inputAddressRequest_vld);
    accelerator.inputDataResponse.dat(inputDataResponse_dat);
    accelerator.inputDataResponse.rdy(inputDataResponse_rdy);
    accelerator.inputDataResponse.vld(inputDataResponse_vld);
    accelerator.weightAddressRequest.dat(weightAddressRequest_dat);
    accelerator.weightAddressRequest.rdy(weightAddressRequest_rdy);
    accelerator.weightAddressRequest.vld(weightAddressRequest_vld);
    accelerator.weightDataResponse.dat(weightDataResponse_dat);
    accelerator.weightDataResponse.rdy(weightDataResponse_rdy);
    accelerator.weightDataResponse.vld(weightDataResponse_vld);
    accelerator.vectorFetch0AddressRequest.dat(vectorFetch0AddressRequest_dat);
    accelerator.vectorFetch0AddressRequest.rdy(vectorFetch0AddressRequest_rdy);
    accelerator.vectorFetch0AddressRequest.vld(vectorFetch0AddressRequest_vld);
    accelerator.vectorFetch0DataResponse.dat(vectorFetch0DataResponse_dat);
    accelerator.vectorFetch0DataResponse.rdy(vectorFetch0DataResponse_rdy);
    accelerator.vectorFetch0DataResponse.vld(vectorFetch0DataResponse_vld);
    accelerator.vectorFetch1AddressRequest.dat(vectorFetch1AddressRequest_dat);
    accelerator.vectorFetch1AddressRequest.rdy(vectorFetch1AddressRequest_rdy);
    accelerator.vectorFetch1AddressRequest.vld(vectorFetch1AddressRequest_vld);
    accelerator.vectorFetch1DataResponse.dat(vectorFetch1DataResponse_dat);
    accelerator.vectorFetch1DataResponse.rdy(vectorFetch1DataResponse_rdy);
    accelerator.vectorFetch1DataResponse.vld(vectorFetch1DataResponse_vld);
    accelerator.vectorFetch2AddressRequest.dat(vectorFetch2AddressRequest_dat);
    accelerator.vectorFetch2AddressRequest.rdy(vectorFetch2AddressRequest_rdy);
    accelerator.vectorFetch2AddressRequest.vld(vectorFetch2AddressRequest_vld);
    accelerator.vectorFetch2DataResponse.dat(vectorFetch2DataResponse_dat);
    accelerator.vectorFetch2DataResponse.rdy(vectorFetch2DataResponse_rdy);
    accelerator.vectorFetch2DataResponse.vld(vectorFetch2DataResponse_vld);
    accelerator.vectorOutput.dat(vectorOutput_dat);
    accelerator.vectorOutput.rdy(vectorOutput_rdy);
    accelerator.vectorOutput.vld(vectorOutput_vld);
    accelerator.vectorOutputAddress.dat(vectorOutputAddress_dat);
    accelerator.vectorOutputAddress.rdy(vectorOutputAddress_rdy);
    accelerator.vectorOutputAddress.vld(vectorOutputAddress_vld);
    accelerator.scalarUnitOutput.dat(scalarUnitOutput_dat);
    accelerator.scalarUnitOutput.rdy(scalarUnitOutput_rdy);
    accelerator.scalarUnitOutput.vld(scalarUnitOutput_vld);
    accelerator.scalarOutputAddress.dat(scalarOutputAddress_dat);
    accelerator.scalarOutputAddress.rdy(scalarOutputAddress_rdy);
    accelerator.scalarOutputAddress.vld(scalarOutputAddress_vld);
    accelerator.startSignal.rdy(startSignal_rdy);
    accelerator.startSignal.vld(startSignal_vld);
    accelerator.doneSignal.rdy(doneSignal_rdy);
    accelerator.doneSignal.vld(doneSignal_vld);
  }
};
