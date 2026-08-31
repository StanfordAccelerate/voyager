package voyager

import sys.process._

import chisel3._
import chisel3.util._
import chisel3.experimental.{BaseModule, IntParam}
import freechips.rocketchip.amba.axi4._
import freechips.rocketchip.subsystem.{BaseSubsystem, PBUS, SBUS}
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.regmapper.{HasRegMap, RegField, RegFieldDesc}
import freechips.rocketchip.resources.SimpleDevice
import freechips.rocketchip.subsystem.SystemBusKey
import freechips.rocketchip.tilelink._
import freechips.rocketchip.util._
import freechips.rocketchip.prci.{ClockSinkDomain, ClockSinkParameters}
import freechips.rocketchip.interrupts.{IntSourceNode, IntSourcePortSimple}
import org.chipsalliance.cde.config.{Config, Field, Parameters}
import org.chipsalliance.diplomacy.lazymodule.LazyModule
import scala.collection.immutable.ListMap
import scala.collection.mutable.ArrayBuffer

import testchipip.serdes._
case class VoyagerParams(
  mmioAddress:               BigInt  = 0x20000000L,
  datatype:                  String  = "E4M3",
  icDimension:               Int     = 16,
  ocDimension:               Int     = 16,
  inputBufferSize:           Int     = 1024,
  weightBufferSize:          Int     = 1024,
  accumBufferSize:           Int     = 1024,
  clockPeriod:               Double  = 5,
  technology:                String  = "generic",
  scaleDatatype:             String  = "",
  doubleBufferedAccumBuffer: Boolean = false,
  supportMVM:                Boolean = false,
  supportSpmm:               Boolean = false)

case object VoyagerKey extends Field[Option[VoyagerParams]](None)

class BaseVoyagerIO(
  bitWidth:      Int,
  icDimension:   Int,
  ocDimension:   Int,
  scaleBitWidth: Int)
    extends Bundle {
  val clk  = Input(Clock())
  val rstn = Input(Bool())

  val matrix_unit_params_in_vld = Input(Bool())
  val matrix_unit_params_in_rdy = Output(Bool())
  val matrix_unit_params_in_dat = Input(UInt(64.W))

  val matrix_unit_input_req_vld  = Output(Bool())
  val matrix_unit_input_req_rdy  = Input(Bool())
  val matrix_unit_input_req_dat  = Output(UInt(96.W))
  val matrix_unit_input_resp_vld = Input(Bool())
  val matrix_unit_input_resp_rdy = Output(Bool())
  val matrix_unit_input_resp_dat = Input(UInt((bitWidth * icDimension).W))

  val matrix_unit_weight_req_vld  = Output(Bool())
  val matrix_unit_weight_req_rdy  = Input(Bool())
  val matrix_unit_weight_req_dat  = Output(UInt(96.W))
  val matrix_unit_weight_resp_vld = Input(Bool())
  val matrix_unit_weight_resp_rdy = Output(Bool())
  val matrix_unit_weight_resp_dat = Input(UInt((bitWidth * ocDimension).W))

  val matrix_unit_bias_req_vld  = Output(Bool())
  val matrix_unit_bias_req_rdy  = Input(Bool())
  val matrix_unit_bias_req_dat  = Output(UInt(96.W))
  val matrix_unit_bias_resp_vld = Input(Bool())
  val matrix_unit_bias_resp_rdy = Output(Bool())
  val matrix_unit_bias_resp_dat = Input(UInt((bitWidth * ocDimension).W))

  val matrix_unit_output_data_vld = Output(Bool())
  val matrix_unit_output_data_rdy = Input(Bool())
  val matrix_unit_output_data_dat = Output(UInt((bitWidth * ocDimension).W))
  val matrix_unit_output_addr_vld = Output(Bool())
  val matrix_unit_output_addr_rdy = Input(Bool())
  val matrix_unit_output_addr_dat = Output(UInt(32.W))

  val matrix_unit_start_vld = Output(Bool())
  val matrix_unit_start_rdy = Input(Bool())
  val matrix_unit_done_vld  = Output(Bool())
  val matrix_unit_done_rdy  = Input(Bool())

  val vector_unit_params_in_vld = Input(Bool());
  val vector_unit_params_in_rdy = Output(Bool());
  val vector_unit_params_in_dat = Input(UInt(64.W));

  val vector_fetch_0_req_rdy  = Input(Bool())
  val vector_fetch_0_req_vld  = Output(Bool())
  val vector_fetch_0_req_dat  = Output(UInt(96.W))
  val vector_fetch_0_resp_rdy = Output(Bool())
  val vector_fetch_0_resp_vld = Input(Bool())
  val vector_fetch_0_resp_dat = Input(UInt((bitWidth * ocDimension).W))

  val vector_fetch_1_req_rdy  = Input(Bool())
  val vector_fetch_1_req_vld  = Output(Bool())
  val vector_fetch_1_req_dat  = Output(UInt(96.W))
  val vector_fetch_1_resp_rdy = Output(Bool())
  val vector_fetch_1_resp_vld = Input(Bool())
  val vector_fetch_1_resp_dat = Input(UInt((bitWidth * ocDimension).W))

  val vector_fetch_2_req_rdy  = Input(Bool())
  val vector_fetch_2_req_vld  = Output(Bool())
  val vector_fetch_2_req_dat  = Output(UInt(96.W))
  val vector_fetch_2_resp_rdy = Output(Bool())
  val vector_fetch_2_resp_vld = Input(Bool())
  val vector_fetch_2_resp_dat = Input(UInt((bitWidth * ocDimension).W))

  val vector_output_data_vld = Output(Bool())
  val vector_output_data_rdy = Input(Bool())
  val vector_output_data_dat = Output(UInt((bitWidth * ocDimension).W))
  val vector_output_addr_vld = Output(Bool())
  val vector_output_addr_rdy = Input(Bool())
  val vector_output_addr_dat = Output(UInt(32.W))

  val mx_scale_output_data_vld = Output(Bool())
  val mx_scale_output_data_rdy = Input(Bool())
  val mx_scale_output_data_dat = Output(UInt(scaleBitWidth.W))
  val mx_scale_output_addr_vld = Output(Bool())
  val mx_scale_output_addr_rdy = Input(Bool())
  val mx_scale_output_addr_dat = Output(UInt(32.W))

  val sparse_tensor_output_data_vld = Output(Bool())
  val sparse_tensor_output_data_rdy = Input(Bool())
  val sparse_tensor_output_data_dat = Output(UInt((bitWidth * ocDimension).W))
  val sparse_tensor_output_addr_vld = Output(Bool())
  val sparse_tensor_output_addr_rdy = Input(Bool())
  val sparse_tensor_output_addr_dat = Output(UInt(32.W))

  val vector_unit_start_vld = Output(Bool())
  val vector_unit_start_rdy = Input(Bool())
  val vector_unit_done_vld  = Output(Bool())
  val vector_unit_done_rdy  = Input(Bool())
}

class MatrixUnitMxIO(bitWidth: Int, scaleBitWidth: Int, oc: Int) extends Bundle {
  val matrix_unit_input_scale_req_rdy  = Input(Bool())
  val matrix_unit_input_scale_req_vld  = Output(Bool())
  val matrix_unit_input_scale_req_dat  = Output(UInt(96.W))
  val matrix_unit_input_scale_resp_rdy = Output(Bool())
  val matrix_unit_input_scale_resp_vld = Input(Bool())
  val matrix_unit_input_scale_resp_dat = Input(UInt(scaleBitWidth.W))

  val matrix_unit_weight_scale_req_rdy  = Input(Bool())
  val matrix_unit_weight_scale_req_vld  = Output(Bool())
  val matrix_unit_weight_scale_req_dat  = Output(UInt(96.W))
  val matrix_unit_weight_scale_resp_rdy = Output(Bool())
  val matrix_unit_weight_scale_resp_vld = Input(Bool())
  val matrix_unit_weight_scale_resp_dat = Input(UInt((bitWidth * oc).W))
}

class MatrixVectorUnitIO(bitWidth: Int, oc: Int) extends Bundle {
  val matrix_vector_unit_params_in_rdy = Output(Bool())
  val matrix_vector_unit_params_in_vld = Input(Bool())
  val matrix_vector_unit_params_in_dat = Input(UInt(64.W))

  val matrix_vector_unit_input_req_rdy  = Input(Bool())
  val matrix_vector_unit_input_req_vld  = Output(Bool())
  val matrix_vector_unit_input_req_dat  = Output(UInt(96.W))
  val matrix_vector_unit_input_resp_rdy = Output(Bool())
  val matrix_vector_unit_input_resp_vld = Input(Bool())
  val matrix_vector_unit_input_resp_dat = Input(UInt((bitWidth * oc).W))

  val matrix_vector_unit_weight_req_rdy  = Input(Bool())
  val matrix_vector_unit_weight_req_vld  = Output(Bool())
  val matrix_vector_unit_weight_req_dat  = Output(UInt(96.W))
  val matrix_vector_unit_weight_resp_rdy = Output(Bool())
  val matrix_vector_unit_weight_resp_vld = Input(Bool())
  val matrix_vector_unit_weight_resp_dat = Input(UInt((bitWidth * oc).W))

  val matrix_vector_unit_bias_req_rdy  = Input(Bool())
  val matrix_vector_unit_bias_req_vld  = Output(Bool())
  val matrix_vector_unit_bias_req_dat  = Output(UInt(96.W))
  val matrix_vector_unit_bias_resp_rdy = Output(Bool())
  val matrix_vector_unit_bias_resp_vld = Input(Bool())
  val matrix_vector_unit_bias_resp_dat = Input(UInt((bitWidth * oc).W))

  val matrix_vector_unit_weight_dq_scale_req_rdy  = Input(Bool())
  val matrix_vector_unit_weight_dq_scale_req_vld  = Output(Bool())
  val matrix_vector_unit_weight_dq_scale_req_dat  = Output(UInt(96.W))
  val matrix_vector_unit_weight_dq_scale_resp_rdy = Output(Bool())
  val matrix_vector_unit_weight_dq_scale_resp_vld = Input(Bool())
  val matrix_vector_unit_weight_dq_scale_resp_dat = Input(UInt((bitWidth * oc).W))

  val matrix_vector_unit_weight_dq_zp_req_rdy  = Input(Bool())
  val matrix_vector_unit_weight_dq_zp_req_vld  = Output(Bool())
  val matrix_vector_unit_weight_dq_zp_req_dat  = Output(UInt(96.W))
  val matrix_vector_unit_weight_dq_zp_resp_rdy = Output(Bool())
  val matrix_vector_unit_weight_dq_zp_resp_vld = Input(Bool())
  val matrix_vector_unit_weight_dq_zp_resp_dat = Input(UInt((bitWidth * oc).W))

  val matrix_vector_unit_start_vld = Output(Bool())
  val matrix_vector_unit_start_rdy = Input(Bool())
  val matrix_vector_unit_done_vld  = Output(Bool())
  val matrix_vector_unit_done_rdy  = Input(Bool())
}

class MatrixVectorUnitMxIO(bitWidth: Int, scaleBitWidth: Int, oc: Int) extends Bundle {
  val matrix_vector_unit_input_scale_req_rdy  = Input(Bool())
  val matrix_vector_unit_input_scale_req_vld  = Output(Bool())
  val matrix_vector_unit_input_scale_req_dat  = Output(UInt(96.W))
  val matrix_vector_unit_input_scale_resp_rdy = Output(Bool())
  val matrix_vector_unit_input_scale_resp_vld = Input(Bool())
  val matrix_vector_unit_input_scale_resp_dat = Input(UInt((scaleBitWidth * 2).W))

  val matrix_vector_unit_weight_scale_req_rdy  = Input(Bool())
  val matrix_vector_unit_weight_scale_req_vld  = Output(Bool())
  val matrix_vector_unit_weight_scale_req_dat  = Output(UInt(96.W))
  val matrix_vector_unit_weight_scale_resp_rdy = Output(Bool())
  val matrix_vector_unit_weight_scale_resp_vld = Input(Bool())
  val matrix_vector_unit_weight_scale_resp_dat = Input(UInt((scaleBitWidth * 2).W))
}

class SpMMUnitIO(bitWidth: Int, scaleBitWidth: Int, oc: Int) extends Bundle {
  val spmm_unit_params_in_rdy = Output(Bool())
  val spmm_unit_params_in_vld = Input(Bool())
  val spmm_unit_params_in_dat = Input(UInt(64.W))

  val spmm_unit_input_indptr_req_rdy  = Input(Bool())
  val spmm_unit_input_indptr_req_vld  = Output(Bool())
  val spmm_unit_input_indptr_req_dat  = Output(UInt(96.W))
  val spmm_unit_input_indptr_resp_rdy = Output(Bool())
  val spmm_unit_input_indptr_resp_vld = Input(Bool())
  val spmm_unit_input_indptr_resp_dat = Input(UInt((bitWidth * oc).W))

  val spmm_unit_input_indices_req_rdy  = Input(Bool())
  val spmm_unit_input_indices_req_vld  = Output(Bool())
  val spmm_unit_input_indices_req_dat  = Output(UInt(96.W))
  val spmm_unit_input_indices_resp_rdy = Output(Bool())
  val spmm_unit_input_indices_resp_vld = Input(Bool())
  val spmm_unit_input_indices_resp_dat = Input(UInt((bitWidth * oc).W))

  val spmm_unit_input_data_req_rdy  = Input(Bool())
  val spmm_unit_input_data_req_vld  = Output(Bool())
  val spmm_unit_input_data_req_dat  = Output(UInt(96.W))
  val spmm_unit_input_data_resp_rdy = Output(Bool())
  val spmm_unit_input_data_resp_vld = Input(Bool())
  val spmm_unit_input_data_resp_dat = Input(UInt((bitWidth * oc).W))

  val spmm_unit_weight_req_rdy  = Input(Bool())
  val spmm_unit_weight_req_vld  = Output(Bool())
  val spmm_unit_weight_req_dat  = Output(UInt(96.W))
  val spmm_unit_weight_resp_rdy = Output(Bool())
  val spmm_unit_weight_resp_vld = Input(Bool())
  val spmm_unit_weight_resp_dat = Input(UInt((bitWidth * oc).W))

  val spmm_unit_weight_scale_req_rdy  = Input(Bool())
  val spmm_unit_weight_scale_req_vld  = Output(Bool())
  val spmm_unit_weight_scale_req_dat  = Output(UInt(96.W))
  val spmm_unit_weight_scale_resp_rdy = Output(Bool())
  val spmm_unit_weight_scale_resp_vld = Input(Bool())
  val spmm_unit_weight_scale_resp_dat = Input(UInt((bitWidth * oc).W))

  val spmm_unit_start_vld = Output(Bool())
  val spmm_unit_start_rdy = Input(Bool())
  val spmm_unit_done_vld  = Output(Bool())
  val spmm_unit_done_rdy  = Input(Bool())
}

class VoyagerIO(
  bitWidth:      Int,
  icDimension:   Int,
  ocDimension:   Int,
  scaleBitWidth: Int,
  supportMX:     Boolean,
  supportMVM:    Boolean,
  supportSpmm:   Boolean)
    extends Record {

  private val baseBundle = new BaseVoyagerIO(bitWidth, icDimension, ocDimension, scaleBitWidth)
  private val mxBundle   = if (supportMX) Some(new MatrixUnitMxIO(bitWidth, scaleBitWidth, ocDimension)) else None
  private val mvmBundle  = if (supportMVM) Some(new MatrixVectorUnitIO(bitWidth, ocDimension)) else None
  private val mmBundle   =
    if (supportMX && supportMVM) Some(new MatrixVectorUnitMxIO(bitWidth, scaleBitWidth, ocDimension))
    else None
  private val spmmBundle = if (supportSpmm) Some(new SpMMUnitIO(bitWidth, scaleBitWidth, ocDimension)) else None

  override val elements: ListMap[String, Data] = {
    ListMap.from(baseBundle.elements) ++
      mxBundle.map(b => ListMap.from(b.elements)).getOrElse(ListMap.empty) ++
      mvmBundle.map(b => ListMap.from(b.elements)).getOrElse(ListMap.empty) ++
      mmBundle.map(b => ListMap.from(b.elements)).getOrElse(ListMap.empty) ++
      spmmBundle.map(b => ListMap.from(b.elements)).getOrElse(ListMap.empty)
  }
}

class Accelerator(
  datatype:                  String,
  icDimension:               Int,
  ocDimension:               Int,
  inputBufferSize:           Int,
  weightBufferSize:          Int,
  accumBufferSize:           Int,
  clockPeriod:               Double,
  technology:                String,
  doubleBufferedAccumBuffer: Boolean = false,
  supportMVM:                Boolean = false,
  supportSpmm:               Boolean = false,
  bitWidth:                  Int,
  scaleBitWidth:             Int,
  supportMX:                 Boolean)
    extends BlackBox
    with HasBlackBoxPath {

  val io = IO(
    new VoyagerIO(
      bitWidth,
      icDimension,
      ocDimension,
      scaleBitWidth,
      supportMX,
      supportMVM,
      supportSpmm,
    )
  )

  val chipyardDir = System.getProperty("user.dir")
  val voyagerDir  = s"$chipyardDir/generators/voyager"

  // Call Voyager to generate RTL
  val makeStr = s"""make -C $voyagerDir -j rtl
                   | DATATYPE=$datatype
                   | IC_DIMENSION=$icDimension
                   | OC_DIMENSION=$ocDimension
                   | CLOCK_PERIOD=$clockPeriod
                   | TECHNOLOGY=$technology
                   | INPUT_BUFFER_SIZE=$inputBufferSize
                   | WEIGHT_BUFFER_SIZE=$weightBufferSize
                   | ACCUM_BUFFER_SIZE=$accumBufferSize
                   | DOUBLE_BUFFERED_ACCUM_BUFFER=$doubleBufferedAccumBuffer
                   | SUPPORT_MVM=$supportMVM
                   | SUPPORT_SPMM=$supportSpmm
                   |""".stripMargin.replace("\n", " ")

  require(makeStr.! == 0, "Failed to run RTL generation for Voyager")

  // Add the path to the generated Verilog
  val genVerilogPath = s"""$voyagerDir/build/
                          |${datatype}_${icDimension}x${ocDimension}_
                          |${inputBufferSize}x${weightBufferSize}x${accumBufferSize}_
                          |${doubleBufferedAccumBuffer}_${supportMVM}_${supportSpmm}/
                          |Catapult/${technology}/clock_${clockPeriod}/
                          |Accelerator/Accelerator.v1/concat_rtl.v
                          |""".stripMargin.replaceAll("\n", "")
  addPath(s"$genVerilogPath")

  // Get value of environment variable CONFIG
  val config = System.getenv("CONFIG")

  val shellPath = s"$voyagerDir/build/configs/${config}/config.sh"
  val shellStr  = s"""export DATATYPE=${datatype}
                    |export IC_DIMENSION=${icDimension}
                    |export OC_DIMENSION=${ocDimension}
                    |export INPUT_BUFFER_SIZE=${inputBufferSize}
                    |export WEIGHT_BUFFER_SIZE=${weightBufferSize}
                    |export ACCUM_BUFFER_SIZE=${accumBufferSize}
                    |export DOUBLE_BUFFERED_ACCUM_BUFFER=${doubleBufferedAccumBuffer}
                    |export SUPPORT_MVM=${supportMVM}
                    |export SUPPORT_SPMM=${supportSpmm}
                    |""".stripMargin
  val file      = new java.io.File(shellPath);
  if (!file.getParentFile().exists()) {
    file.getParentFile().mkdirs();
  }
  val shellFile = new java.io.PrintWriter(shellPath)
  shellFile.write(shellStr)
  shellFile.close()
}

class TLVoyager(
  address:       BigInt,
  pbusBeatBytes: Int,
  sbusBeatBytes: Int,
)(
  implicit p:    Parameters)
    extends ClockSinkDomain(ClockSinkParameters())(p) {
  override lazy val desiredName = "TLVoyager"

  val config             = p(VoyagerKey).get
  val datatypeWidth      = config.datatype match {
    case "E4M3"   => 8
    case "P8"     => 8
    case "INT8"   => 8
    case "MXINT8" => 8
    case "BF16"   => 16
    case "FP32"   => 32
    case "MXNF4"  => 4
    case _        => throw new Exception("Unknown width for datatype. Please update the datatypeWidth mapping")
  }
  val icPortWidth        = datatypeWidth * config.icDimension
  val ocPortWidth        = datatypeWidth * config.ocDimension
  val scaleDatatypeWidth = config.scaleDatatype match {
    case "E8M0" => 8
    case "E5M3" => 8
    case ""     => 0
    case _      => throw new Exception("Unknown width for scale datatype. Please update the scaleDatatypeWidth mapping")
  }
  val supportMX          = config.datatype.startsWith("MX")
  val supportMVM         = config.supportMVM

  val device       = new SimpleDevice("voyager", Seq("stanford,voyager"))
  val voyager_node = TLRegisterNode(
    address     = Seq(AddressSet(address, 0xfff)),
    device      = device,
    beatBytes   = pbusBeatBytes,
    concurrency = 1,
  )

  // Matrix Unit
  val matrix_unit_input_tl_node        = TLIdentityNode()
  val matrix_unit_weight_tl_node       = TLIdentityNode()
  val matrix_unit_bias_tl_node         = TLIdentityNode()
  val matrix_unit_input_scale_tl_node  = TLIdentityNode()
  val matrix_unit_weight_scale_tl_node = TLIdentityNode()
  val matrix_unit_output_tl_node       = TLIdentityNode()
  // Vector Unit
  val vector_fetch_0_tl_node           = TLIdentityNode()
  val vector_fetch_1_tl_node           = TLIdentityNode()
  val vector_fetch_2_tl_node           = TLIdentityNode()
  val vector_output_tl_node            = TLIdentityNode()
  val mx_scale_output_tl_node          = TLIdentityNode()
  val sparse_tensor_output_tl_node     = TLIdentityNode()
  // Matrix Vector Unit
  val mvm_unit_input_tl_node           = TLIdentityNode()
  val mvm_unit_weight_tl_node          = TLIdentityNode()
  val mvm_unit_bias_tl_node            = TLIdentityNode()
  val mvm_unit_input_scale_tl_node     = TLIdentityNode()
  val mvm_unit_weight_scale_tl_node    = TLIdentityNode()
  val mvm_unit_weight_dq_scale_tl_node = TLIdentityNode()
  val mvm_unit_weight_dq_zp_tl_node    = TLIdentityNode()
  // SpMM Unit
  val spmm_unit_input_indptr_tl_node   = TLIdentityNode()
  val spmm_unit_input_indices_tl_node  = TLIdentityNode()
  val spmm_unit_input_data_tl_node     = TLIdentityNode()
  val spmm_unit_weight_tl_node         = TLIdentityNode()
  val spmm_unit_weight_scale_tl_node   = TLIdentityNode()

  def connectInterface(sourceNode: TLNode, destNode: TLNode, portWidth: Int): Unit = {
    if (portWidth != sbusBeatBytes * 8) {
      destNode := TLFIFOFixer(TLFIFOFixer.all) := TLWidthWidget(portWidth / 8) := sourceNode
    } else {
      destNode := TLFIFOFixer(TLFIFOFixer.all) := sourceNode
    }
  }

  def makeRead(name: String, width: Int, destNode: TLNode, enabled: Boolean = true) = {
    if (enabled) {
      val interface = LazyModule(new TLBurstReadInterface(name, width, 16)(p))
      connectInterface(interface.node, destNode, width)
      interface
    } else null
  }

  def makeWrite(name: String, width: Int, destNode: TLNode, enabled: Boolean = true) = {
    if (enabled) {
      val interface = LazyModule(new TLFullWriteInterface(name, width, 16, 16)(p))
      connectInterface(interface.node, destNode, width)
      interface
    } else null
  }

  // --- Matrix Unit ---
  val matrix_unit_input_interface  = makeRead("Matrix Unit Input", icPortWidth, matrix_unit_input_tl_node)
  val matrix_unit_weight_interface = makeRead("Matrix Unit Weight", ocPortWidth, matrix_unit_weight_tl_node)
  val matrix_unit_bias_interface   = makeRead("Matrix Unit Bias", ocPortWidth, matrix_unit_bias_tl_node)

  val matrix_unit_input_scale_interface  =
    makeRead("Matrix Unit Input Scale", scaleDatatypeWidth, matrix_unit_input_scale_tl_node, supportMX)
  val matrix_unit_weight_scale_interface =
    makeRead("Matrix Unit Weight Scale", ocPortWidth, matrix_unit_weight_scale_tl_node, supportMX)

  val matrix_unit_output_interface = makeWrite("Matrix Unit Output", ocPortWidth, matrix_unit_output_tl_node)

  // --- Vector Unit ---
  val vector_fetch_0_interface       = makeRead("Vector Fetch 0", ocPortWidth, vector_fetch_0_tl_node)
  val vector_fetch_1_interface       = makeRead("Vector Fetch 1", ocPortWidth, vector_fetch_1_tl_node)
  val vector_fetch_2_interface       = makeRead("Vector Fetch 2", ocPortWidth, vector_fetch_2_tl_node)
  val vector_output_interface        = makeWrite("Vector Output", ocPortWidth, vector_output_tl_node)
  val mx_scale_output_interface      = makeWrite("MX Scale Output", 8, mx_scale_output_tl_node)
  val sparse_tensor_output_interface = makeWrite("Sparse Tensor Output", ocPortWidth, sparse_tensor_output_tl_node)

  // --- Matrix Vector Unit ---
  val mvm_unit_input_interface  = makeRead("Matrix Vector Unit Input", ocPortWidth, mvm_unit_input_tl_node, supportMVM)
  val mvm_unit_weight_interface =
    makeRead("Matrix Vector Unit Weight", ocPortWidth, mvm_unit_weight_tl_node, supportMVM)
  val mvm_unit_bias_interface   = makeRead("Matrix Vector Unit Bias", ocPortWidth, mvm_unit_bias_tl_node, supportMVM)

  val mvm_unit_input_scale_interface     = makeRead(
    "Matrix Vector Unit Input Scale",
    scaleDatatypeWidth * 2,
    mvm_unit_input_scale_tl_node,
    supportMVM && supportMX,
  )
  val mvm_unit_weight_scale_interface    = makeRead(
    "Matrix Vector Unit Weight Scale",
    scaleDatatypeWidth * 2,
    mvm_unit_weight_scale_tl_node,
    supportMVM && supportMX,
  )
  val mvm_unit_weight_dq_scale_interface = makeRead(
    "Matrix Vector Unit Weight DQ Scale",
    ocPortWidth,
    mvm_unit_weight_dq_scale_tl_node,
    supportMVM && supportMX,
  )
  val mvm_unit_weight_dq_zp_interface    = makeRead(
    "Matrix Vector Unit Weight DQ Zero Point",
    ocPortWidth,
    mvm_unit_weight_dq_zp_tl_node,
    supportMVM && supportMX,
  )

  // SpMM Unit
  val spmm_unit_input_indptr_interface  =
    makeRead("SpMM Unit Input Indptr", ocPortWidth, spmm_unit_input_indptr_tl_node, config.supportSpmm)
  val spmm_unit_input_indices_interface =
    makeRead("SpMM Unit Input Indices", ocPortWidth, spmm_unit_input_indices_tl_node, config.supportSpmm)
  val spmm_unit_input_data_interface    =
    makeRead("SpMM Unit Input Data", ocPortWidth, spmm_unit_input_data_tl_node, config.supportSpmm)
  val spmm_unit_weight_interface        =
    makeRead("SpMM Unit Weight", ocPortWidth, spmm_unit_weight_tl_node, config.supportSpmm)
  val spmm_unit_weight_scale_interface  =
    makeRead("SpMM Unit Weight Scale", ocPortWidth, spmm_unit_weight_scale_tl_node, config.supportSpmm)

  val intNode = IntSourceNode(IntSourcePortSimple())

  override lazy val module = new VoyagerImpl

  class VoyagerImpl extends Impl {
    withClockAndReset(clock, reset) {
      val config      = p(VoyagerKey).get
      val accelerator = Module(
        new Accelerator(
          config.datatype,
          config.icDimension,
          config.ocDimension,
          config.inputBufferSize,
          config.weightBufferSize,
          config.accumBufferSize,
          config.clockPeriod,
          config.technology,
          config.doubleBufferedAccumBuffer,
          config.supportMVM,
          config.supportSpmm,
          datatypeWidth,
          scaleDatatypeWidth,
          supportMX,
        )
      )

      val voyager_base_addr = Reg(UInt(64.W))
      val voyager_reg_map   = ArrayBuffer[(Int, Seq[RegField])]()

      val numSemaphores = 8
      val semaphores    = RegInit(VecInit(Seq.fill(numSemaphores)(0.U(8.W))))

      // Default wires for optional units (prevents compilation errors if they are disabled)
      val mvm_dec_sem   = WireDefault(false.B)
      val mvm_wait_id   = WireDefault(0.U(log2Ceil(numSemaphores).W))
      val mvm_inc_sem   = WireDefault(false.B)
      val mvm_signal_id = WireDefault(0.U(log2Ceil(numSemaphores).W))

      val spmm_dec_sem   = WireDefault(false.B)
      val spmm_wait_id   = WireDefault(0.U(log2Ceil(numSemaphores).W))
      val spmm_inc_sem   = WireDefault(false.B)
      val spmm_signal_id = WireDefault(0.U(log2Ceil(numSemaphores).W))

      // DMA is not implemented yet; use MMIO-driven pulse registers to emulate DMA completion/wait
      val dma_dec_sem   = RegInit(false.B)
      val dma_wait_id   = RegInit(0.U(log2Ceil(numSemaphores).W))
      val dma_inc_sem   = RegInit(false.B)
      val dma_signal_id = RegInit(0.U(log2Ceil(numSemaphores).W))

      // Auto-clear the pulse bits after one cycle so each MMIO write triggers at most one
      // semaphore update (prevents repeated inc/dec across cycles).
      when(dma_inc_sem) { dma_inc_sem := false.B }
      when(dma_dec_sem) { dma_dec_sem := false.B }

      voyager_reg_map ++= Seq(
        0x500 -> Seq(RegField(1, dma_dec_sem)),
        0x508 -> Seq(RegField(log2Ceil(numSemaphores), dma_wait_id)),
        0x510 -> Seq(RegField(1, dma_inc_sem)),
        0x518 -> Seq(RegField(log2Ceil(numSemaphores), dma_signal_id)),
      )

      val bank_0_loaded = RegInit(false.B)
      val bank_1_loaded = RegInit(false.B)

      // Inflight counters for each unit
      val matrix_op_inflight        = RegInit(0.U(2.W))
      val matrix_vector_op_inflight = RegInit(0.U(2.W))
      val spmm_op_inflight          = RegInit(0.U(2.W))
      val vector_op_inflight        = RegInit(0.U(2.W))

      val io_map = accelerator.io.elements

      def bool(name: String) = io_map(name).asInstanceOf[Bool]

      def connectInput(
        prefix:    String,
        dataWidth: Int,
        interface: TLBurstReadInterface,
        fifoDepth: Int = 4,
      ): (Queue[UInt], Queue[UInt]) = {
        val req_q  = Module(new Queue(UInt((64 + 32).W), fifoDepth))
        val resp_q = Module(new Queue(UInt(dataWidth.W), fifoDepth))

        interface.module.io.baseAddress := voyager_base_addr

        req_q.io.enq.valid           := io_map(s"${prefix}_req_vld")
        req_q.io.enq.bits            := io_map(s"${prefix}_req_dat")
        io_map(s"${prefix}_req_rdy") := req_q.io.enq.ready
        interface.module.io.request  <> req_q.io.deq

        resp_q.io.enq                 <> interface.module.io.response
        io_map(s"${prefix}_resp_dat") := resp_q.io.deq.bits
        io_map(s"${prefix}_resp_vld") := resp_q.io.deq.valid
        resp_q.io.deq.ready           := io_map(s"${prefix}_resp_rdy")

        (req_q, resp_q)
      }

      def connectOutput(
        prefix:    String,
        dataWidth: Int,
        interface: TLFullWriteInterface,
        fifoDepth: Int = 4,
      ): (Queue[UInt], Queue[UInt]) = {
        val addr_q = Module(new Queue(UInt(32.W), fifoDepth))
        val data_q = Module(new Queue(UInt(dataWidth.W), fifoDepth))

        interface.module.io.baseAddress := voyager_base_addr

        // Address channel
        addr_q.io.enq.valid           := io_map(s"${prefix}_addr_vld")
        addr_q.io.enq.bits            := io_map(s"${prefix}_addr_dat")
        io_map(s"${prefix}_addr_rdy") := addr_q.io.enq.ready
        interface.module.io.addr      <> addr_q.io.deq

        // Data channel
        data_q.io.enq.valid           := io_map(s"${prefix}_data_vld")
        data_q.io.enq.bits            := io_map(s"${prefix}_data_dat")
        io_map(s"${prefix}_data_rdy") := data_q.io.enq.ready
        interface.module.io.data      <> data_q.io.deq

        (addr_q, data_q)
      }

      io_map("clk")  := clock
      io_map("rstn") := ~(reset.asBool)

      // -------------------------------
      // Matrix Unit
      // -------------------------------

      val matrix_unit_params_in_q = Module(new Queue(UInt(64.W), 1))
      io_map("matrix_unit_params_in_vld")  := matrix_unit_params_in_q.io.deq.valid
      io_map("matrix_unit_params_in_dat")  := matrix_unit_params_in_q.io.deq.bits
      matrix_unit_params_in_q.io.deq.ready := io_map("matrix_unit_params_in_rdy")

      val (matrix_unit_input_req_q, matrix_unit_input_resp_q)   =
        connectInput("matrix_unit_input", icPortWidth, matrix_unit_input_interface)
      val (matrix_unit_weight_req_q, matrix_unit_weight_resp_q) =
        connectInput("matrix_unit_weight", ocPortWidth, matrix_unit_weight_interface)
      val (matrix_unit_bias_req_q, matrix_unit_bias_resp_q)     =
        connectInput("matrix_unit_bias", ocPortWidth, matrix_unit_bias_interface)
      val matrix_unit_output_addr_q, matrix_unit_output_data_q  =
        connectOutput("matrix_unit_output", ocPortWidth, matrix_unit_output_interface)

      if (supportMX) {
        val (matrix_unit_input_scale_req_q, matrix_unit_input_scale_resp_q)   =
          connectInput("matrix_unit_input_scale", scaleDatatypeWidth, matrix_unit_input_scale_interface)
        val (matrix_unit_weight_scale_req_q, matrix_unit_weight_scale_resp_q) =
          connectInput("matrix_unit_weight_scale", ocPortWidth, matrix_unit_weight_scale_interface)
      }

      // --- Matrix Unit Sync Config ---
      val mu_wait_en   = RegInit(false.B)
      val mu_wait_id   = RegInit(0.U(log2Ceil(numSemaphores).W))
      val mu_signal_en = RegInit(false.B)
      val mu_signal_id = RegInit(0.U(log2Ceil(numSemaphores).W))

      io_map("matrix_unit_start_rdy") := !mu_wait_en || (semaphores(mu_wait_id) > 0.U)
      io_map("matrix_unit_done_rdy")  := 1.U

      val mu_start_fire = bool("matrix_unit_start_rdy") && bool("matrix_unit_start_vld")
      val mu_dec_sem    = mu_start_fire && mu_wait_en

      when(mu_start_fire) {
        assert(matrix_op_inflight =/= 2.U, "Matrix inflight overflow!")
        matrix_op_inflight := matrix_op_inflight + 1.U
      }

      val mu_done_fire = bool("matrix_unit_done_rdy") && bool("matrix_unit_done_vld")
      val mu_inc_sem   = mu_done_fire && mu_signal_en // Trigger increment

      when(mu_done_fire) {
        assert(matrix_op_inflight =/= 0.U, "Matrix done with no inflight!")
        matrix_op_inflight := matrix_op_inflight - 1.U
      }

      val matrix_unit_running     = matrix_op_inflight =/= 0.U
      val matrix_unit_cycle_count = WideCounter(64, matrix_unit_running)

      voyager_reg_map ++= Seq(
        0x100 -> Seq(RegField.w(64, matrix_unit_params_in_q.io.enq)),
        0x108 -> Seq(RegField.r(64, matrix_unit_cycle_count.value)),
        0x110 -> Seq(RegField.r(2, matrix_op_inflight)),
        0x118 -> Seq(RegField(1, mu_wait_en)),
        0x120 -> Seq(RegField(8, mu_wait_id)),
        0x128 -> Seq(RegField(1, mu_signal_en)),
        0x130 -> Seq(RegField(8, mu_signal_id)),
      )

      // -------------------------------
      // Vector Unit
      // -------------------------------

      val vector_unit_params_in_q = Module(new Queue(UInt(64.W), 1))
      io_map("vector_unit_params_in_vld")  := vector_unit_params_in_q.io.deq.valid
      io_map("vector_unit_params_in_dat")  := vector_unit_params_in_q.io.deq.bits
      vector_unit_params_in_q.io.deq.ready := io_map("vector_unit_params_in_rdy")

      val (vector_fetch_0_req_q, vector_fetch_0_resp_q) =
        connectInput("vector_fetch_0", ocPortWidth, vector_fetch_0_interface)
      val (vector_fetch_1_req_q, vector_fetch_1_resp_q) =
        connectInput("vector_fetch_1", ocPortWidth, vector_fetch_1_interface)
      val (vector_fetch_2_req_q, vector_fetch_2_resp_q) =
        connectInput("vector_fetch_2", ocPortWidth, vector_fetch_2_interface)

      val vector_output_addr_q, vector_output_data_q     =
        connectOutput("vector_output", ocPortWidth, vector_output_interface)
      val mx_scale_output_addr_q, mx_scale_output_data_q =
        connectOutput("mx_scale_output", 8, mx_scale_output_interface)
      val sparse_output_addr_q, sparse_output_data_q     =
        connectOutput("sparse_tensor_output", ocPortWidth, sparse_tensor_output_interface)

      val vu_wait_en   = RegInit(false.B)
      val vu_wait_id   = RegInit(0.U(log2Ceil(numSemaphores).W))
      val vu_signal_en = RegInit(false.B)
      val vu_signal_id = RegInit(0.U(log2Ceil(numSemaphores).W))

      io_map("vector_unit_start_rdy") := (vector_op_inflight === 0.U) && (!vu_wait_en || (semaphores(vu_wait_id) > 0.U))
      io_map("vector_unit_done_rdy")  := 1.U

      val vu_start_fire = bool("vector_unit_start_rdy") && bool("vector_unit_start_vld")
      val vu_dec_sem    = vu_start_fire && vu_wait_en

      when(vu_start_fire) {
        assert(vector_op_inflight === 0.U, "Vector inflight overflow!")
        vector_op_inflight := vector_op_inflight + 1.U
      }

      val vu_done_fire = bool("vector_unit_done_rdy") && bool("vector_unit_done_vld")
      val vu_inc_sem   = vu_done_fire && vu_signal_en

      when(vu_done_fire) {
        assert(vector_op_inflight =/= 0.U, "Vector done with no inflight!")
        vector_op_inflight := vector_op_inflight - 1.U
      }

      val vector_unit_running     = vector_op_inflight =/= 0.U
      val vector_unit_cycle_count = WideCounter(64, vector_unit_running)

      voyager_reg_map ++= Seq(
        0x200 -> Seq(RegField.w(64, vector_unit_params_in_q.io.enq)),
        0x208 -> Seq(RegField.r(64, vector_unit_cycle_count.value)),
        0x210 -> Seq(RegField.r(2, vector_op_inflight)),
        0x218 -> Seq(RegField(1, vu_wait_en)),
        0x220 -> Seq(RegField(8, vu_wait_id)),
        0x228 -> Seq(RegField(1, vu_signal_en)),
        0x230 -> Seq(RegField(8, vu_signal_id)),
      )

      // -------------------------------
      // Matrix Vector Unit
      // -------------------------------

      if (supportMVM) {
        val mvm_unit_params_in_q = Module(new Queue(UInt(64.W), 1))
        io_map("matrix_vector_unit_params_in_vld") := mvm_unit_params_in_q.io.deq.valid
        io_map("matrix_vector_unit_params_in_dat") := mvm_unit_params_in_q.io.deq.bits
        mvm_unit_params_in_q.io.deq.ready          := io_map("matrix_vector_unit_params_in_rdy")

        val (mvm_unit_input_req_q, mvm_unit_input_resp_q)                     =
          connectInput("matrix_vector_unit_input", ocPortWidth, mvm_unit_input_interface)
        val (mvm_unit_weight_req_q, mvm_unit_weight_resp_q)                   =
          connectInput("matrix_vector_unit_weight", ocPortWidth, mvm_unit_weight_interface)
        val (mvm_unit_bias_req_q, mvm_unit_bias_resp_q)                       =
          connectInput("matrix_vector_unit_bias", ocPortWidth, mvm_unit_bias_interface)
        val (mvm_unit_weight_dq_scale_req_q, mvm_unit_weight_dq_scale_resp_q) =
          connectInput("matrix_vector_unit_weight_dq_scale", ocPortWidth, mvm_unit_weight_dq_scale_interface)
        val (mvm_unit_weight_dq_zp_req_q, mvm_unit_weight_dq_zp_resp_q)       =
          connectInput("matrix_vector_unit_weight_dq_zp", ocPortWidth, mvm_unit_weight_dq_zp_interface)

        if (supportMX) {
          val (mvm_unit_input_scale_req_q, mvm_unit_input_scale_resp_q)   =
            connectInput("matrix_vector_unit_input_scale", scaleDatatypeWidth * 2, mvm_unit_input_scale_interface)
          val (mvm_unit_weight_scale_req_q, mvm_unit_weight_scale_resp_q) =
            connectInput("matrix_vector_unit_weight_scale", scaleDatatypeWidth * 2, mvm_unit_weight_scale_interface)
        }

        val mvm_wait_en_reg   = RegInit(false.B)
        val mvm_wait_id_reg   = RegInit(0.U(log2Ceil(numSemaphores).W))
        val mvm_signal_en_reg = RegInit(false.B)
        val mvm_signal_id_reg = RegInit(0.U(log2Ceil(numSemaphores).W))

        // Wire up the registers to the global visibility wires
        mvm_wait_id   := mvm_wait_id_reg
        mvm_signal_id := mvm_signal_id_reg

        io_map("matrix_vector_unit_start_rdy") := !mvm_wait_en_reg || (semaphores(mvm_wait_id_reg) > 0.U)
        io_map("matrix_vector_unit_done_rdy")  := 1.U

        val mvm_start_fire = bool("matrix_vector_unit_start_rdy") && bool("matrix_vector_unit_start_vld")
        mvm_dec_sem := mvm_start_fire && mvm_wait_en_reg

        when(mvm_start_fire) {
          assert(matrix_vector_op_inflight =/= 2.U, "MatrixVector inflight overflow!")
          matrix_vector_op_inflight := matrix_vector_op_inflight + 1.U
        }

        val mvm_done_fire = bool("matrix_vector_unit_done_rdy") && bool("matrix_vector_unit_done_vld")
        mvm_inc_sem := mvm_done_fire && mvm_signal_en_reg

        when(mvm_done_fire) {
          assert(matrix_vector_op_inflight =/= 0.U, "MatrixVector done with no inflight!")
          matrix_vector_op_inflight := matrix_vector_op_inflight - 1.U
        }

        val mvm_unit_running     = matrix_vector_op_inflight =/= 0.U
        val mvm_unit_cycle_count = WideCounter(64, mvm_unit_running)

        voyager_reg_map ++= Seq(
          0x300 -> Seq(RegField.w(64, mvm_unit_params_in_q.io.enq)),
          0x308 -> Seq(RegField.r(64, mvm_unit_cycle_count.value)),
          0x310 -> Seq(RegField.r(2, matrix_vector_op_inflight)),
          0x318 -> Seq(RegField(1, mvm_wait_en_reg)),
          0x320 -> Seq(RegField(8, mvm_wait_id_reg)),
          0x328 -> Seq(RegField(1, mvm_signal_en_reg)),
          0x330 -> Seq(RegField(8, mvm_signal_id_reg)),
        )
      }

      // -------------------------------
      // SpMM Unit
      // -------------------------------

      if (config.supportSpmm) {
        val spmm_unit_params_in_q = Module(new Queue(UInt(64.W), 1))
        io_map("spmm_unit_params_in_vld")  := spmm_unit_params_in_q.io.deq.valid
        io_map("spmm_unit_params_in_dat")  := spmm_unit_params_in_q.io.deq.bits
        spmm_unit_params_in_q.io.deq.ready := io_map("spmm_unit_params_in_rdy")

        val (spmm_unit_input_indptr_req_q, spmm_unit_input_indptr_resp_q)   =
          connectInput("spmm_unit_input_indptr", ocPortWidth, spmm_unit_input_indptr_interface)
        val (spmm_unit_input_indices_req_q, spmm_unit_input_indices_resp_q) =
          connectInput("spmm_unit_input_indices", ocPortWidth, spmm_unit_input_indices_interface)
        val (spmm_unit_input_data_req_q, spmm_unit_input_data_resp_q)       =
          connectInput("spmm_unit_input_data", ocPortWidth, spmm_unit_input_data_interface)
        val (spmm_unit_weight_req_q, spmm_unit_weight_resp_q)               =
          connectInput("spmm_unit_weight", ocPortWidth, spmm_unit_weight_interface)

        if (supportMX) {
          val (spmm_unit_weight_scale_req_q, spmm_unit_weight_scale_resp_q) =
            connectInput("spmm_unit_weight_scale", ocPortWidth, spmm_unit_weight_scale_interface)
        }

        val spmm_wait_en_reg   = RegInit(false.B)
        val spmm_wait_id_reg   = RegInit(0.U(log2Ceil(numSemaphores).W))
        val spmm_signal_en_reg = RegInit(false.B)
        val spmm_signal_id_reg = RegInit(0.U(log2Ceil(numSemaphores).W))

        // Wire up the registers to the global visibility wires
        spmm_wait_id   := spmm_wait_id_reg
        spmm_signal_id := spmm_signal_id_reg

        io_map("spmm_unit_start_rdy") := !spmm_wait_en_reg || (semaphores(spmm_wait_id_reg) > 0.U)
        io_map("spmm_unit_done_rdy")  := 1.U

        val spmm_start_fire = bool("spmm_unit_start_rdy") && bool("spmm_unit_start_vld")
        spmm_dec_sem := spmm_start_fire && spmm_wait_en_reg

        when(spmm_start_fire) {
          assert(spmm_op_inflight =/= 2.U, "SpMM inflight overflow!")
          spmm_op_inflight := spmm_op_inflight + 1.U
        }

        val spmm_done_fire = bool("spmm_unit_done_rdy") && bool("spmm_unit_done_vld")
        spmm_inc_sem := spmm_done_fire && spmm_signal_en_reg

        when(spmm_done_fire) {
          assert(spmm_op_inflight =/= 0.U, "SpMM done with no inflight!")
          spmm_op_inflight := spmm_op_inflight - 1.U
        }

        val spmm_unit_running     = spmm_op_inflight =/= 0.U
        val spmm_unit_cycle_count = WideCounter(64, spmm_unit_running)

        voyager_reg_map ++= Seq(
          0x400 -> Seq(RegField.w(64, spmm_unit_params_in_q.io.enq)),
          0x408 -> Seq(RegField.r(64, spmm_unit_cycle_count.value)),
          0x410 -> Seq(RegField.r(2, spmm_op_inflight)),
          0x418 -> Seq(RegField(1, spmm_wait_en_reg)),
          0x420 -> Seq(RegField(8, spmm_wait_id_reg)),
          0x428 -> Seq(RegField(1, spmm_signal_en_reg)),
          0x430 -> Seq(RegField(8, spmm_signal_id_reg)),
        )
      }

      // Accelerator (either unit)
      val accel_running           = matrix_op_inflight =/= 0.U ||
        vector_op_inflight =/= 0.U ||
        matrix_vector_op_inflight =/= 0.U ||
        spmm_op_inflight =/= 0.U
      val accelerator_cycle_count = WideCounter(64, accel_running)

      for (i <- 0 until numSemaphores) {
        // Sum up all increments for semaphore 'i' in this cycle
        val incs = Seq(
          (mu_inc_sem && mu_signal_id === i.U),
          (vu_inc_sem && vu_signal_id === i.U),
          (mvm_inc_sem && mvm_signal_id === i.U),
          (spmm_inc_sem && spmm_signal_id === i.U),
          (dma_inc_sem && dma_signal_id === i.U),
        ).map(_.asUInt).reduce(_ +& _)

        // Sum up all decrements for semaphore 'i' in this cycle
        val decs = Seq(
          (mu_dec_sem && mu_wait_id === i.U),
          (vu_dec_sem && vu_wait_id === i.U),
          (mvm_dec_sem && mvm_wait_id === i.U),
          (spmm_dec_sem && spmm_wait_id === i.U),
          (dma_dec_sem && dma_wait_id === i.U),
        ).map(_.asUInt).reduce(_ +& _)

        // Update hardware counter
        semaphores(i) := semaphores(i) + incs - decs
      }

      // -------------------------------
      // Interrupts
      // -------------------------------

      val int_status = RegInit(0.U(16.W))
      val int_enable = RegInit(0.U(16.W))

      val bank_0_loaded_prev = RegNext(bank_0_loaded)
      val bank_1_loaded_prev = RegNext(bank_1_loaded)
      val accel_running_prev = RegNext(accel_running)
      val semaphores_prev    = RegNext(semaphores)

      val int_set_signals = Cat(
        semaphores.zip(semaphores_prev).map { case (curr, prev) => curr =/= prev }.asUInt, // bits 15-8
        0.U(5.W),                             // bits 7-3 (unused)
        !accel_running && accel_running_prev, // bit 2
        bank_1_loaded && !bank_1_loaded_prev, // bit 1
        bank_0_loaded && !bank_0_loaded_prev, // bit 0
      )

      val (interrupts, _) = intNode.out(0)
      interrupts(0) := (int_status & int_enable).orR

      voyager_reg_map ++= Seq(
        0x000 -> Seq(RegField(64, voyager_base_addr)),
        0x008 -> Seq(RegField.r(64, accelerator_cycle_count.value)),
        0x010 -> Seq(RegField.r(1, accel_running)),
        0x01a -> Seq(RegField(8, bank_0_loaded)),
        0x01b -> Seq(RegField(8, bank_1_loaded)),
        0x020 -> Seq(RegField.w1ToClear(16, int_status, int_set_signals)),
        0x028 -> Seq(RegField(16, int_enable)),
      )
      voyager_node.regmap(voyager_reg_map.toSeq: _*)
    }
  }
}

trait CanHavePeripheryVoyager { this: BaseSubsystem =>
  private val pbus = locateTLBusWrapper(PBUS)
  private val sbus = locateTLBusWrapper(SBUS)

  p(VoyagerKey).map { k =>
    val voyager = LazyModule(new TLVoyager(k.mmioAddress, pbus.beatBytes, sbus.beatBytes)(p))
    voyager.clockNode := sbus.fixedClockNode

    sbus.coupleFrom("voyager_matrix_unit_input") { _ := sbus { TLBuffer() } := voyager.matrix_unit_input_tl_node }
    sbus.coupleFrom("voyager_matrix_unit_weight") { _ := sbus { TLBuffer() } := voyager.matrix_unit_weight_tl_node }
    sbus.coupleFrom("voyager_matrix_unit_bias") { _ := sbus { TLBuffer() } := voyager.matrix_unit_bias_tl_node }

    if (k.datatype.startsWith("MX")) {
      sbus.coupleFrom("voyager_matrix_unit_input_scale") {
        _ := sbus { TLBuffer() } := voyager.matrix_unit_input_scale_tl_node
      }
      sbus.coupleFrom("voyager_matrix_unit_weight_scale") {
        _ := sbus { TLBuffer() } := voyager.matrix_unit_weight_scale_tl_node
      }
    }
    sbus.coupleFrom("voyager_matrix_unit_output") { _ := sbus { TLBuffer() } := voyager.matrix_unit_output_tl_node }

    sbus.coupleFrom("voyager_vector_fetch_0") { _ := sbus { TLBuffer() } := voyager.vector_fetch_0_tl_node }
    sbus.coupleFrom("voyager_vector_fetch_1") { _ := sbus { TLBuffer() } := voyager.vector_fetch_1_tl_node }
    sbus.coupleFrom("voyager_vector_fetch_2") { _ := sbus { TLBuffer() } := voyager.vector_fetch_2_tl_node }
    sbus.coupleFrom("voyager_vector_output") { _ := sbus { TLBuffer() } := voyager.vector_output_tl_node }
    sbus.coupleFrom("voyager_mx_scale_output") { _ := sbus { TLBuffer() } := voyager.mx_scale_output_tl_node }
    sbus.coupleFrom("voyager_sparse_tensor_output") { _ := sbus { TLBuffer() } := voyager.sparse_tensor_output_tl_node }

    if (k.supportMVM) {
      sbus.coupleFrom("voyager_matrix_vector_unit_input") {
        _ := sbus { TLBuffer() } := voyager.mvm_unit_input_tl_node
      }
      sbus.coupleFrom("voyager_matrix_vector_unit_weight") {
        _ := sbus { TLBuffer() } := voyager.mvm_unit_weight_tl_node
      }
      sbus.coupleFrom("voyager_matrix_vector_unit_bias") {
        _ := sbus { TLBuffer() } := voyager.mvm_unit_bias_tl_node
      }
      sbus.coupleFrom("voyager_matrix_vector_unit_weight_dq_scale") {
        _ := sbus { TLBuffer() } := voyager.mvm_unit_weight_dq_scale_tl_node
      }
      sbus.coupleFrom("voyager_matrix_vector_unit_weight_dq_zp") {
        _ := sbus { TLBuffer() } := voyager.mvm_unit_weight_dq_zp_tl_node
      }

      if (k.datatype.startsWith("MX")) {
        sbus.coupleFrom("voyager_matrix_vector_unit_input_scale") {
          _ := sbus { TLBuffer() } := voyager.mvm_unit_input_scale_tl_node
        }
        sbus.coupleFrom("voyager_matrix_vector_unit_weight_scale") {
          _ := sbus { TLBuffer() } := voyager.mvm_unit_weight_scale_tl_node
        }
      }
    }

    if (k.supportSpmm) {
      sbus.coupleFrom("voyager_spmm_unit_input_indptr") {
        _ := sbus { TLBuffer() } := voyager.spmm_unit_input_indptr_tl_node
      }
      sbus.coupleFrom("voyager_spmm_unit_input_indices") {
        _ := sbus { TLBuffer() } := voyager.spmm_unit_input_indices_tl_node
      }
      sbus.coupleFrom("voyager_spmm_unit_input_data") {
        _ := sbus { TLBuffer() } := voyager.spmm_unit_input_data_tl_node
      }
      sbus.coupleFrom("voyager_spmm_unit_weight") {
        _ := sbus { TLBuffer() } := voyager.spmm_unit_weight_tl_node
      }
      sbus.coupleFrom("voyager_spmm_unit_weight_scale") {
        _ := sbus { TLBuffer() } := voyager.spmm_unit_weight_scale_tl_node
      }
    }

    pbus.coupleTo("voyager") {
      voyager.voyager_node := TLFragmenter(pbus.beatBytes, pbus.blockBytes) := _
    }

    // connect voyager interrupt node to ibus
    ibus.fromSync := voyager.intNode
  }
}

class WithVoyager(
  datatype:                  String,
  icDimension:               Int,
  ocDimension:               Int,
  inputBufferSize:           Int,
  weightBufferSize:          Int,
  accumBufferSize:           Int,
  clockPeriod:               Double,
  technology:                String,
  scaleDatatype:             String  = "",
  doubleBufferedAccumBuffer: Boolean = false,
  supportMVM:                Boolean = false,
  supportSpmm:               Boolean = false)
    extends Config((site, here, up) => {
      case VoyagerKey => Some(
          VoyagerParams(
            datatype                  = datatype,
            icDimension               = icDimension,
            ocDimension               = ocDimension,
            inputBufferSize           = inputBufferSize,
            weightBufferSize          = weightBufferSize,
            accumBufferSize           = accumBufferSize,
            clockPeriod               = clockPeriod,
            technology                = technology,
            scaleDatatype             = scaleDatatype,
            doubleBufferedAccumBuffer = doubleBufferedAccumBuffer,
            supportMVM                = supportMVM,
            supportSpmm               = supportSpmm,
          )
        )
    })
