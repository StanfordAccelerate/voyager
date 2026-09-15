package voyager

import chisel3._
import chisel3.util._
import chisel3.experimental.{BaseModule, IntParam}
import freechips.rocketchip.amba.axi4._
import freechips.rocketchip.subsystem.BaseSubsystem
import org.chipsalliance.cde.config.{Config, Field, Parameters}
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.regmapper.{HasRegMap, RegField}
import freechips.rocketchip.tilelink._
import freechips.rocketchip.util._
import freechips.rocketchip.interrupts.{IntSourceNode, IntSourcePortSimple}
import testchipip.serdes._

import scala.math._

trait HasPortWidth {
  def dataBits: Int
}

trait HasReadInterface extends Module with HasPortWidth {
  val io = IO(new Bundle {
    val request     = Flipped(DecoupledIO(UInt(((64 + 32)).W)))
    val response    = DecoupledIO(UInt(dataBits.W))
    val baseAddress = Input(UInt(64.W))
  })
}

class TLBurstReadInterface(
  interfaceName: String,
  dataWidth:     Int,
  fifoDepth:     Int,
)(
  implicit p:    Parameters)
    extends LazyModule {
  val node = TLClientNode(
    Seq(
      TLMasterPortParameters.v1(
        Seq(TLMasterParameters.v1(name = interfaceName, sourceId = IdRange(0, fifoDepth), requestFifo = true))
      )
    )
  )

  lazy val module = new LazyModuleImp(this) with HasReadInterface with HasPortWidth {
    override def desiredName = interfaceName

    override def dataBits = dataWidth

    val (tl, edge) = node.out(0)

    val beatBytes    = dataWidth / 8
    val beatBytesLg2 = log2Ceil(dataWidth / 8).toInt

    val sourceId     = RegInit(0.U(4.W))
    val credits      = RegInit(fifoDepth.U(8.W))
    val output_queue = Module(new Queue(UInt(dataWidth.W), fifoDepth + 4))

    class RequestMeta extends Bundle {
      val addrOffset = UInt(16.W)
      val beatCount  = UInt(16.W)
      val extraBeat  = Bool()
    }
    val req_meta_q = Module(new Queue(new RequestMeta, fifoDepth + 4))

    val rawBurstBytes = io.request.bits(95, 64)
    val addrBase      = io.request.bits(63, 0) + io.baseAddress

    // Align the address to the nearest beat boundary
    val addrOffset  = addrBase(beatBytesLg2 - 1, 0)
    val mask        = ~(beatBytes - 1).U(64.W)
    val alignedAddr = addrBase & mask
    val totalBytes  = (rawBurstBytes + addrOffset + beatBytes.U - 1.U) & mask

    // FSM registers
    val burstActive = RegInit(false.B)
    val currAddr    = RegInit(0.U(64.W))
    val remaining   = RegInit(0.U(16.W))

    // smallest power-of-2 chunk that covers the current address
    def lowestOneBit(x: UInt) = x & (~x + 1.U)
    val alignment = lowestOneBit(currAddr)
    val remLowest = lowestOneBit(remaining)
    val chunkSize = Mux(alignment < remLowest, alignment, remLowest)

    val startAlign = lowestOneBit(alignedAddr)
    val totalAlign = lowestOneBit(totalBytes)
    val firstChunk = Mux(startAlign < totalAlign, startAlign, totalAlign)

    when(tl.a.fire) {
      sourceId := sourceId + 1.U
      when(!burstActive) {
        currAddr    := alignedAddr + firstChunk
        remaining   := totalBytes - firstChunk
        burstActive := (totalBytes =/= firstChunk)
      }.otherwise {
        currAddr  := currAddr + chunkSize
        remaining := remaining - chunkSize
        when(remaining === chunkSize) {
          when(io.request.fire) {
            currAddr  := alignedAddr
            remaining := totalBytes
          }.otherwise {
            burstActive := false.B
          }
        }
      }
    }

    val address    = Mux(burstActive, currAddr, alignedAddr)
    val size       = Mux(burstActive, chunkSize, firstChunk)
    val numCredits = size >> beatBytesLg2

    tl.a.valid       := (burstActive || (io.request.valid && req_meta_q.io.enq.ready)) && credits >= numCredits
    tl.a.bits        := edge.Get(sourceId, address, Log2(size))._2
    io.request.ready := tl.a.fire && (!burstActive || remaining === chunkSize) && req_meta_q.io.enq.ready

    val rawBurstBytesRounded = (rawBurstBytes + beatBytes.U - 1.U) & mask
    val rawBeatCount         = rawBurstBytesRounded >> beatBytesLg2
    val totalBeatCount       = totalBytes >> beatBytesLg2
    val needsExtraBeat       = totalBeatCount =/= rawBeatCount

    req_meta_q.io.enq.valid           := io.request.fire
    req_meta_q.io.enq.bits.addrOffset := addrOffset
    req_meta_q.io.enq.bits.beatCount  := rawBeatCount
    req_meta_q.io.enq.bits.extraBeat  := needsExtraBeat

    val beatCount       = RegInit(0.U(16.W))
    val lastBeat        = RegInit(0.U(dataWidth.W))
    // Keep track if we took an aligned beat during unaligned read
    val isLastUnaligned = RegInit(false.B)

    val meta       = req_meta_q.io.deq.bits
    val unaligned  = meta.addrOffset =/= 0.U
    // For unaligned reads, we need an extra beat unless there's only one
    // beat of data and it doesn't cross a beat boundary
    val extraBeat  = unaligned && (meta.extraBeat || meta.beatCount =/= 1.U)
    val isLastBeat = beatCount === meta.beatCount + extraBeat - 1.U
    val offsetBits = meta.addrOffset << 3

    output_queue.io.enq.valid := false.B
    output_queue.io.enq.bits  := DontCare

    when(tl.d.fire) {
      beatCount := Mux(isLastBeat, 0.U, beatCount + 1.U)
      when(unaligned) {
        lastBeat := tl.d.bits.data
        when(meta.beatCount === 1.U && !meta.extraBeat) {
          beatCount                 := 0.U
          output_queue.io.enq.valid := true.B
          output_queue.io.enq.bits  := tl.d.bits.data >> offsetBits
        }.elsewhen(isLastBeat && !meta.extraBeat) {
          beatCount                 := 1.U
          output_queue.io.enq.valid := true.B
          output_queue.io.enq.bits  := lastBeat >> offsetBits
          isLastUnaligned           := true.B
        }.elsewhen(beatCount =/= 0.U) {
          val merged = Cat(tl.d.bits.data, lastBeat) >> offsetBits
          output_queue.io.enq.valid := true.B
          output_queue.io.enq.bits  := merged(dataWidth - 1, 0)
        }
      }.otherwise {
        output_queue.io.enq.valid := true.B
        output_queue.io.enq.bits  := tl.d.bits.data
      }
    }.elsewhen(output_queue.io.enq.ready && unaligned && meta.beatCount =/= 1.U && isLastBeat && !meta.extraBeat) {
      beatCount                 := 0.U
      output_queue.io.enq.valid := true.B
      output_queue.io.enq.bits  := lastBeat >> offsetBits
      isLastUnaligned           := false.B
    }.elsewhen(output_queue.io.enq.ready && !unaligned && isLastUnaligned) {
      output_queue.io.enq.valid := true.B
      output_queue.io.enq.bits  := lastBeat
      isLastUnaligned           := false.B
    }

    tl.d.ready  := output_queue.io.enq.ready && req_meta_q.io.deq.valid && (unaligned || !isLastUnaligned)
    io.response <> output_queue.io.deq

    req_meta_q.io.deq.ready := isLastBeat && (tl.d.fire || (meta.beatCount =/= 1.U && unaligned && !meta.extraBeat))

    credits := credits - Mux(tl.a.fire, numCredits, 0.U) + io.response.fire.asUInt +
      (tl.d.fire && unaligned && beatCount === 0.U && meta.extraBeat).asUInt
    assert(credits <= fifoDepth.U, p"Incorrect credit count for $name: credits=$credits fifoDepth=$fifoDepth")
  }
}

trait HasWriteInterface extends Module with HasPortWidth {
  val io = IO(new Bundle {
    val addr        = Flipped(DecoupledIO(UInt(32.W)))
    val data        = Flipped(DecoupledIO(UInt(dataBits.W)))
    val baseAddress = Input(UInt(64.W))
  })
}

class TLFullWriteInterface(
  name:       String,
  dataWidth:  Int,
  fifoDepth:  Int,
  maxFlight:  Int,
)(
  implicit p: Parameters)
    extends LazyModule {
  val node = TLClientNode(
    Seq(
      TLMasterPortParameters.v1(
        Seq(TLMasterParameters.v1(name = name, sourceId = IdRange(0, maxFlight), requestFifo = true))
      )
    )
  )

  lazy val module = new LazyModuleImp(this) with HasWriteInterface with HasPortWidth {
    override def dataBits = dataWidth

    val (tl, edge) = node.out(0)

    val beatBytes    = dataWidth / 8
    val beatBytesLg2 = log2Ceil(beatBytes).toInt

    val addrQueue = Module(new Queue(UInt(32.W), fifoDepth))
    val dataQueue = Module(new Queue(UInt(dataWidth.W), fifoDepth))

    addrQueue.io.enq <> io.addr
    dataQueue.io.enq <> io.data

    val sourceId = RegInit(0.U(log2Ceil(maxFlight).W))

    // FSM registers
    val isSecondPart = RegInit(false.B)
    val addr2        = RegInit(0.U(64.W))
    val size2        = RegInit(0.U(16.W))
    val data2        = RegInit(0.U(dataWidth.W))

    val addrBase = addrQueue.io.deq.bits + io.baseAddress
    val data     = dataQueue.io.deq.bits

    val addrOffset  = addrBase(beatBytesLg2 - 1, 0)
    val alignedAddr = addrBase & ~(beatBytes - 1).U(64.W)

    val bytesToWrite   = beatBytes.U - addrOffset
    val firstPartSize  = Mux(bytesToWrite > beatBytes.U, beatBytes.U, bytesToWrite)
    val secondPartSize = beatBytes.U - firstPartSize

    // Extract first beat
    val data1 = data << (addrOffset << 3)
    val mask1 = ((1L.U << firstPartSize) - 1.U) << addrOffset

    when(!isSecondPart && addrQueue.io.deq.fire && dataQueue.io.deq.fire) {
      addr2        := alignedAddr + beatBytes.U
      data2        := data >> (firstPartSize << 3)
      size2        := secondPartSize
      isSecondPart := addrOffset =/= 0.U && secondPartSize =/= 0.U
    }

    when(isSecondPart && tl.a.fire) {
      isSecondPart := false.B
    }

    val address   = Mux(isSecondPart, addr2, alignedAddr)
    val writeData = Mux(isSecondPart, data2, data1)
    val size      = Mux(isSecondPart, size2, firstPartSize)
    val mask      = Mux(isSecondPart, (1L.U << size2) - 1.U, mask1)

    tl.a.bits  := edge.Put(sourceId, address, Log2(size), writeData, mask)._2
    tl.a.valid := addrQueue.io.deq.valid && dataQueue.io.deq.valid || isSecondPart

    addrQueue.io.deq.ready := tl.a.fire && !isSecondPart
    dataQueue.io.deq.ready := tl.a.fire && !isSecondPart

    // don't really care about the responses to the writes
    tl.d.ready := true.B

    when(tl.a.fire) {
      sourceId := sourceId + 1.U
    }
  }
}
