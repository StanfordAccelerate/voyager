package voyager

import chisel3._

import freechips.rocketchip.subsystem._
import org.chipsalliance.cde.config.{Field, Config, Parameters}
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.resources.{DiplomacyUtils}
import freechips.rocketchip.prci.{ClockSinkDomain, ClockSinkParameters}
import scala.collection.immutable.{ListMap}
import testchipip.soc.{BankedScratchpadKey}


case object MaxBurstSize extends Field[Int]

class WithMaxBurstSize(numBytes: Int) extends Config((site, here, up) => {
  case MaxBurstSize => numBytes
})

class WithDefaultMaxBurstSize extends Config((site, here, up) => {
  case MaxBurstSize => up(CacheBlockBytes)
})

class ScratchpadBank(subBanks: Int, address: AddressSet, beatBytes: Int, devOverride: MemoryDevice, buffer: BufferParams)(implicit p: Parameters) extends ClockSinkDomain(ClockSinkParameters())(p) {
  val xbar = TLXbar()
  val subBankSize = (address.mask + 1) / subBanks
  println(s"[ScratchpadBank] MaxBurstSize = ${p(MaxBurstSize)}")
  (0 until subBanks).map { sb =>
    val ram = LazyModule(new TLRAM(
      address = AddressSet(address.base + sb * subBankSize, subBankSize - 1),
      atomics = true,
      beatBytes = beatBytes,
      devOverride = Some(devOverride)) {
      override lazy val desiredName = s"TLRAM_ScratchpadBank"
    })
    ram.node := TLFragmenter(beatBytes, p(MaxBurstSize), nameSuffix = Some("ScratchpadBank")) := TLBuffer(buffer) := xbar
  }
  override lazy val desiredName = "ScratchpadBank"
}

trait CanHaveBankedScratchpad { this: BaseSubsystem =>
  p(BankedScratchpadKey).zipWithIndex.foreach { case (params, si) =>
    val bus = locateTLBusWrapper(params.busWhere)

    require (params.subBanks >= 1)

    val name = params.name
    val banks = params.banks
    val size = params.size
    val bankSize = size / banks
    val device = new MemoryDevice {
      override def describe(resources: ResourceBindings): Description = {
        Description(describeName("memory", resources), ListMap(
          "reg"         -> resources.map.filterKeys(DiplomacyUtils.regFilter).flatMap(_._2).map(_.value).toList,
          "device_type" -> Seq(ResourceString("memory")),
          "status"      -> Seq(ResourceString(if (params.dtsEnabled) "okay" else "disabled"))
        ))
      }
    }

    def genBanks()(implicit p: Parameters) = (0 until banks).map { b =>
      val bank = LazyModule(new ScratchpadBank(
        params.subBanks,
        AddressSet(params.base + bankSize * b, bankSize - 1),
        bus.beatBytes,
        device,
        params.buffer))
      bank.clockNode := bus.fixedClockNode
      bus.coupleTo(s"$name-$si-$b") { bank.xbar := bus { TLBuffer(params.outerBuffer) } := _ }
    }

    if (params.disableMonitors) DisableMonitors { implicit p => genBanks()(p) } else genBanks()
  }
}
