package voyager

import chisel3._
import chisel3.util._

class VoyagerVerification extends BlackBox with HasBlackBoxPath {
  val io = IO(new Bundle {
    val clock = Input(Clock())
    val reset = Input(Reset())
  })

  val chipyardDir       = System.getProperty("user.dir")
  val voyagerWrapperDir = s"$chipyardDir/generators/voyager/soc/src/main/resources"
  val voyagerDir        = s"$chipyardDir/generators/voyager"

  // SoC Verification Collateral
  addPath(s"$voyagerWrapperDir/vsrc/VerificationCollateral.v")
  addPath(s"$voyagerWrapperDir/csrc/VerificationCollateral.cc")
  addPath(s"$voyagerWrapperDir/csrc/SoCSimulation.cc")
  addPath(s"$voyagerWrapperDir/csrc/SoCSimulation.h")
  addPath(s"$voyagerWrapperDir/csrc/SoCMemory.cc")
  addPath(s"$voyagerWrapperDir/csrc/SoCMemory.h")

  // Accelerator Verification Collateral (new-IR toolchain)
  addPath(s"$voyagerDir/test/common/GoldModel.cc")
  addPath(s"$voyagerDir/test/common/ArrayMemory.cc")
  addPath(s"$voyagerDir/test/common/Tiling.cc")
  addPath(s"$voyagerDir/test/common/Model.cc")
  addPath(s"$voyagerDir/test/common/GraphUtils.cc")
  addPath(s"$voyagerDir/test/common/Interpreter.cc")
  addPath(s"$voyagerDir/test/common/Checker.cc")
  addPath(s"$voyagerDir/test/common/Simulation.cc")
  addPath(s"$voyagerDir/test/compiler/proto/voyager_ir.pb.cc")
  addPath(s"$voyagerDir/test/toolchain/MapOperation.cc")
  addPath(s"$voyagerDir/test/soc/ScheduleRecorder.cc")
}
