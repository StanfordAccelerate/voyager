#include "test/common/Utils.h"

#define NO_SYSC
#include "src/ArchitectureParams.h"

int validateMapping(Tiling tiling) {
  int x0 = tiling.loops[1][tiling.x_loop_index[1]];
  int y0 = tiling.loops[1][tiling.y_loop_index[1]];
  int c0 = tiling.loops[1][tiling.reduction_loop_index[1]];
  int k0 = tiling.loops[1][tiling.weight_loop_index[1]];
  int fx = tiling.loops[1][tiling.fx_index];
  int fy = tiling.loops[1][tiling.fy_index];
  int stride = tiling.stride;

  // TODO(fpedd): Fix and re-enable these checks
  // // Input buffer
  // int input_buffer_tile_size = (x0 * stride + fx - 1) * (y0 * stride + fy -
  // 1); if (tiling.replication) {
  //   // don't check temporarily TODO(fpedd): Why not?
  //   input_buffer_tile_size = 1;
  // }
  // if (input_buffer_tile_size > INPUT_BUFFER_SIZE) {
  //   std::cerr << "ERROR: Input buffer tile size violation." << std::endl;
  //   std::cerr << "Constraint " << INPUT_BUFFER_SIZE << " but is " <<
  //   input_buffer_tile_size << std::endl; return -1;
  // }

  // Weight buffer
  // TODO(fpedd): The constraint should be c0, not 16. But this is causing
  // issues with the the last 3 conv layers of the ResNet18 model. Need to
  // investigate...
  if (fx * fy * k0 * (tiling.replication ? 3 : 16) > WEIGHT_BUFFER_SIZE) {
    std::ostringstream oss;
    oss << "ERROR: Weight buffer tile size violation." << std::endl
        << "Constraint " << WEIGHT_BUFFER_SIZE << " but is " << fx << " * "
        << fy << " * " << k0 << " * " << (tiling.replication ? 3 : 16) << " = "
        << fx * fy * k0 * (tiling.replication ? 3 : 16) << std::endl;
    spdlog::error(oss.str());

    return -1;
  }

  // Accumulation buffer
  if (x0 * y0 * k0 > ACCUM_BUFFER_SIZE) {
    std::ostringstream oss;
    oss << "ERROR: Accumulation buffer tile size violation." << std::endl
        << "Constraint " << ACCUM_BUFFER_SIZE << " but is " << x0 << " * " << y0
        << " * " << k0 << " = " << x0 * y0 * k0 << std::endl;
    spdlog::error(oss.str());
    return -1;
  }

  int x_check =
      tiling.x_loop_index[1] >= 4 ? tiling.loops[1][tiling.x_loop_index[1]] : 1;
  int y_check =
      tiling.y_loop_index[1] >= 4 ? tiling.loops[1][tiling.y_loop_index[1]] : 1;
  if (x_check * y_check < 32) {
    std::ostringstream oss;
    oss << "ERROR: Innermost X*Y must be >= 32." << std::endl
        << "X -> tiling.loops[1][" << tiling.x_loop_index[1]
        << "] = " << tiling.loops[1][tiling.x_loop_index[1]] << std::endl
        << "Y -> tiling.loops[1][" << tiling.y_loop_index[1]
        << "] = " << tiling.loops[1][tiling.y_loop_index[1]] << std::endl
        << "X*Y (with index >= 4) is " << x_check * y_check << std::endl;
    spdlog::error(oss.str());
    return -1;
  }

  if (tiling.reduction_loop_index[1] != 0) {
    spdlog::error(
        "ERROR: Input channel needs to be outermost loop of buffer "
        "level. But is {} \n",
        tiling.reduction_loop_index[1]);
    return -1;
  }

  return 0;
}
