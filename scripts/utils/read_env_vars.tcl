# Read and set relevant environment variables
# Make supplies the parameter list and native compiler defines from config.mk.
# Keep direct Catapult invocations compatible with the existing environment.
set ARCHITECTURE_PARAMS {
  IC_DIMENSION OC_DIMENSION IC_PORT_WIDTH OC_PORT_WIDTH
  INPUT_BUFFER_SIZE WEIGHT_BUFFER_SIZE ACCUM_BUFFER_SIZE
  DOUBLE_BUFFERED_ACCUM_BUFFER SUPPORT_MVM SUPPORT_SPMM SUPPORT_DWC
  MATRIX_BACKEND CIM_MACRO_INPUT_LANES CIM_MACRO_OUTPUT_LANES CIM_WEIGHT_SETS
  CIM_BASE_A_WIDTH CIM_BASE_B_WIDTH CIM_BASE_C_WIDTH
  CIM_MACRO_WRITE_INPUT_LANES CIM_MAC_LATENCY CIM_MODE CIM_SIGNED
  CIM_TILE_INPUT_AXIS_ELEMENTS CIM_TILE_OUTPUT_AXIS_ELEMENTS
  CIM_INPUT_AXIS_TILES CIM_OUTPUT_AXIS_TILES CIM_A_PORT_TILES CIM_B_PORT_TILES
  CIM_C_PORT_TILES CIM_C_BEAT_LAYOUT CIM_ARRAY_RESULT_SLOTS
  CIM_LOCAL_ACCUM_CONTEXTS CLOCK_PERIOD
}
if {[info exists ::env(ARCHITECTURE_PARAMS)]} {
  set ARCHITECTURE_PARAMS $::env(ARCHITECTURE_PARAMS)
}
foreach var {BLOCK CLOCK_PERIOD DATATYPE IC_DIMENSION OC_DIMENSION TECHNOLOGY CATAPULT_BUILD_DIR PROJ_ROOT} {
  if {![info exists ::env($var)] || $::env($var) eq ""} {
    error "Required environment variable $var is unset"
  }
  set $var $::env($var)
}
foreach var $ARCHITECTURE_PARAMS {
  if {[info exists ::env($var)] && $::env($var) ne ""} {
    set $var $::env($var)
  }
}
set ROOT [file normalize $PROJ_ROOT]
