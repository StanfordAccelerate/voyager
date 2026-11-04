set block CIMWeightController
set full_block_name [cim_weight_controller_template]

proc pre_architect {} {
  global IC_DIMENSION OC_DIMENSION full_block_name
  set name [string map {" " ""} $full_block_name]
  if {$IC_DIMENSION < 64 && $OC_DIMENSION < 64} {
    directive set /$name/$name:transposer/transposer/while:if:transpose_buffer:rsc \
      -MAP_TO_MODULE {[Register]}
  }
}
