set block "DwCUnit"
set full_block_name "DwCUnit<$DWC_DATATYPE, $DWC_DATATYPE, $DWC_PSUM, $ACCUM_BUFFER_DATATYPE, $OC_DIMENSION, $DWC_DATATYPE>"

proc pre_libraries {} {
  solution library add {[Block] MulAddTree.v1}
}

proc pre_assembly {} {
  global full_block_name
  set full_block_name_stripped [string map {" " ""} $full_block_name]

  global DWC_DATATYPE DWC_PSUM ACCUM_BUFFER_DATATYPE
  set mul_add_tree_name "MulAddTree<$DWC_DATATYPE,$DWC_DATATYPE,$DWC_PSUM,$ACCUM_BUFFER_DATATYPE>"
  set mul_add_tree_name_stripped [string map {" " ""} $mul_add_tree_name]

  directive set /$full_block_name_stripped/$mul_add_tree_name_stripped -MAP_TO_MODULE {[Block] MulAddTree.v1}
}

proc pre_architect {} {
  global full_block_name TECHNOLOGY DWC_WIDTH SUPPORT_MX
  set full_block_name_stripped [string map {" " ""} $full_block_name]

  if {$TECHNOLOGY == "tsmc16"} {
    # TSMC Compiled base RF Size 40x128m4s for Single Port Memory
    set tsmc16_memory_width 128
    set tsmc16_memory_depth 40
    # set memory_interleave [expr $memory_width/$tsmc16_memory_width]
    set memory_interleave 2
    set memory_block_size $DWC_WIDTH
    directive set /$full_block_name_stripped/$full_block_name_stripped:input_buffer_write_addr_gen/input_buffer:rsc -INTERLEAVE $memory_interleave
    directive set /$full_block_name_stripped/$full_block_name_stripped:input_buffer_write_addr_gen/input_buffer:rsc -BLOCK_SIZE $memory_block_size
    directive set /$full_block_name_stripped/$full_block_name_stripped:input_buffer_write_addr_gen/input_buffer -WORD_WIDTH $tsmc16_memory_width 
    directive set /$full_block_name_stripped/$full_block_name_stripped:input_buffer_write_addr_gen/input_buffer:rsc -MAP_TO_MODULE [get_memory_name 0 $tsmc16_memory_depth $tsmc16_memory_width]
  } else {
    set memory_interleave 2
    set memory_block_size $DWC_WIDTH
    directive set /$full_block_name_stripped/$full_block_name_stripped:input_buffer_write_addr_gen/input_buffer:rsc -INTERLEAVE $memory_interleave
    directive set /$full_block_name_stripped/$full_block_name_stripped:input_buffer_write_addr_gen/input_buffer:rsc -BLOCK_SIZE $memory_block_size
    directive set /$full_block_name_stripped/$full_block_name_stripped:input_buffer_write_addr_gen/input_buffer:rsc -MAP_TO_MODULE {ccs_sample_mem.ccs_ram_sync_1R1W}
    if {$SUPPORT_MX == true} {
      directive set /$full_block_name_stripped/$full_block_name_stripped:input_buffer_write_addr_gen/input_scale_buffer:rsc -INTERLEAVE $memory_interleave
      directive set /$full_block_name_stripped/$full_block_name_stripped:input_buffer_write_addr_gen/input_scale_buffer:rsc -BLOCK_SIZE $memory_block_size
      directive set /$full_block_name_stripped/$full_block_name_stripped:input_buffer_write_addr_gen/input_scale_buffer:rsc -MAP_TO_MODULE {ccs_sample_mem.ccs_ram_sync_1R1W}
    }
  }
}

proc pre_extract {} {
  global SUPPORT_MX
  ignore_memory_precedences -from while:for#3:for:for:for:if#2:write_mem(input_buffer:rsc(*)(*).@) -to while:for#3:for:for:for:for:else:read_mem(input_buffer:rsc(*)(*).@)*
  if {$SUPPORT_MX == true} {
    ignore_memory_precedences -from while:for#3:for:for:for:if#2:write_mem(input_scale_buffer:rsc(*)(*).@) -to while:for#3:for:for:for:for:else:read_mem(input_scale_buffer:rsc(*)(*).@)*
  }
}
