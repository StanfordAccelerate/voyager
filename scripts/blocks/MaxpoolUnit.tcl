set block "MaxpoolUnit"
set full_block_name "MaxpoolUnit<$ACCUM_DATATYPE, $IO_DATATYPE, $DIMENSION>"
set full_block_name_stripped [string map {" " ""} $full_block_name]

# proc pre_architect {} {
#   global full_block_name_stripped C_DATA_REP_NAME
#   directive set /$full_block_name_stripped/run/while:maxpool_comparator.value.$C_DATA_REP_NAME:rsc -MAP_TO_MODULE {[Register]}
# }
